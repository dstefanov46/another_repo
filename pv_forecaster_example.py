"""
PV Forecaster - Python app for prediction of PV generation across Slovenia.
"""

__version__ = "0.1.0"

def imput_train(t_latest, t0, pv_ids, sim_name, dump=False):
    """Trains MLR models. One model for each pv_id.

    Parameters
    ----------
    t_latest : pandas.Timestamp
        First timestamp that is used for training.
    t0 : pandas.Timestamp, optional
        Last timestamp that is used for training.
    pv_ids : list
        List of pv_ids for which MLR model has to be trained.
    sim_name : str
        String representing simulation name.
    dump : bool, optional
        Denotes whether to dump results to DB (DB has to be emptied before).
        In case of True, empty tables:
        [RtData15min_SE_clean], [ss_agg_clean], [slo_agg_clean]

    Returns
    -------
    pv_df_clean : pandas.DataFrame, if dump=False
        Dataframe of series.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ["Pg_kW", "filled"]
    """

    a = '5'

    return 'bruhu'


def read_sql(server, database, username, password, query):
    """Returns data from database in a pd.DataFrame for a specified query.

    Parameters
    ----------
    server : str
    
    database : str
    
    username : str
    
    password : str
    
    query : str

    Returns
    -------
    df : pd.DataFrame
              
    """
    eng_str = fr'mssql+pymssql://{username}:{password}@{server}/{database}?charset=CP1250'
    engine = create_engine(eng_str)

    conn = engine.connect()
    conn.close()
    engine.dispose()

    df = pd.read_sql(query, engine)
    return df


def get_rt_pv_data(t_start, t_end, data_type):
    """Returns real time PV data from database from t_start to t_end.

    Parameters
    ----------
    t_start : pd.Timestamp
    
    t_end : pd.Timestamp
    
    data_type : str
        Whether to return clean or raw data.

    Returns
    -------
    rt : pd.DataFrame
              
    """
    print("Loading Real Time Data")
    
    if data_type == "raw":
        query = """SELECT * FROM [{}].[osse].[RtData15min_SE_v2]
                  where timestamp_utc >= '{}' and timestamp_utc <= '{}' and
                  pv_id in (SELECT [pv_id] FROM [DwSE].[osse].[TechDataSE_v2])

                """.format(db_raw, t_start, t_end).replace("\n", "")
        rt = read_sql(server, db_raw, username, password, query)
        rt = rt.rename({"Pg": "Pg_kW"}, axis=1)
        rt = rt.set_index(["pv_id", "timestamp_utc"]).Pg_kW
        
    elif data_type == "clean":

        query = """SELECT * FROM [{}].[osse].[RtData15min_SE_clean] \
                   where timestamp_utc >= '{}' and timestamp_utc <= '{}'
                   ;

                """.format(db_clean, t_start, t_end).replace("\n", "")

        rt = read_sql(server, db_clean, username, password, query)
        rt = rt.set_index(["pv_id", "timestamp_utc"])
    return rt
    
    
def get_pv_meta(pv_id=None, Type="selected"):
    """Returns PV meta data.

    Parameters
    ----------
    pv_id : str, optional
    
    Type : str, optional

    Returns
    -------
    pv_meta : pd.DataFrame
    
              
    """
    if Type == "selected":

        if pv_id is not None:
            query = "SELECT * FROM [{}].[osse].[pv_meta_selected] where pv_id={};".format(db_clean, pv_id)
        else: 
            query = "SELECT * FROM [{}].[osse].[pv_meta_selected]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query).set_index("pv_id")
        return df
    
    elif Type == "orig":
        query = """SELECT * FROM [{}].[osse].[TechDataSE]""".format(db_raw)

        pv_meta = read_sql(server, db_raw, username, password, query)

        pv_meta = pv_meta.rename({"MeteringPointID": "pv_id",
                                  "ratedPowerPn": "Pinst_kW"}, axis=1)
        cols = ['pv_id', 'ValidFrom', 'ValidTo', 'Pinst_kW', 'ss_name', 'dso_name']
        pv_meta = pv_meta.loc[:, cols]
        pv_meta.Pinst_kW = pv_meta.Pinst_kW.astype(np.float32)

        # select (remove PV without ss_name & dso_name)
        mask = pv_meta.ValidTo.isnull() & \
               ~pv_meta.loc[:, ["Pinst_kW", "ss_name", "dso_name"]].isnull().any(axis=1)  
        pv_meta = pv_meta.loc[mask]

        pv_meta = (pv_meta.drop_duplicates("pv_id")
                          .set_index("pv_id")
                  )
        return pv_meta


def get_weather_history(idx, ws_id):
    """Returns weather history.

    Parameters
    ----------
    idx : pd.DatetimeIndex
    
    ws_id : int
        Weather station id.

    Returns
    -------
    w : pd.DataFrame
    
              
    """
    idx = idx.sort_values()

    t_start15, t_end15 = idx[0], idx[-1]
    t_start = t_start15.floor("1h")
    t_end = t_end15.ceil("1h")

    query = """SELECT * FROM [{}].[ose].[WHystoryData_v2] \
               where timestamp_utc >= '{}' and timestamp_utc <= '{}' and WPointID={}\
               ;

            """.format(db_weather, t_start - datetime.timedelta(hours=6), t_end, ws_id).replace("\n", "")

    w = read_sql(server, db_weather, username, password, query)
    w = (w.drop(["WPointID", "timestamp_utc_received"], axis=1)
          .set_index("timestamp_utc")
        )

    temporary_idx = pd.date_range(t_start15 - datetime.timedelta(hours=6), t_end15, freq="15min")
    w = w.reindex(temporary_idx).interpolate(limit_direction="both")

    # select
    w = w.loc[w.index.isin(idx)]
    if w.isnull().any().any(): 
        raise ValueError('Weather history not available for more than 6 hours ws_id={}.'.format(ws_id))

    irr = w.GHI
    temp = w.temp
    snow = w.snow
    return irr, temp, snow

    
def check_weather_forecast(t0, horizon):
    """Check wether weather forecast is available.

    Parameters
    ----------
    t0 : pd.Timestamp
    
    horizon : int
        Forecasting horizon.

    Returns
    -------
    w_ws : bool
    
              
    """

    if horizon == 72*4:
        t0_hour = t0.floor("1h")
        idx_future = pd.date_range(t0.ceil("1h"),  # from next full hour
                                   freq="15min", 
                                   periods=69*4-3)

        query = """SELECT * FROM [{}].[ose].[WforecastData72hInAdvance_v2] \
           where timestamp_utc_created = '{}' and WPointID={}\
           ;

        """.format(db_weather, t0_hour, 100).replace("\n", "")  # only for ws_id == 100

        w_ws = read_sql(server, db_weather, username, password, query).sort_values("tplusH")

        if w_ws.shape[0] == 69: return True    
        else: False


    elif horizon == 192*4:
        t0_hour = t0.floor("d") + datetime.timedelta(hours=3)  # at 03:00 in the morning
        idx_future = pd.date_range(t0.ceil("1h"), 
                                   freq="15min", 
                                   periods=186*4-3)


        query = """SELECT * FROM [{}].[ose].[WforecastData192hInAdvance_v2] \
                   where timestamp_utc_created = '{}' and WPointID={}\
                   ;

                """.format(db_weather, t0_hour, 100).replace("\n", "")

        w_ws = read_sql(server, db_weather, username, password, query).sort_values("tplusH")
        if w_ws.shape[0] == 189: return True    
        else: False
            
            
def get_weather_forecast_all_raw(t0_hour, horizon):
    """Returns weather forecasts.

    Parameters
    ----------
    t0_hour : pd.Timestamp
    
    horizon : int
        Forecasting horizon.

    Returns
    -------
    w : pd.DataFrame
    
              
    """

    if horizon == 4*12: table_name = "WforecastData12hInAdvance"
    elif horizon == 4*72: table_name = "WforecastData72hInAdvance"
    elif horizon == 4*192: table_name = "WforecastData192hInAdvance"
        
    ws_ids = get_pv_meta().ws_0.unique()

    query = """SELECT * FROM [{}].[ose].[{}] \
               where TimeStampDataCreatedUTC >= '{}' and TimeStampDataCreatedUTC <= '{}' 
               ;

            """.format(db_weather, table_name, t0_hour - datetime.timedelta(hours=3), t0_hour).replace("\n", "")

    w = read_sql(server, db_weather, username, password, query)

    w = w.drop(["TimeStampLocal"], axis=1)  # "TimeStampDataReceivedUTC", 
    w = (w.rename({"TimeStampUTC": "timestamp_utc",
                   "TimeStampDataCreatedUTC": "timestamp_utc_created",
                   "WPointID": "ws_id"
                  }, axis=1)
                .sort_values("tplusH")
                .set_index("timestamp_utc")
                .loc[:, ["ws_id", "timestamp_utc_created", "GHI", "temp", "snow"]]
            )
    w = w.loc[w.ws_id.isin(ws_ids)]
    return w


def get_weather_forecast_all(t0, horizon):
    """Returns weather forecasts.

    Parameters
    ----------
    t0 : pd.Timestamp
    
    horizon : int
        Forecasting horizon.

    Returns
    -------
    w : pd.DataFrame
    
              
    """

    if horizon == 12*4:
        t0_hour = t0.floor("1h")
        idx_future = pd.date_range(t0, 
                                   freq="15min", 
                                   periods=horizon+1, 
                                   closed="right")

    elif horizon == 72*4:
        t0_hour = t0.floor("1h")
        idx_future = pd.date_range(t0.ceil("1h"),  
                                   freq="15min", 
                                   periods=69*4-3)

    elif horizon == 192*4:
        t0_hour = t0.floor("d") + datetime.timedelta(hours=3)  
        idx_future = pd.date_range(t0.ceil("1h"), 
                                   freq="15min", 
                                   periods=186*4-3)

    w = get_weather_forecast_all_raw(t0_hour, horizon)

    # Preprocess every WS separately
    w_out = []
    for ws_id in w.ws_id.unique():
        ws = w.loc[(w.ws_id == ws_id)]
        # chose last available forecast (relevantno za eno urne napovedi)
        ts_created_max = ws.timestamp_utc_created.max()
        ws = ws.loc[ws.timestamp_utc_created == ts_created_max, ["GHI", "temp", "snow"]]
        ws = ws.resample("15min").mean().interpolate().reindex(idx_future).interpolate()
        ws = ws.assign(ws_id = ws_id)
        w_out.append(ws)

    w_out = pd.concat(w_out).loc[:, ["ws_id", "GHI", "temp", "snow"]]
    return w_out
    

def get_ss_agg(Type):

    if Type == "clean":

        query = "SELECT * FROM [{}].[osse].[ss_agg_clean]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["ss_name", "timestamp_utc"]).sort_index()
        return df
    
    elif Type == "forecast12h":

        query = "SELECT * FROM [{}].[osse].[ss_agg_forecast12h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["ss_name", "timestamp_utc", "tplusH"]).sort_index()
        return df
    
    elif Type == "forecast72h":

        query = "SELECT * FROM [{}].[osse].[ss_agg_forecast72h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["ss_name", "timestamp_utc", "tplusH"]).sort_index()
        return df
    
    elif Type == "forecast192h":

        query = "SELECT * FROM [{}].[osse].[ss_agg_forecast192h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["ss_name", "timestamp_utc", "tplusH"]).sort_index()
        return df
    
    
def get_slo_agg(Type):
    if Type == "clean":

        query = "SELECT * FROM [{}].[osse].[slo_agg_clean]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["timestamp_utc"]).sort_index()
        return df
    
    elif Type == "forecast12h":

        query = "SELECT * FROM [{}].[osse].[slo_agg_forecast12h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["timestamp_utc", "tplusH"]).sort_index()
        return df
    
    elif Type == "forecast72h":

        query = "SELECT * FROM [{}].[osse].[slo_agg_forecast72h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["timestamp_utc", "tplusH"]).sort_index()
        return df
    
    elif Type == "forecast192h":

        query = "SELECT * FROM [{}].[osse].[slo_agg_forecast192h]".format(db_clean)

        df = read_sql(server, db_clean, username, password, query)
        df = df.set_index(["timestamp_utc", "tplusH"]).sort_index()
        return df


# ############### WRITE ###############
    
def write_rt_pv_data(df):
    """Writes Real Time data to database.

    Parameters
    ----------
    df : pd.DataFrame
              
    """
    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("RtData15min_SE_clean", 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False,
                            dtype={"pv_id": sa.VARCHAR(length=18)}
                           )
    engine.dispose()
    
def write_rt_pv_forecast(df, horizon):
    """Writes Real Time forecasts to database.

    Parameters
    ----------
    df : pd.DataFrame
    
    horizon: int
              
    """
    
    if horizon == 4*12: Type = "12h"
    elif horizon == 4*72: Type = "72h"
    elif horizon == 4*192: Type = "192h"

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("RtData15min_SE_forecast{}".format(Type), 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False,
                            dtype={"pv_id": sa.VARCHAR(length=18)}
                           )
    engine.dispose()
        
    
def write_ss_agg_forecast(df, horizon):
    """Writes SS agregate forecasts to database.

    Parameters
    ----------
    df : pd.DataFrame
    
    horizon: int
              
    """
    print("Writing SS Agg. forecasts")
    
    if horizon == 4*12: Type = "12h"
    elif horizon == 4*72: Type = "72h"
    elif horizon == 4*192: Type = "192h"

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("ss_agg_forecast{}".format(Type), 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False,
                            dtype={"ss_name": sa.VARCHAR(length=30)}
                           )
    engine.dispose()
    
def write_slo_agg_forecast(df, horizon):
    """Writes Slo. agregate forecasts to database.

    Parameters
    ----------
    df : pd.DataFrame
    
    horizon: int
              
    """
    print("Writing Slo Agg. forecasts")
    
    if horizon == 4*12: Type = "12h"
    elif horizon == 4*72: Type = "72h"
    elif horizon == 4*192: Type = "192h"

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("slo_agg_forecast{}".format(Type), 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False
                           )
    engine.dispose()
    
    
def write_ss_agg_clean(df):
    """Writes SS agregate data to database.

    Parameters
    ----------
    df : pd.DataFrame
              
    """
    print("Writing Agg. data")

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("ss_agg_clean", 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False,
                            dtype={"ss_name": sa.VARCHAR(length=30)}
                           )
    engine.dispose()
    
    
def write_slo_agg_clean(df):
    """Writes Slo. agregate forecasts to database.

    Parameters
    ----------
    df : pd.DataFrame
              
    """
    print("Writing Agg. data")

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df.reset_index().to_sql("slo_agg_clean", 
                            schema="osse", 
                            con=engine,
                            if_exists="append", 
                            index=False
                           )
    engine.dispose()

    
def write_wforecast_evaluation(df, horizon, username, password, server, db_clean):
    """Writes weather forecast evalution to database.

    Parameters
    ----------
    df : pd.DataFrame
    
    horizon: int
    
    username
              
    """
    print("Writing calculated errors for the weather forecasts")
    
    if horizon == 4*12: Type = "12h"
    elif horizon == 4*72: Type = "72h"
    elif horizon == 4*192: Type = "192h"

    eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(eng_str, fast_executemany=True)
    df = df.reset_index()
    df.drop_duplicates(['timestamp_utc', 'tplusH', 'ws_id'], inplace=True)
    df.to_sql("wforecast_evaluation{}".format(Type), 
              schema="osse", 
              con=engine,
              if_exists="append", 
              index=False
             )
    engine.dispose()
    
    
def get_weather_data(start_timestamp, stop_timestamp, server, database, username, password, horizon):
    """Returns actual measurements and forecasts for temp, GHI and snow for the specified time period.

    Parameters
    ----------
    start_timestamp : pd.Timestamp
        First timestamp for which to get the weather data.
        
    stop_timestamp : pd.Timestamp
        Up to which timestamp (excluding it) to get the weather data.
        
    server: str
    
    database: str
    
    username: str
    
    password: str
    
    horizon : int
        Forecasting horizon.

    Returns
    -------
    w_true : pd.DataFrame
        Dataframe containing the actual measurements of temp, GHI and snow during 
        a given time period.
    w_pred : pd.DataFrame
        Dataframe containing the predictions for temp, GHI and snow during a 
        given time period.           
    """
    
    ## calculate dt
    if horizon == 12:
        dt = datetime.timedelta(hours=horizon)

    elif horizon == 72:
        # because 72h horizon is actually 69h horizon
        dt = datetime.timedelta(hours=69)

    elif horizon == 192:
        # because 192h horizon is actually 201h horizon
        dt = datetime.timedelta(hours=189)

    query = f"""SELECT * FROM [DWVreme].[ose].[WHystoryData_v2] 
                WHERE timestamp_utc >= '{start_timestamp}' AND timestamp_utc < '{stop_timestamp}';"""

    # true
    w = read_sql(server, database, username, password, query)
    w = (w.drop(["timestamp_utc_received"], axis=1))
    w_true = w.rename({"WPointID": "ws_id"}, axis=1).set_index("timestamp_utc")
    print("history done!")

    # predictions
    query = """SELECT * FROM [DWVreme].[ose].[WforecastData{}hInAdvance_v2] 
                where timestamp_utc_created >= '{}' and timestamp_utc_created < '{}'""".format(horizon, 
                                                                                               start_timestamp, 
                                                                                               stop_timestamp - dt).replace("\n", "")

    w_pred = read_sql(server, database, username, password, query)
    w_pred = w_pred.rename({"WPointID": "ws_id"}, axis=1)
    w_pred = (w_pred.loc[:, ["timestamp_utc", "ws_id", "tplusH", "GHI", "temp", "snow"]]
                    .set_index(["timestamp_utc"]))

    return w_true, w_pred


