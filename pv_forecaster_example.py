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


def mask_night_hours(s_in):
    """Sets zero PV generation for night hours.

    Parameters
    ----------
    s_in : pd.Series
        Timeseries with PV generation.

    Returns
    -------
    s : pd.Series
        Timeseries with PV generation (where night hours are set to 0).
    """
    s = s_in.copy()

    idx = s.index
    mask1 = idx.month.isin([1]) & ((idx.hour < 7) | (idx.hour > 15))
    mask2 = idx.month.isin([2]) & ((idx.hour < 7) | (idx.hour > 16))
    mask34 = idx.month.isin([3, 4]) & ((idx.hour < 5) | (idx.hour > 17))
    mask567 = idx.month.isin([5, 6, 7]) & ((idx.hour < 4) | (idx.hour > 18))
    mask8 = idx.month.isin([8]) & ((idx.hour < 5) | (idx.hour > 18))
    mask9 = idx.month.isin([9]) & ((idx.hour < 5) | (idx.hour > 17))
    mask10 = idx.month.isin([10]) & ((idx.hour < 6) | (idx.hour > 16))
    mask11 = idx.month.isin([11]) & ((idx.hour < 6) | (idx.hour > 15))
    mask12 = idx.month.isin([12]) & ((idx.hour < 7) | (idx.hour > 15))

    mask_nosun = mask1 + mask2 + mask34 + mask567 + mask8 + mask9 + mask10 + mask11 + mask12

    s[mask_nosun] = 0
    return s


def get_interactions(s, dummies, poly_degree=1):
    """Calculates interactions between input numerical time series and dummy dataframe.
       Used for piece-wise MLR.

    Parameters
    ----------
    s : pd.Series
        Timeseries.
    dummies: pd.DataFrame
        Dummy dataframe.

    Returns
    -------
    df : pd.DataFrame
        DataFrame.
    """
    if s.name == None: s.name = "_"
    s1 = dummies.apply(lambda x: x * s).rename("{}".format(s.name + "_{}").format, axis=1)
    s2 = (dummies.apply(lambda x: x * s) ** 2).rename("{}".format(s.name + "2_{}").format, axis=1)
    s3 = (dummies.apply(lambda x: x * s) ** 3).rename("{}".format(s.name + "3_{}").format, axis=1)

    if poly_degree == 1: s2, s3 = None, None
    if poly_degree == 2: s3 = None

    df = pd.concat([s1, s2, s3], axis=1)
    return df


def get_X(idx, ghi_ts, temp_ts):
    """Create feature matrix X.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Time indices.
    ghi_ts : pd.Series
        GHI time series.
    temp_ts : pd.Series
        Temperature time series.

    Returns
    -------
    X  : pd.DataFrame
        Feature matrix X for all input time indices idx.
        Index: "timestamp_utc"
        Columns: ["feature_1", "feature_2", ...]
    """

    # extract features & create dummies
    month = pd.Series(idx.month.astype(str), index=idx, name="month").apply(lambda x: "m{}".format(x))
    hour = pd.Series(idx.strftime("%H:%M"), index=idx, name="hour")

    month_dummies = pd.get_dummies(month.sort_values()).reindex(idx)
    hour_dummies = pd.get_dummies(hour.sort_values()).reindex(idx)

    # in case of missing time intervals (columns in hour_dummies) add missing ones
    all_cols = pd.date_range("2020-1-1", periods=96, freq="15min").strftime("%H:%M")
    new_cols = set(all_cols) - set(hour_dummies.columns)

    if new_cols != 0:
        hour_dummies_add = pd.DataFrame(np.zeros((len(idx), len(new_cols))), columns=new_cols, index=idx)
        hour_dummies = pd.concat([hour_dummies, hour_dummies_add], axis=1).loc[:, all_cols]

    X = pd.concat([
        get_interactions(ghi_ts, hour_dummies, poly_degree=1),
        get_interactions(temp_ts, hour_dummies, poly_degree=1),

    ], axis=1)
    return X


def imput_missing_values(s, pv_id, t0, t_latest, fit_model, sim_name):
    """Imputes missing values for desired pv id. When fit_model=True, it loads
       pv_meta_selected from a file (output of notebook 03_Create_Xy-LongTerm),
       because latest pv_meta_selected is not in DB yet.

    Parameters
    ----------
    s : pd.Series
        Measurements for a single PV. Reindex is already applied in
        function imput_missing_values_all() from
        first available timestamp/value in a series until t0.
        Because reindex is already applied, timestamps are OK, whereas nans
        are in a series instead of missing values.
    pv_id : str
        PV id.
    t0 : pd.Timestamp, optional
        Last timestamp that is used for training/imputing.
    t_latest : pd.Timestamp
        First timestamp that is used for training/imputing.
    fit_model : bool
        Whether to fit new models or load already trained models
    sim_name : str
        String representing simulation name.

    Returns
    -------
    s_filled  : pd.DataFrame
        Original series with imputed missing values.
        Index: ["pv_id", "timestamp_utc"]
        Values: ["Pg_kW", "filled"]
    mlr : sklearn.linear_model.LinearRegression
        MLR model.
    """

    # get idx & nan values
    idx = s.index
    nans = s.isnull()

    # separate idx to train & missing idx
    # train set starts always with first available value/timestamp
    idx_train = idx[~nans]
    idx_missing = idx[nans]

    # fill only values from t_latest forward
    idx_missing = idx_missing[idx_missing > t_latest]

    if len(idx_missing) > 0:  # in case of missing values

        # geat weather data

        if fit_model:
            # when training, read from file because
            # the latest pv_meta_selected is not in DB yet
            file_path = f"D:\\forecaster_offline\generated_data/pv_meta_selected {sim_name}.csv"
            meta = pd.read_csv(file_path)
            meta.pv_id = meta.pv_id.astype(str)
            meta = meta.set_index("pv_id")
            ws_id = meta.loc[pv_id].ws_0

        else:  # during inference read from DB
            ws_id = get_pv_meta(pv_id).ws_0.iloc[0]

        ## get weather history for train idx
        if fit_model:
            irr_train, temp_train, snow_train = get_weather_history(idx_train, ws_id)

        else:
            irr_train, temp_train, snow_train = None, None, None

        ## get weather history for missing idx
        irr_missing, temp_missing, snow_missing = get_weather_history(idx_missing, ws_id)

        irr = pd.concat([irr_train, irr_missing])
        temp = pd.concat([temp_train, temp_missing])

        # create features for train & missing idx
        X = get_X(idx, irr, temp)

        if fit_model:
            X_train = X.loc[idx_train]
            y_train = s.loc[idx_train]

        X_missing = X.loc[idx_missing]
        y_missing = s.loc[idx_missing]

        # fit & save model
        if fit_model:
            mlr = LinearRegression().fit(X_train, y_train.to_frame())
        else:
            mlr_dict = joblib.load(folder_mod + 'MLR models {}.joblib'.format(sim_name))
            mlr = mlr_dict[pv_id]

        # predict missing values
        coefs = pd.Series(mlr.coef_.flatten(), index=X.columns)
        intercept = mlr.intercept_

        y_missing_pred = X_missing @ coefs + intercept
        y_missing_pred = y_missing_pred.clip(0).round(1).rename("P_backcast_kW")

        # replace missing values with predicted (only from latest forward)
        nans = nans[idx > t_latest]
        s_filled = s[idx > t_latest].copy()
        s_filled[nans] = y_missing_pred

    else:  # if there are no missing values
        s_filled = s[idx > t_latest].copy()
        mlr = None

    # set night hours to zero, parse & return
    s_filled = mask_night_hours(s_filled)
    s_filled = s_filled.to_frame().assign(pv_id=pv_id,
                                          filled=nans)
    s_filled.index.name = "timestamp_utc"
    s_filled = s_filled.reset_index().set_index(["pv_id", "timestamp_utc"])

    return s_filled, mlr


def imput_missing_values_all(pv_df, pv_ids, t0, t_latest, sim_name, fit_model):
    """Iterates of PVs, trains one MLR model for each series and imputs missing values.
       It also works in case pv_df is empty.

       There are two sets of ids in a function: pv_ids & pv_ids_df.
       pv_ids denote ids for which input series have to be imputed, whereas
       pv_ids_df denote ids in input raw data. There are cases, where
       there is PV in pv_ids, but there is no data available
       (when this function is used inside of function imput_inference()).

    Parameters
    ----------
    pv_df : pd.Series
        Set of raw series with missing values.
        Index: ["pv_id", "timestamp_utc"]
        Values: "Pg_kW"
    pv_ids : list
        List of pv_ids for which MLR models have to be trained and missing values have to be imputed.
    t0 : pd.Timestamp, optional
        Last timestamp that is used for training/imputing.
    t_latest : pd.Timestamp
        First timestamp that is used for training/imputing.
    sim_name : str
        String representing simulation name.
    fit_model : bool
        Whether to fit new models or load already trained models

    Returns
    -------
    pv_df_clean : pd.DataFrame, if dump=False else save to DB.
        Dataframe of series.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ["Pg_kW", "filled"]

    Saves to disk
    -------
    mlr_dict : dict
        Dictionary of MLR models.
    """

    pv_df_clean = []
    mlr_dict = {}

    # save pv ids from input raw data
    if len(pv_df) > 0:
        pv_ids_df = pv_df.reset_index().pv_id.unique()
    else:
        pv_ids_df = []

    # iterate over desired pv ids
    for i, pv_id in enumerate(pv_ids):
        if i % 50 == 0: print("Imputing: {}/{}".format(i, len(pv_ids)))

        if pv_id in pv_ids_df:  # in case there is data for desired series (almost all cases)
            s = pv_df.loc[pv_id].resample("15min").mean()

            if fit_model:
                # generate whole idx from the beginning of series
                idx = pd.date_range(s.index[0], t0, freq="15min")
            else:
                # generate idx from t_latest forward
                idx = pd.date_range(t_latest, t0, freq="15min", closed="right")

            s = s.reindex(idx)

        else:
            # in case of empty series
            # (only when function is used inside imput_inference()),
            # where you use already trained models for imputation)
            idx = pd.date_range(t_latest, t0, freq="15min", closed="right")
            s = pd.Series(np.nan, idx, dtype=np.float32)

        s = s.rename("Pg_kW")
        df_filled, mlr = imput_missing_values(s, pv_id, t0, t_latest, fit_model, sim_name)
        pv_df_clean.append(df_filled)
        mlr_dict[pv_id] = mlr

    pv_df_clean = pd.concat(pv_df_clean)

    # save models
    if fit_model:
        joblib.dump(mlr_dict, folder_mod + 'MLR models {}.joblib'.format(sim_name))

    return pv_df_clean


def imput_train(t_latest, t0, pv_ids, sim_name):
    """Trains MLR models & imputs missing values.

    Parameters
    ----------
    t_latest : pd.Timestamp
        First timestamp that is used for training/imputing.
    t0 : pd.Timestamp, optional
        Last timestamp that is used for training/imputing.
    pv_ids : list
        List of pv_ids for which MLR model has to be trained.
    sim_name : str
        String representing simulation name.

    Returns
    -------
    pv_df_clean : pd.DataFrame, if dump=False else save to DB.
        Dataframe of series.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ["Pg_kW", "filled"]
    """

    # load raw data
    t_start = t_latest
    pv_df = get_rt_pv_data(t_start=t_start,
                           t_end=t0,
                           data_type="raw")

    # fill missing values for all series
    pv_df_clean = imput_missing_values_all(pv_df,
                                           pv_ids,
                                           t0,
                                           t_latest,
                                           sim_name=sim_name,
                                           fit_model=True
                                           )
    return pv_df_clean


def imput_inference(t_latest, t0, pv_ids, sim_name, dump=True):
    """Imputs missing values.

    Parameters
    ----------
    t_latest : pd.Timestamp
        Last timestamp with clean data. In production
        t_latest is loaded from t0_log.csv.

    t0 : pd.Timestamp, optional
        Last timestamp that is used for imputing (denotes current timestamp).
    pv_ids : list
        List of pv_ids for which imputation is applied.
    sim_name : str
        String representing simulation name.
    dump : bool, optional
        Denotes whether to dump results to DB.

    Returns
    -------
    pv_df_clean : pd.DataFrame.
        Dataframe of series.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ["Pg_kW", "filled"]
    ss_agg : pd.DataFrame
        SS aggregate.
        Index: ["ss_name", "timestamp_utc"]
        Columns: ['Pg_kW', 'scaling_factor', 'Pg_scaled_MW']
    slo_agg : pd.DataFrame
        SS aggregate.
        Index: ["timestamp_utc"]
        Columns: ['Pg_kW', 'Pg_scaled_MW', 'scaling_factor']
    """

    # load raw data (whole history preceding/including t0 and after t_latest (not included))
    dt = datetime.timedelta(minutes=15)
    pv_df = get_rt_pv_data(t_start=t_latest + dt,
                           t_end=t0,
                           data_type="raw")

    # fill missing values
    pv_df_clean = imput_missing_values_all(pv_df,
                                           pv_ids,
                                           t0,
                                           t_latest,
                                           sim_name=sim_name,
                                           fit_model=False
                                           )
    ss_agg = get_ss_agg(pv_df_clean)
    slo_agg = get_slo_agg(ss_agg)

    if dump:
        write_rt_pv_data(pv_df_clean)
        write_ss_agg_clean(ss_agg)
        write_slo_agg_clean(slo_agg)

    return pv_df_clean, ss_agg, slo_agg


def get_ss_agg(pv_df_clean):
    """Generates SS aggregates from individual PVs.
       Let k denote scaling factor, Pg_all total installed power of all PVs in a set,
       Pg_rt total installed power of all Real Time PVs. Then, scaling factor k is
       calculated as k = Pg_all / Pg_rt. Final aggregates are calculated by first
       summarizing Pg of individual PVs for each SS and then multiplied by k.

    Parameters
    ----------
    pv_df_clean : pd.DataFrame.
        Dataframe of series.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ["Pg_kW", "filled"]

    Returns
    -------
    ss_agg : pd.DataFrame
        SS aggregate.
        Index: ["ss_name", "timestamp_utc"]
        Columns: ['Pg_kW', 'scaling_factor', 'Pg_scaled_MW']
    """

    # calculate total installed power of all PVs (real time & other) for each SS separately
    pv_meta_orig = get_pv_meta(pv_id=None, Type="orig")
    ss_Pinst_kW_orig = pv_meta_orig.groupby("ss_name").Pinst_kW.sum()

    # calculate total installed power of all Real Time PVs for each SS separately
    pv_meta = get_pv_meta(pv_id=None, Type="selected")
    ss_Pinst_kW = pv_meta.groupby("ss_name").Pinst_kW.sum()

    # calculate scaling factor k for each SS
    ss_Pinst_scaling = (ss_Pinst_kW_orig / ss_Pinst_kW).dropna().round(4)

    # calculate sum of individual PVs forecats per SS & add scaling factor to DataFrame
    df = pv_df_clean.reset_index()
    df = df.merge(pv_meta, left_on="pv_id", right_index=True)

    cols = ["ss_name", "timestamp_utc"]
    ss_agg = df.groupby(cols).Pg_kW.sum().reset_index()
    ss_agg = pd.concat([ss_agg,
                        ss_agg.ss_name.map(ss_Pinst_scaling).rename("scaling_factor")], axis=1)

    # scale forecasts of each SS separately
    ss_agg = ss_agg.assign(Pg_scaled_MW=((ss_agg.scaling_factor * ss_agg.Pg_kW) / 1000).round(3))
    ss_agg = ss_agg.set_index(["ss_name", "timestamp_utc"])
    return ss_agg


def get_slo_agg(ss_agg):
    """Generates aggregate for Slo. from SS (only SS with RT PVs).
       Let k denote scaling factor, Pg_all total installed power of all PVs in Slo.,
       Pg_ss_rt total installed power of PVs (only the ones connected to SS with RT PVs).
       Then, scaling factor k is calculated as k = Pg_all / Pg_ss_rt total. Final aggregate
       is calculated by first summarizing aggregates of SS (only SS with RT PVs) and then
       multiplied by k.

    Parameters
    ----------
    ss_agg : pd.DataFrame
        SS aggregate.
        Index: ["ss_name", "timestamp_utc"]
        Columns: ['Pg_kW', 'scaling_factor', 'Pg_scaled_MW']

    Returns
    -------

    slo_agg : pd.DataFrame
        SS aggregate.
        Index: ["timestamp_utc"]
        Columns: ['Pg_kW', 'Pg_scaled_MW', 'scaling_factor']
    """
    # calculate total installed power of all PVs (real time & other)
    pv_meta_orig = get_pv_meta(pv_id=None, Type="orig")
    slo_Pinst_kW_orig = pv_meta_orig.Pinst_kW.sum()

    # calculate total installed power of all PVs (only the ones on SS with RT PVs)
    pv_meta = get_pv_meta(pv_id=None, Type="selected")
    ss_with_rt = pv_meta.ss_name.unique().tolist()  # list of SS with RT PVs
    ssrt_Pinst_kW = pv_meta_orig.loc[pv_meta_orig.ss_name.isin(ss_with_rt)].Pinst_kW.sum()

    # calculate scaling factor k
    slo_Pinst_scaling = round((slo_Pinst_kW_orig / ssrt_Pinst_kW), 4)

    # calculate sum of SS (only SS with RT PVs) forecasts & scale according to slo_Pinst_scaling
    slo_agg = ss_agg.reset_index().set_index("timestamp_utc").Pg_scaled_MW.resample("15min").sum().rename(
        "Pg_kW") * 1000
    slo_agg_scaled = (slo_agg * slo_Pinst_scaling / 1000).rename("Pg_scaled_MW")

    slo_agg = pd.concat([slo_agg, slo_agg_scaled], axis=1).assign(scaling_factor=slo_Pinst_scaling)
    return slo_agg


def create_base_addresses(meta_data):
    return (meta_data['LocationAddressStreetNumber'] + ' ' +
            meta_data['LocationAddressStreetName'] + ', ' +
            meta_data['LocationAddressTownName'])


def create_requests(meta_data, request_addresses, gmaps):
    for i, idx in enumerate(request_addresses.index):
        if i % 100 == 0: print("creating pv lat/lon: {} / {}".format(i, len(request_addresses.index)))

        geocode_result = gmaps.geocode(request_addresses.loc[idx])
        try:
            mask = (geocode_result[0]['address_components'][-1]['long_name'] == 'Slovenia') | \
                   (geocode_result[0]['address_components'][-2]['long_name'] == 'Slovenia')
            if mask:
                meta_data.loc[idx, 'lon'] = geocode_result[0]["geometry"]["location"]["lng"]
                meta_data.loc[idx, 'lat'] = geocode_result[0]["geometry"]["location"]["lat"]
        except:
            pass

    return meta_data


def get_coordinates(meta_data_in, rtp_data, gmaps):
    meta_data = meta_data_in.copy()
    meta_data[['lon', 'lat']] = meta_data.loc[:, ['PositionPointX', 'PositionPointY']]
    final_meta_data = meta_data.loc[meta_data['lon'] != 0, :]

    # First simplest attempt
    tmp_data = meta_data.loc[meta_data['lon'] == 0]
    request_addresses = create_base_addresses(tmp_data)
    request_addresses_1 = request_addresses + ', Slovenia'
    tmp_data = create_requests(tmp_data, request_addresses_1, gmaps)
    final_meta_data = pd.concat([final_meta_data, tmp_data.loc[tmp_data['lon'] != 0]])

    # Second attempt (only on those for which couldn't find the location)
    tmp_data = tmp_data.loc[tmp_data['lon'] == 0]
    request_addresses_2 = request_addresses.loc[tmp_data.index]
    tmp_data = tmp_data.astype({'LocationAddressPostalCode': 'str'})
    request_addresses_2 = request_addresses_2 + ', ' + tmp_data['LocationAddressPostalCode'] + ', Slovenia'
    tmp_data = create_requests(tmp_data, request_addresses_2, gmaps)
    final_meta_data = pd.concat([final_meta_data, tmp_data])

    # Correct remaining few PVs
    # if for some PV we still haven't found its location, then we set its location to the RTP it belongs
    indices = final_meta_data.loc[final_meta_data['lon'] == 0].index
    for idx in indices:
        rtp = rtp_data.loc[rtp_data['ss_name'] == final_meta_data.loc[idx, 'ss_name'], :]
        final_meta_data.loc[idx, 'lon'] = rtp['lon'].values
        final_meta_data.loc[idx, 'lat'] = rtp['lat'].values

    final_meta_data = final_meta_data.reindex(meta_data.index)
    final_meta_data.loc[:, ["lat", "lon"]] = final_meta_data.loc[:, ["lat", "lon"]].astype(np.float64)

    return final_meta_data


def add_weather_station(meta_latlon, ws_meta):
    def get_distance(s_lat, s_lng, e_lat, e_lng):
        R = 6373.0  # approximate radius of earth in km

        s_lat = s_lat * np.pi / 180.0
        s_lng = np.deg2rad(s_lng)
        e_lat = np.deg2rad(e_lat)
        e_lng = np.deg2rad(e_lng)

        d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2

        return 2 * R * np.arcsin(np.sqrt(d))  # in km

    pv_dist_id = []
    pv_dist_km = []

    alt_ids = meta_latlon.index

    for i, alt_id in enumerate(alt_ids):
        if i % 100 == 0: print(i)
        pv_lat, pv_lon = meta_latlon.loc[alt_id, ["lat", "lon"]]

        distances = {}
        for ws_id in ws_meta.index:
            ws_lat, ws_lon = ws_meta.loc[ws_id]
            dist = get_distance(pv_lat, pv_lon, ws_lat, ws_lon)
            distances[ws_id] = dist

        distances = pd.Series(distances)

        # weather station ids for closest stations
        ws_ids = distances.sort_values().index[:1]
        ws_km = distances.min()
        pv_dist_id.append(ws_ids)
        pv_dist_km.append(ws_km)

    pv_dist_id = pd.DataFrame(pv_dist_id, index=alt_ids).rename("ws_{}".format, axis=1)
    pv_dist_km = pd.Series(pv_dist_km, index=alt_ids).rename("ws_km", axis=1).round(2)
    pv_dist = pd.concat([pv_dist_km, pv_dist_id], axis=1)

    meta_latlon_out = meta_latlon.merge(pv_dist, left_index=True, right_index=True)
    return meta_latlon_out


def create_ts_idx(idx, ts_ids_list, mask_dict=None):
    """Creates time indices for multiple PVs from ts_ids_list using idx.

    Parameters
    ----------
    idx: pd.DatetimeIndex
        Time index of future X.
    ts_ids_list: list
        List of PV ids.
    mask_dict: dict, optional
        Dictionary used for sampling during training. Key=pv_id, value=numpy indices

    Returns
    -------
    s.index : pd.DatetimeIndex
        Time indices for multiple PVs.
    s: pd.Series
        Series of PV ids stacked horizontally.
    """

    s = []
    for ts_id in ts_ids_list:
        if mask_dict is None:
            s.append(pd.Series(ts_id, index=idx))
        else:
            s.append(pd.Series(ts_id, index=idx).iloc[mask_dict[ts_id]])
    s = pd.concat(s).rename("ts_id")
    return s.index, s


def mask_night_hours(s_in):
    """Sets zero PV generation for night hours.

    Parameters
    ----------
    s_in : pd.Series
        Timeseries with PV generation.

    Returns
    -------
    s : pd.Series
        Timeseries with PV generation (where night hours are set to 0).
    """
    s = s_in.copy()

    idx = s.index
    mask1 = idx.month.isin([1]) & ((idx.hour < 7) | (idx.hour > 15))
    mask2 = idx.month.isin([2]) & ((idx.hour < 7) | (idx.hour > 16))
    mask34 = idx.month.isin([3, 4]) & ((idx.hour < 5) | (idx.hour > 17))
    mask567 = idx.month.isin([5, 6, 7]) & ((idx.hour < 4) | (idx.hour > 18))
    mask8 = idx.month.isin([8]) & ((idx.hour < 5) | (idx.hour > 18))
    mask9 = idx.month.isin([9]) & ((idx.hour < 5) | (idx.hour > 17))
    mask10 = idx.month.isin([10]) & ((idx.hour < 6) | (idx.hour > 16))
    mask11 = idx.month.isin([11]) & ((idx.hour < 6) | (idx.hour > 15))
    mask12 = idx.month.isin([12]) & ((idx.hour < 7) | (idx.hour > 15))

    mask_nosun = mask1 + mask2 + mask34 + mask567 + mask8 + mask9 + mask10 + mask11 + mask12

    s[mask_nosun] = 0
    return s


def get_X_longterm(t0, horizon, idx_future):
    """Creates feature matrix X for long-term model.

    Parameters
    ----------
    t0 : pd.Timestamp
        Last timestamp with available data.
    horizon : int
        Forecasting horizon measured in 15-min intervals (three horizons: 4*12, 4*72, 4*192).
    idx_future: pd.DatetimeIndex
        Time index of future X.

    Returns
    -------
    X_future : pd.DataFrame
        Feature matrix for all PVs. Shape=(len(idx_future) * len(pv_id_list), number_of_features)
        Index: idx_future (repeated for every PV)
        Columns: ['ghi', 'temp', ..., 'lon', 'emb_id']
    """

    w = get_weather_forecast_all(t0, horizon)
    w = w.loc[w.index.isin(idx_future)]
    meta = get_pv_meta(Type="selected")

    # create hour dummies
    ## dummy column =1
    hour = pd.Series(idx_future.strftime("%H:%M"), index=idx_future, name="hour")
    hour_dummies_ones = pd.get_dummies(hour.sort_values()).reindex(idx_future)

    ## dummy columns - other with zero
    hour_cols = pd.date_range("2020-1-1 04:00", "2020-1-1 18:45", freq="15min").strftime("%H:%M")
    missing_hour_cols = hour_cols.difference(hour_dummies_ones.columns)
    hour_dummies_zeros = pd.DataFrame(np.zeros((len(idx_future), len(missing_hour_cols))),
                                      index=idx_future, columns=missing_hour_cols)

    hour_dummies = pd.concat([hour_dummies_ones, hour_dummies_zeros], axis=1).loc[:, hour_cols]

    # create month dummies
    ## dummy column =1
    month = pd.Series(idx_future.month.astype(str), index=idx_future, name="month").apply(lambda x: "m{}".format(x))
    month_dummies_ones = pd.get_dummies(month.sort_values()).reindex(idx_future)

    ## dummy columns - other with zero
    month_cols = pd.Index(['m1', 'm10', 'm11', 'm12', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'])
    missing_month_cols = month_cols.difference(month_dummies_ones.columns)
    month_dummies_zeros = pd.DataFrame(np.zeros((len(idx_future), len(missing_month_cols))),
                                       index=idx_future, columns=missing_month_cols)

    month_dummies = pd.concat([month_dummies_ones, month_dummies_zeros], axis=1).loc[:, month_cols]

    X_future = []
    for i, pv_id in enumerate(meta.index):
        # get data
        ws_id = meta.loc[pv_id].ws_0
        ws = w.loc[w.ws_id == ws_id]
        ghi_ts = ws.GHI.rename("ghi") / 973.9
        temp_ts = ws.temp.rename("temp") / 20

        # create features
        lat_ts = pd.Series(meta.loc[pv_id].lat, index=idx_future, name="lat") / 46
        lon_ts = pd.Series(meta.loc[pv_id].lon, index=idx_future, name="lon") / 15

        emb_id_ts = pd.Series(meta.loc[pv_id].emb_id, index=idx_future,
                              name="emb_id")  # emb_id (kasneje popravi v emb_id)

        feats_ts = pd.concat([hour_dummies,
                              month_dummies,
                              ghi_ts,
                              temp_ts,
                              lat_ts,
                              lon_ts,
                              emb_id_ts
                              ], axis=1)
        X_future.append(feats_ts)

    X_future = pd.concat(X_future)
    return X_future


def get_X_shortterm(t0, horizon):
    """Creates feature matrix X for short-term model.

    Parameters
    ----------
    t0 : pd.Timestamp
        Last timestamp with available data.

    Returns
    -------
    X_future : pd.DataFrame
        Feature matrix for all PVs. Shape=(number of all PVs, number of features)
        Index: [t0, ..., t0]
        Columns: ['04:00', '04:15', ..., 'temp_t8', 'emb_id']
    """

    idx_future = pd.date_range(t0, freq="15min", periods=4 * 2 + 1, closed="right")

    # get data (weather & PV)
    w = get_weather_forecast_all(t0, horizon)
    w = w.loc[w.index.isin(idx_future)]

    meta = get_pv_meta(Type="selected")
    lags_df = get_rt_pv_data(t0, t0, 'clean').reset_index('timestamp_utc')

    # create hour dummies
    ## dummy column =1
    hour_str = t0.strftime("%H:%M")
    hour_one = pd.DataFrame({hour_str: [1]})
    hour_one.index = [t0]

    ## dummy columns - other with zero
    hour_cols = pd.date_range("2020-1-1 04:00", "2020-1-1 18:45", freq="15min").strftime("%H:%M")
    missing_hour_cols = hour_cols.difference([hour_str])
    hour_zeros = pd.DataFrame({key: [0] for key in missing_hour_cols})
    hour_zeros.index = [t0]
    hour_dummies = pd.concat([hour_one, hour_zeros], axis=1).loc[:, hour_cols]

    # create month dummies
    ## dummy column =1
    month_str = "m{}".format(t0.month)
    month_one = pd.DataFrame({month_str: [1]})
    month_one.index = [t0]

    ## dummy columns - other with zero
    month_cols = pd.Index(['m1', 'm10', 'm11', 'm12', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'])
    missing_month_cols = month_cols.difference([month_str])
    month_zeros = pd.DataFrame({key: [0] for key in missing_month_cols})
    month_zeros.index = [t0]
    month_dummies = pd.concat([month_one, month_zeros], axis=1).loc[:, month_cols]

    # get feats for every PV separately & concat in the end
    # feats for one time series = one row for timestamp t0

    X_future = []
    for i, pv_id in enumerate(meta.index):
        # create feats
        ## weather
        ws_id = meta.loc[pv_id].ws_0
        ws = w.loc[w.ws_id == ws_id]
        ghi_ts = ws.GHI.rename("ghi") / 973.9
        temp_ts = ws.temp.rename("temp") / 20

        ## reshape ghi & temp, shape=(1, 8)
        ghi_ts = pd.DataFrame(ghi_ts.values.reshape(1, -1), columns=["ghi_t{}".format(i) for i in range(1, 9)])
        ghi_ts.index = [t0]

        temp_ts = pd.DataFrame(temp_ts.values.reshape(1, -1), columns=["temp_t{}".format(i) for i in range(1, 9)])
        temp_ts.index = [t0]

        ## lat & lon
        lat_ts = pd.DataFrame({"lat": [meta.loc[pv_id].lat]}) / 46
        lat_ts.index = [t0]
        lon_ts = pd.DataFrame({"lon": [meta.loc[pv_id].lon]}) / 15
        lon_ts.index = [t0]

        ## emb
        emb_id_ts = pd.DataFrame({"emb_id": [meta.loc[pv_id].emb_id]})
        emb_id_ts.index = [t0]

        ## lags
        train_max_ts = meta.pv_max.loc[pv_id]
        lag_ts = pd.DataFrame({"lag": [lags_df.loc[pv_id].Pg_kW]}) / train_max_ts
        lag_ts.index = [t0]

        feats_ts = pd.concat([hour_dummies,
                              month_dummies,
                              ghi_ts,
                              temp_ts,
                              lat_ts,
                              lon_ts,
                              lag_ts,
                              emb_id_ts
                              ], axis=1)
        X_future.append(feats_ts)
    X_future = pd.concat(X_future)
    X_future.index.name = "t0"

    return X_future


def get_forecasts(t0, horizon, sim_name):
    """Generates forecasts for individual PVs.

    Parameters
    ----------
    t0 : pd.Timestamp
        Last timestamp with available data.
    horizon : int
        Forecasting horizon measured in 15-min intervals (three horizons: 4*12, 4*72, 4*192).

    Returns
    -------
    pv_forecast : pd.DataFrame
        PV forecast for individual PVs.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ['tplusH', 'P_forecast_kW', 'timestamp_utc_created', 'timestamp_utc_received']
    """

    meta = get_pv_meta(Type="selected")
    pv_id_list = meta.index.tolist()

    t0_actual = datetime.datetime.utcnow()

    # create idx_future & tplusH_arr
    if horizon == 4 * 12:
        # only from 2 hours ahead (without first 2 hours)
        idx_future = pd.date_range(t0 + pd.Timedelta(2, 'h'),
                                   freq="15min",
                                   periods=horizon + 1 - (2 * 4),
                                   closed="right")
        tplusH_arr = np.linspace(2.25, horizon / 4, horizon - (2 * 4))

    elif horizon == 4 * 72:
        # from next hour ahead
        idx_future = pd.date_range(t0.ceil("1h"),
                                   freq="15min",
                                   periods=69 * 4 - 3)
        tplusH_arr = np.arange(0.25, 68.26, 0.25)

    elif horizon == 4 * 192:
        # from next hour ahead
        idx_future = pd.date_range(t0.ceil("1h"),
                                   freq="15min",
                                   periods=186 * 4 - 3)
        tplusH_arr = np.arange(0.25, 185.26, 0.25)

    ##### SHORT-TERM MODEL #####
    # only forecasts for next 2 hours
    if horizon == 4 * 12:
        # Create Features (one row for each pv_id)
        X_future = get_X_shortterm(t0, horizon)

        # Load model
        model = tf.keras.models.load_model(folder_mod + f"NN_shortterm/NN_shortterm model {sim_name}.h5")

        # Predict
        cols = np.arange(0.25, 2.25, 0.25)
        pred = model.predict(X_future)
        y_pred_st = pd.DataFrame(pred,
                                 index=meta.index,
                                 columns=cols)
        y_pred_st.index.name = "pv_id"
        y_pred_st = y_pred_st.clip(lower=0)  # y_pred_st.shape=(len(pv_id_list), 8)

        # Rescale
        y_pred_st = y_pred_st.apply(lambda s: s * meta.pv_max, axis=0)

        # Reshape (change forecast format: horizon with tplusH, instead of columns)
        y_pred_st = (y_pred_st.stack()
                     .reset_index()
                     .rename({"level_1": "tplusH",
                              0: 'P_forecast_kW'
                              }, axis=1)
                     )
        y_pred_st.loc[:, 'timestamp_utc'] = y_pred_st.loc[:, 'tplusH'].map(lambda x: t0 + pd.Timedelta(x, 'h'))
        y_pred_st = y_pred_st.set_index('timestamp_utc').assign(timestamp_utc_created=t0,
                                                                timestamp_utc_received=t0_actual)

        # y_pred_st.shape = (len(pv_list) * 8, ...)

    ##### LONGTERM MODEL #####
    # Create Features (len(idx_future) for every PV)
    X_future = get_X_longterm(t0, horizon, idx_future)

    # Load model
    model = tf.keras.models.load_model(folder_mod + f"NN_longterm/NN_longterm model {sim_name}.h5")

    # Predict
    idx_all, ids = create_ts_idx(idx_future, pv_id_list)
    pred = model.predict(X_future).flatten()
    y_pred_lt = pd.Series(pred, index=idx_all, name="P_forecast_kW").clip(lower=0)
    y_pred_lt = pd.concat([ids.rename("pv_id"), y_pred_lt], axis=1)
    y_pred_lt.index.name = "timestamp_utc"

    # Rescale
    y_pred_lt.P_forecast_kW = y_pred_lt.P_forecast_kW * y_pred_lt.pv_id.map(meta.pv_max)

    # Add cols
    tplusH_mapping = pd.Series(tplusH_arr, index=idx_future)
    y_pred_lt = y_pred_lt.assign(tplusH=tplusH_mapping,
                                 timestamp_utc_created=t0,
                                 timestamp_utc_received=t0_actual
                                 )

    if horizon == 4 * 12:  # Combine forecasts from shortterm and longterm models
        pv_forecast = pd.concat([y_pred_st, y_pred_lt])
    else:
        pv_forecast = y_pred_lt.copy()

    # mask night hours
    pv_forecast.P_forecast_kW = mask_night_hours(pv_forecast.P_forecast_kW)

    # set index etc.
    pv_forecast = (pv_forecast.reset_index()
                       .sort_values(["pv_id", "timestamp_utc"])
                       .set_index(["pv_id", "timestamp_utc"])
                       .loc[:, ['tplusH', 'P_forecast_kW', 'timestamp_utc_created', 'timestamp_utc_received']]
                       )
    return pv_forecast


def inference(t0, horizon, sim_name, dump=True):
    """Generates forecasts for individual PVs and aggregates (SS & SLO).

    Parameters
    ----------
    t0 : pd.Timestamp
        Last timestamp with available data.
    horizon : int
        Forecasting horizon measured in 15-min intervals (three horizons: 4*12, 4*72, 4*192).
    dump : bool, optinal
        Whether to save results to DB.

    Returns
    -------
    pv_forecast : pd.DataFrame
        PV forecast for individual PVs.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ['tplusH', 'P_forecast_kW', 'timestamp_utc_created', 'timestamp_utc_received']
    ss_agg : pd.DataFrame
        SS aggregate forecast.
        Index: ["ss_name", "tplusH", timestamp_utc"]
        Columns: ['timestamp_utc_created', 'timestamp_utc_received', 'P_forecast_kW',
                  'scaling_factor', 'P_forecast_scaled_MW', 'P_forecast_scaled_MW_p5',
                  'P_forecast_scaled_MW_p25', 'P_forecast_scaled_MW_p75',
                  'P_forecast_scaled_MW_p95']
    slo_agg : pd.DataFrame
        Slo. aggregate forecast.
        Index: [timestamp_utc", "tplusH"]
        Columns: ['timestamp_utc_created', 'timestamp_utc_received', 'P_forecast_kW',
                  'scaling_factor', 'P_forecast_scaled_MW', 'P_forecast_scaled_MW_p5',
                  'P_forecast_scaled_MW_p25', 'P_forecast_scaled_MW_p75',
                  'P_forecast_scaled_MW_p95']
    """

    # get forecasts
    pv_forecast = get_forecasts(t0, horizon, sim_name)
    ss_agg = get_ss_agg_forecasts(pv_forecast)
    slo_agg = get_slo_agg_forecasts(ss_agg)

    # dump
    if dump:
        write_rt_pv_forecast(pv_forecast, horizon)
        write_ss_agg_forecast(ss_agg, horizon)
        write_slo_agg_forecast(slo_agg, horizon)

    return pv_forecast, ss_agg, slo_agg


def get_ss_agg_forecasts(pv_forecast):
    """Generates forecasts for SS aggregates from individual PVs.
       Let k denote scaling factor, Pg_all total installed power of all PVs in a set,
       Pg_rt total installed power of all Real Time PVs. Then, scaling factor k is
       calculated as k = Pg_all / Pg_rt. Final forecasts are calculated by first
       summarizing forecasts of individual PVs for each SS and then multiplied by k.

    Parameters
    ----------
    pv_forecast : pd.DataFrame
        PV forecast for individual PVs.
        Index: ["pv_id", "timestamp_utc"]
        Columns: ['tplusH', 'P_forecast_kW', 'timestamp_utc_created', 'timestamp_utc_received']

    Returns
    -------

    ss_agg : pd.DataFrame
        SS aggregate forecast.
        Index: ["ss_name", "tplusH", "timestamp_utc"]
        Columns: ['timestamp_utc_created', 'timestamp_utc_received', 'P_forecast_kW',
                  'scaling_factor', 'P_forecast_scaled_MW', 'P_forecast_scaled_MW_p5',
                  'P_forecast_scaled_MW_p25', 'P_forecast_scaled_MW_p75',
                  'P_forecast_scaled_MW_p95']
    """
    # calculate total installed power of all PVs (real time & other) for each SS separately
    pv_meta_orig = get_pv_meta(pv_id=None, Type="orig")
    ss_Pinst_kW_orig = pv_meta_orig.groupby("ss_name").Pinst_kW.sum()

    # calculate total installed power of all Real Time PVs for each SS separately
    pv_meta = get_pv_meta(pv_id=None, Type="selected")
    ss_Pinst_kW = pv_meta.groupby("ss_name").Pinst_kW.sum()

    # calculate scaling factor k for each SS
    ss_Pinst_scaling = (ss_Pinst_kW_orig / ss_Pinst_kW).dropna().round(4)

    # calculate sum of individual PVs forecats per SS & add scaling factor to DataFrame
    df = pv_forecast.reset_index()
    df = df.merge(pv_meta, left_on="pv_id", right_index=True)

    cols = ["ss_name", "timestamp_utc", "tplusH", "timestamp_utc_created", "timestamp_utc_received"]
    ss_agg = df.groupby(cols).P_forecast_kW.sum().reset_index()
    ss_agg = pd.concat([ss_agg,
                        ss_agg.ss_name.map(ss_Pinst_scaling).rename("scaling_factor")], axis=1)

    # scale forecasts of each SS separately
    ss_agg = ss_agg.assign(P_forecast_scaled_MW=((ss_agg.scaling_factor * ss_agg.P_forecast_kW) / 1000).round(3))

    # add PI
    ss_agg = add_PI_ss(ss_agg)

    ss_agg = ss_agg.set_index(["ss_name", "timestamp_utc", "tplusH"])
    ss_agg.loc[:, "P_forecast_scaled_MW":] = ss_agg.loc[:, "P_forecast_scaled_MW":].round(2)
    return ss_agg


def get_slo_agg_forecasts(ss_agg):
    """Generates forecasts for Slo. from SS (only SS with RT PVs).
       Let k denote scaling factor, Pg_all total installed power of all PVs in Slo.,
       Pg_ss_rt total installed power of PVs (only the ones connected to SS with RT PVs).
       Then, scaling factor k is calculated as k = Pg_all / Pg_ss_rt total. Final forecasts
       are calculated by first summarizing forecasts of SS (only SS with RT PVs) and then
       multiplied by k.

    Parameters
    ----------
    ss_agg : pd.DataFrame
        SS aggregate forecast.
        Index: ["ss_name", "tplusH", timestamp_utc"]
        Columns: ['timestamp_utc_created', 'timestamp_utc_received', 'P_forecast_kW',
                  'scaling_factor', 'P_forecast_scaled_MW', 'P_forecast_scaled_MW_p5',
                  'P_forecast_scaled_MW_p25', 'P_forecast_scaled_MW_p75',
                  'P_forecast_scaled_MW_p95']

    Returns
    -------
    slo_agg : pd.DataFrame
        Slo. aggregate forecast.
        Index: [timestamp_utc", "tplusH"]
        Columns: ['timestamp_utc_created', 'timestamp_utc_received', 'P_forecast_kW',
                  'scaling_factor', 'P_forecast_scaled_MW', 'P_forecast_scaled_MW_p5',
                  'P_forecast_scaled_MW_p25', 'P_forecast_scaled_MW_p75',
                  'P_forecast_scaled_MW_p95']
    """

    # calculate total installed power of all PVs (real time & other)
    pv_meta_orig = get_pv_meta(pv_id=None, Type="orig")
    slo_Pinst_kW_orig = pv_meta_orig.Pinst_kW.sum()

    # calculate total installed power of all PVs (only the ones on SS with RT PVs)
    pv_meta = get_pv_meta(pv_id=None, Type="selected")
    ss_with_rt = pv_meta.ss_name.unique().tolist()  # list of SS with RT PVs
    ssrt_Pinst_kW = pv_meta_orig.loc[pv_meta_orig.ss_name.isin(ss_with_rt)].Pinst_kW.sum()

    # calculate scaling factor k
    slo_Pinst_scaling = round((slo_Pinst_kW_orig / ssrt_Pinst_kW), 4)

    # calculate sum of SS (only SS with RT PVs) forecasts & scale according to slo_Pinst_scaling
    cols = ["timestamp_utc", "tplusH", "timestamp_utc_created", "timestamp_utc_received"]
    slo_agg = (ss_agg.reset_index()
               .groupby(cols).P_forecast_scaled_MW.sum()
               .reset_index()
               .rename({"P_forecast_scaled_MW": "P_forecast_kW"}, axis=1)
               .assign(scaling_factor=slo_Pinst_scaling)
               )
    slo_agg.P_forecast_kW = slo_agg.P_forecast_kW * 1000

    slo_agg_scaled = (slo_agg.P_forecast_kW * slo_Pinst_scaling / 1000).rename("P_forecast_scaled_MW").round(1)
    slo_agg = pd.concat([slo_agg, slo_agg_scaled], axis=1)

    # add PI
    slo_agg = add_PI_slo(slo_agg)

    slo_agg = slo_agg.set_index(["timestamp_utc", "tplusH"])
    slo_agg.loc[:, "P_forecast_scaled_MW":] = slo_agg.loc[:, "P_forecast_scaled_MW":].round(1)
    return slo_agg


def add_PI_ss(ss_agg):
    arr_down = np.linspace(1, 0.7, len(ss_agg))
    arr_up = np.linspace(1, 1.3, len(ss_agg))
    y_pred_q = ss_agg.copy().assign(p5=ss_agg.P_forecast_scaled_MW * 0.4 * arr_down,
                                    p25=ss_agg.P_forecast_scaled_MW * 0.8 * arr_down,
                                    p75=ss_agg.P_forecast_scaled_MW * 1.1 * arr_up,
                                    p95=ss_agg.P_forecast_scaled_MW * 1.2 * arr_up)

    y_pred_q = y_pred_q.rename({"p5": "P_forecast_scaled_MW_p5",
                                "p25": "P_forecast_scaled_MW_p25",
                                "p75": "P_forecast_scaled_MW_p75",
                                "p95": "P_forecast_scaled_MW_p95"}, axis=1)
    return y_pred_q


def add_PI_slo(slo_agg):
    arr_down = np.linspace(1, 0.7, len(slo_agg))
    arr_up = np.linspace(1, 1.3, len(slo_agg))
    y_pred_q = slo_agg.copy().assign(p5=slo_agg.P_forecast_scaled_MW * 0.4 * arr_down,
                                     p25=slo_agg.P_forecast_scaled_MW * 0.8 * arr_down,
                                     p75=slo_agg.P_forecast_scaled_MW * 1.1 * arr_up,
                                     p95=slo_agg.P_forecast_scaled_MW * 1.2 * arr_up)

    y_pred_q = y_pred_q.rename({"p5": "P_forecast_scaled_MW_p5",
                                "p25": "P_forecast_scaled_MW_p25",
                                "p75": "P_forecast_scaled_MW_p75",
                                "p95": "P_forecast_scaled_MW_p95"}, axis=1)
    return y_pred_q


def evaluate_ss_forecasts(t_start, t_end, horizon, dump):
    """Evaluates SS forecasts between t_start and t_end, whereas keep in mind
       that each forecast is defined and evaluated with timestamp_utc_created.
       Therefore, evaluation will be done between t_start >= timestamp_utc_created
       and t_end < timestamp_utc_created - dt (look at query for loading predictions).
       Here, dt denotes forecasting horizon in hours (e.g. if horizon = 4*12, dt = 12 hours).

    Parameters
    ----------
    t_start : pd.Timestamp
        First timestamp to perform evaluation.
    t_end : pd.Timestamp
        Last timestamp to perform evaluation.
    horizon : int
        Forecasting horizon (4*12, 4*72, 4*192).

    Returns
    -------
    errors : pd.DataFrame
        Dataframe of series.
        Index: ["timestamp_utc"]
        Columns: ['ss_name', 'tplusH', 'timestamp_utc_created', 'P_scaled_MW',
                  'P_forecast_scaled_MW', 'error_MW', 'abs_error_MW', 'nae_perc']
    t_last : pd.Timestamp
        Last timestamp_utc_created for which evaluation was done.
    """

    t_start = pd.Timestamp(t_start)
    t_end = pd.Timestamp(t_end)

    Pinst_ss = get_pv_meta(Type="orig").groupby("ss_name").sum().Pinst_kW / 1000

    # get true & preds
    print("Loading from DB")
    query = """SELECT * FROM [{}].[osse].[ss_agg_clean] \
               where timestamp_utc >= '{}' and timestamp_utc < '{}'""".format(db_clean,
                                                                              t_start,
                                                                              t_end).replace("\n", "")
    ss_true = read_sql(server, db_clean, username, password, query).set_index("timestamp_utc")

    ## calculate dt
    ## define dt (look at t_end - dt when loading ss_pred)
    if horizon == 4 * 12:
        dt = datetime.timedelta(hours=horizon / 4)

    elif horizon == 4 * 72:
        # because 72h horizon is actually 68.25h horizon
        dt = datetime.timedelta(hours=273 / 4)

    elif horizon == 4 * 192:
        # because 192h horizon is actually 185.25h horizon
        dt = datetime.timedelta(hours=741 / 4)

    query = """SELECT * FROM [{}].[osse].[ss_agg_forecast{}h] \
               where timestamp_utc_created >= '{}' and timestamp_utc_created < '{}'""".format(db_clean,
                                                                                              horizon // 4,
                                                                                              t_start,
                                                                                              t_end - dt).replace("\n",
                                                                                                                  "")
    ss_pred = read_sql(server, db_clean, username, password, query).set_index("timestamp_utc")

    # calculate errors
    print("Calculating Errors")
    errors = []

    for ss_name in ss_pred.ss_name.unique()[:]:
        print(ss_name)

        df_pred = ss_pred.loc[ss_pred.ss_name == ss_name]
        df_true = ss_true.loc[ss_true.ss_name == ss_name]
        temp_index = np.intersect1d(df_pred.index, df_true.index)
        true = df_true.loc[temp_index, "Pg_scaled_MW"]
        df_pred = df_pred.loc[temp_index]

        errors_ss = []
        for timestamp_utc_created in df_pred.timestamp_utc_created.unique()[:]:
            pred_i = df_pred.loc[df_pred.timestamp_utc_created == timestamp_utc_created]
            pred = pred_i.P_forecast_scaled_MW

            temp_index = np.intersect1d(pred.index, true.index)
            true_i = true.loc[temp_index]
            pred = pred.loc[temp_index]
            e = true_i - pred
            ae = e.abs()
            nae = 100 * ae / Pinst_ss.loc[ss_name]

            errors_ss_i = pred_i.loc[:, ["ss_name", 'tplusH', 'timestamp_utc_created', "P_forecast_scaled_MW"]].assign(
                P_scaled_MW=true_i,
                error_MW=e,
                abs_error_MW=ae,
                nae_perc=nae,
            )
            errors_ss.append(errors_ss_i)

        errors_ss = pd.concat(errors_ss).sort_values(["tplusH", "timestamp_utc_created"])
        errors.append(errors_ss)

    errors = pd.concat(errors)

    # set col order
    cols_order = ["ss_name", 'tplusH', 'timestamp_utc_created', 'P_scaled_MW',
                  'P_forecast_scaled_MW', 'error_MW', 'abs_error_MW', 'nae_perc']
    errors = errors.loc[:, cols_order]

    # round selected cols
    cols = ['P_scaled_MW', 'P_forecast_scaled_MW', 'error_MW', 'abs_error_MW', 'nae_perc']
    errors.loc[:, cols] = errors.loc[:, cols].round(3)

    # set index (so that format is the same as forecast)
    errors = errors.reset_index().set_index(["ss_name", "tplusH", "timestamp_utc"])

    if dump:
        # write to DB
        print("Writing to DB")
        eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(eng_str, fast_executemany=True)
        conn = engine.connect()
        errors.reset_index().to_sql("ss_agg_evaluation{}h".format(horizon // 4),
                                    schema="osse",
                                    con=engine,
                                    if_exists="append",
                                    index=False,
                                    dtype={"ss_name": sa.VARCHAR(length=30)}
                                    )

    # last timestamp_utc_created for which evaluation was done + timedelta, because above we have >=
    if horizon == 4 * 12:
        t_last = errors.timestamp_utc_created.max() + pd.Timedelta(15, 'm')

    elif horizon == 4 * 72:
        t_last = errors.timestamp_utc_created.max() + pd.Timedelta(6, 'h')

    elif horizon == 4 * 192:
        t_last = errors.timestamp_utc_created.max() + pd.Timedelta(1, 'd')

    return errors, t_last


def evaluate_slo_forecasts(t_start, t_end, horizon, dump):
    """Evaluates Slo. forecasts between t_start and t_end, whereas keep in mind
       that each forecast is defined and evaluated with timestamp_utc_created.
       Therefore, evaluation will be done between t_start >= timestamp_utc_created
       and t_end < timestamp_utc_created - dt (look at query for loading predictions).
       Here, dt denotes forecasting horizon in hours (e.g. if horizon = 4*12, dt = 12 hours).

    Parameters
    ----------
    t_start : pd.Timestamp
        First timestamp to perform evaluation.
    t_end : pd.Timestamp
        Last timestamp to perform evaluation.
    horizon : int
        Forecasting horizon (4*12, 4*72, 4*192).

    Returns
    -------
    errors : pd.DataFrame
        Dataframe of series.
        Index: ["timestamp_utc", "tplusH"]
        Columns: ['timestamp_utc_created', 'P_scaled_MW', 'P_forecast_scaled_MW',
                  'error_MW', 'abs_error_MW', 'nae_perc']
    t_last : pd.Timestamp
        Last timestamp_utc_created for which evaluation was done.
    """

    t_start = pd.Timestamp(t_start)
    t_end = pd.Timestamp(t_end)

    # get true & preds
    print("Loading from DB")
    query = """SELECT * FROM [{}].[osse].[slo_agg_clean] \
               where timestamp_utc >= '{}' and timestamp_utc < '{}'""".format(db_clean,
                                                                              t_start,
                                                                              t_end).replace("\n", "")
    slo_true = read_sql(server, db_clean, username, password, query).set_index("timestamp_utc")

    ## calculate dt
    ## define dt (look at t_end - dt when loading ss_pred)
    if horizon == 4 * 12:
        dt = datetime.timedelta(hours=horizon / 4)

    elif horizon == 4 * 72:
        # because 72h horizon is actually 68.25h horizon
        dt = datetime.timedelta(hours=273 / 4)

    elif horizon == 4 * 192:
        # because 192h horizon is actually 185.25h horizon
        dt = datetime.timedelta(hours=741 / 4)

    query = """SELECT * FROM [{}].[osse].[slo_agg_forecast{}h] \
               where timestamp_utc_created >= '{}' and timestamp_utc_created < '{}'""".format(db_clean,
                                                                                              horizon // 4,
                                                                                              t_start,
                                                                                              t_end - dt).replace("\n",
                                                                                                                  "")
    slo_pred = read_sql(server, db_clean, username, password, query).set_index("timestamp_utc")

    # calculate errors
    print("Calculating Errors")
    Pinst_slo = get_pv_meta(Type="orig").Pinst_kW.sum() / 1000

    slo_errors = []

    temp_index = np.intersect1d(slo_true.index, slo_pred.index)
    slo_true = slo_true.loc[temp_index, "Pg_scaled_MW"]
    slo_pred = slo_pred.loc[temp_index]

    for timestamp_utc_created in slo_pred.timestamp_utc_created.unique()[:]:
        pred_i = slo_pred.loc[slo_pred.timestamp_utc_created == timestamp_utc_created]
        pred = pred_i.P_forecast_scaled_MW

        temp_index = np.intersect1d(pred.index, slo_true.index)
        slo_true_i = slo_true.loc[temp_index]
        pred = pred.loc[temp_index]
        e = slo_true_i - pred
        ae = e.abs()
        nae = 100 * ae / Pinst_slo

        slo_errors_i = (pred_i.loc[:, ['tplusH', 'timestamp_utc_created', "P_forecast_scaled_MW"]]
                        .assign(P_scaled_MW=slo_true_i,
                                error_MW=e,
                                abs_error_MW=ae,
                                nae_perc=nae))
        slo_errors.append(slo_errors_i)
    slo_errors = pd.concat(slo_errors)

    # set col order
    cols_order = ['tplusH', 'timestamp_utc_created', 'P_scaled_MW', 'P_forecast_scaled_MW', 'error_MW', 'abs_error_MW',
                  'nae_perc']
    slo_errors = slo_errors.loc[:, cols_order]

    # round selected cols
    cols = ['P_scaled_MW', 'P_forecast_scaled_MW', 'error_MW', 'abs_error_MW', 'nae_perc']
    slo_errors.loc[:, cols] = slo_errors.loc[:, cols].round(3)

    # set index (so that format is the same as forecast)
    slo_errors = slo_errors.reset_index().set_index(["timestamp_utc", "tplusH"])

    # write to DB
    if dump:
        print("Writing to DB")
        eng_str = fr"mssql+pyodbc://{username}:{password}@{server}/{db_clean}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(eng_str, fast_executemany=True)
        conn = engine.connect()
        slo_errors.reset_index().to_sql("slo_agg_evaluation{}h".format(horizon // 4),
                                        schema="osse",
                                        con=engine,
                                        if_exists="append",
                                        index=False
                                        )

    # last timestamp_utc_created for which evaluation was done + timedelta, because above we have >=
    if horizon == 4 * 12:
        t_last = slo_errors.timestamp_utc_created.max() + pd.Timedelta(15, 'm')

    elif horizon == 4 * 72:
        t_last = slo_errors.timestamp_utc_created.max() + pd.Timedelta(6, 'h')

    elif horizon == 4 * 192:
        t_last = slo_errors.timestamp_utc_created.max() + pd.Timedelta(1, 'd')

    return slo_errors, t_last


def calc_errors(w_pred, w_true):
    """Calculates the difference (error) between the actual measurements of temp,
       GHI and snow, and the forecasts for those weather variables. We calculate
       these errors for all of ARSO's weather stations from which we currently
       receive weather data. We can pass to the function the forecasts for any
       of the three horizons 12h, 72h or 192h.

    Parameters
    ----------
    w_pred : pd.DataFrame
        Dataframe containing the predictions for temp, GHI and snow during a
        given time period.
    w_true : pd.DataFrame
        Dataframe containing the actual measurements of temp, GHI and snow during
        a given time period.

    Returns
    -------
    w_errors : pd.DataFrame
        Dataframe of series.
        Index: ["timestamp_utc", "tplusH"]
        Columns: ['GHI_error', 'temp_error', 'snow_error', 'tplusH', 'ws_id',
                  'GHI', 'temp', 'snow', 'GHI_forecast', 'temp_forecast', 'snow_forecast']
    """

    w_errors = []

    for ws_id in range(1, w_pred.ws_id.max() + 1):
        if ws_id % 50 == 0: print(ws_id)

        ws_pred = w_pred.loc[w_pred.ws_id == ws_id]
        ws_true = w_true.loc[w_true.ws_id == ws_id, ["GHI", "temp", "snow"]]

        ws_errors = []

        for tplusH in np.arange(1, w_pred.tplusH.max() + 1):
            ws_predt = ws_pred.loc[ws_pred.tplusH == tplusH, ["GHI", "temp", "snow"]]
            temp_index = np.intersect1d(ws_true.index, ws_predt.index)
            ws_predt = ws_predt.loc[temp_index]
            ws_truet = ws_true.loc[temp_index]
            ws_errors_t = (ws_truet - ws_predt).assign(tplusH=tplusH, ws_id=ws_id)
            ws_errors_t.columns = ['GHI_error', 'temp_error', 'snow_error', 'tplusH', 'ws_id']

            # next 4 lines are just to have the true and pred cols in our w_errors df
            true = ws_truet - pd.DataFrame(np.zeros(ws_predt.shape), index=ws_predt.index, columns=ws_predt.columns)
            true.columns = ['GHI', 'temp', 'snow']
            pred = pd.DataFrame(np.zeros(ws_truet.shape), index=ws_truet.index, columns=ws_truet.columns) + ws_predt
            pred.columns = ['GHI_forecast', 'temp_forecast', 'snow_forecast']

            ws_errors_t = pd.concat([ws_errors_t, true, pred], axis=1)
            ws_errors.append(ws_errors_t)

        ws_errors = pd.concat(ws_errors)
        w_errors.append(ws_errors)

    w_errors = pd.concat(w_errors)

    return w_errors


def calc_and_write_errors(folder_logs, horizon, t0, dump):
    """Calculates and stores in the weather DB the difference (error) and its
       absolute value (absolute error) between the actual measurements of temp,
       GHI and snow, and the forecasts for those weather variables. We calculate
       these errors for all of ARSO's weather stations from which we currently
       receive weather data. These calculations can be made for any of the three
       horizons 12h, 72h or 192h.

    Parameters
    ----------
    folder_logs : str
        Path to the folder where we have the file for logging the last
        timestamp_utc_created up to which the evaluation for a particular
        horizon was done.
    horizon : int
        Forecasting horizon.
    t0: pd.Timestamp
        Timestamp at which the evaluation takes place (rounded to 30 min).
    dump: Boolean
        Variable for whether to store the calculations to the DB and update
        the log file with the last timestamp_utc_created for which evaluation
        for a particular horizon was done.

    Returns
    -------
    w_errors : pd.DataFrame
        Dataframe of series.
        Index: ["timestamp_utc", "tplusH"]
        Columns: ['timestamp_utc_created', 'GHI_error', 'temp_error', 'snow_error',
                  'GHI_abs_error', 'temp_abs_error', 'snow_abs_error', 'tplusH', 'ws_id',
                  'GHI', 'temp', 'snow', 'GHI_forecast', 'temp_forecast', 'snow_forecast']
    t_last: pd.Timestamp
        The last timestamp_utc_created for which evaluation for a particular
        horizon was done.
    """

    t_latest = parse(
        pd.read_csv(folder_logs + f"evaluation_weather/evaluation_weather_{horizon}h.csv", header=None).iloc[0, 0])
    w_true, w_pred = get_weather_data(t_latest, t0, server, db_weather, username, password, horizon)
    w_errors = calc_errors(w_pred, w_true)
    w_errors['timestamp_utc_created'] = w_errors.index - pd.to_timedelta(w_errors.tplusH.values, unit='h')
    w_errors = w_errors.assign(GHI_abs_error=lambda x: w_errors.GHI_error.abs(),
                               temp_abs_error=lambda x: w_errors.temp_error.abs(),
                               snow_abs_error=lambda x: w_errors.snow_error.abs())

    # last timestamp_utc_created for which evaluation was done + timedelta, because above we have >=
    if horizon == 12:
        t_last = w_errors.timestamp_utc_created.max() + pd.Timedelta(1, 'h')

    elif horizon == 72:
        t_last = w_errors.timestamp_utc_created.max() + pd.Timedelta(6, 'h')

    elif horizon == 192:
        t_last = w_errors.timestamp_utc_created.max() + pd.Timedelta(1, 'd')

    if dump:
        write_wforecast_evaluation(w_errors, 4 * horizon, username, password, server, db_clean)
        pd.Series(t_last).to_csv(folder_logs + f"evaluation_weather/evaluation_weather_{horizon}h.csv", index=False,
                                 header=False)

    return w_errors, t_last