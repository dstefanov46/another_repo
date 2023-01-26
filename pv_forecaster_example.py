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
