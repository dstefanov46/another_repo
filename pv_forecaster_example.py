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


def get_holidays(year):
    """
    Get holidays dates for specific year

    Parameters
    ----------
    year : year (int) for which you want to get holidays

    Returns
    ----------
    holidays : dict where keys are holiday dates (str) for one year and values are holidays names
    """

    holidays = {"{}/01/01".format(year): "Jan 1", "{}/01/02".format(year): "Jan 2",
                "{}/02/08".format(year): "Feb 8", "{}/04/27".format(year): "Apr 27",
                "{}/05/01".format(year): "May 1", "{}/05/02".format(year): "May 2",
                "{}/06/25".format(year): "Jun 25", "{}/08/15".format(year): "Aug 15",
                "{}/10/31".format(year): "Oct 31", "{}/11/01".format(year): "Nov 1",
                "{}/12/25".format(year): "Dec 25", "{}/12/26".format(year): "Dec 26"}

    if year == 2012:
        holidays["2012/04/07"] = "Easter Sat"
        holidays["2012/04/08"] = "Easter Sun"
        holidays["2012/04/09"] = "Easter Mon"
        holidays["2012/05/31"] = "Binkosti"

    if year == 2013:
        holidays["2013/03/30"] = "Easter Sat"
        holidays["2013/03/31"] = "Easter Sun"
        holidays["2013/04/01"] = "Easter Mon"
        del holidays["2013/01/02"]

    if year == 2014:
        holidays["2014/04/19"] = "Easter Sat"
        holidays["2014/04/20"] = "Easter Sun"
        holidays["2014/04/21"] = "Easter Mon"
        del holidays["2014/01/02"]

    if year == 2015:
        holidays["2015/04/04"] = "Easter Sat"
        holidays["2015/04/05"] = "Easter Sun"
        holidays["2015/04/06"] = "Easter Mon"
        del holidays["2015/01/02"]

    if year == 2016:
        holidays["2016/03/26"] = "Easter Sat"
        holidays["2016/03/27"] = "Easter Sun"
        holidays["2016/03/28"] = "Easter Mon"
        del holidays["2016/01/02"]

    if year == 2017:
        holidays["2017/04/15"] = "Easter Sat"
        holidays["2017/04/16"] = "Easter Sun"
        holidays["2017/04/17"] = "Easter Mon"

    if year == 2018:
        holidays["2018/03/31"] = "Easter Sat"
        holidays["2018/04/01"] = "Easter Sun"
        holidays["2018/04/02"] = "Easter Mon"

    if year == 2019:
        holidays["2019/04/20"] = "Easter Sat"
        holidays["2019/04/21"] = "Easter Sun"
        holidays["2019/04/22"] = "Easter Mon"

    return holidays
