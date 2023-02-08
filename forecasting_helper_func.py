import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import he_normal, glorot_normal

cwd = os.path.dirname(os.getcwd())
folder_gen = cwd +  "/generated_data/"


def get_interactions(s, dummies, poly_degree=1):
    if s.name == None: s.name = "_"
    s1 = dummies.apply(lambda x: x * s).rename("{}".format(s.name + "_{}").format, axis=1)
    s2 = (dummies.apply(lambda x: x * s) **2).rename("{}".format(s.name + "2_{}").format, axis=1)
    s3 = (dummies.apply(lambda x: x * s) **3).rename("{}".format(s.name + "3_{}").format, axis=1)
    
    if poly_degree == 1: s2, s3 = None, None
    if poly_degree == 2: s3 = None

    df = pd.concat([s1, s2, s3], axis=1)
    return df


def get_lags(ts, lag_start, lag_end):
    """ Get lags for interval (e.g. hourly) model, where every row is timestep.
        lag_start is a non-positive number. 
        
        Example:
        1. Use actual temperature without lags: lag_start=0, lag_end=0
        2. Use load lags from 2 days ago: lag_start=-2*periods, lag_end=-2*periods
        3. Use last 24 lags of temp.: lag_start=0, lag_end=-periods
        
        Beware: it returns the same time indices as input, thus some NaNs.
    """
    df_all = []

    for shift in range(lag_end, lag_start + 1):
        df_all.append(ts.shift(-shift).rename("lag_{}".format(shift)))
    df_all = pd.concat(df_all, axis=1)
    
    return df_all


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
    
    holidays ={"{}/01/01".format(year): "Jan 1", "{}/01/02".format(year): "Jan 2",
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


def get_season(ts_in):

    """
    Get information about season for certain year
    
    Parameters
    ----------
    ts_in : input time series for which you want to get seasons
    
    Returns
    ----------
    season : Series with three values:
                w = winter
                i = inter-season
                s = summer
    """
    idx = ts_in.index if isinstance(ts_in, pd.Series) else ts_in
    
    season = pd.Series()

    for year in idx.year.unique():

        # CREATE SEASONS
        yearly_season = pd.Series("i", index=idx[idx.year == year])

        yearly_season.loc[: "{}-2-15".format(year)] = "w"
        yearly_season.loc["{}-11-15".format(year) :] = "w"
        yearly_season.loc["{}-5-15".format(year) : "{}-9-15".format(year)] = "s"

        season = pd.concat([season, yearly_season])
    return season


def get_forecaster(input_size, emb_size, model_type):  # brez predprocesiranja embedinga
    weight_decay = 1e-4
    units = 64
    inputs = Input((input_size,), name="inputs")

    #x = inputs
    
    emb = Embedding(emb_size, 10, input_length=1)(inputs[:, -1])
    emb = Flatten()(emb)

    x = Concatenate(name="concat")([inputs[:, :-1], emb])  # POSKUSI TUDI NAJPREJ PREDPROCESIRAT z Dense
    
    x = Dense(units, "leaky_relu", 
              name="Dense1",
              kernel_regularizer=l1(weight_decay),
             )(x)
    x = Dense(units, "leaky_relu", 
              name="Dense2",
              kernel_regularizer=l1(weight_decay),
             )(x)
    x = Dense(units, "leaky_relu", 
              name="Dense3",
              kernel_regularizer=l1(weight_decay),
             )(x)
    x = Dense(units, "leaky_relu", 
              name="Dense4",
              kernel_regularizer=l1(weight_decay),
             )(x)
    
    # Set number of output neurons 
    if model_type == 'shortterm':
        num_output_units = 2 * 4
    elif model_type == 'longterm':
        num_output_units = 1
    
    x = Dense(num_output_units, "relu",
              name="Dense5"
             )(x)
    outputs = x
    model = Model(inputs, outputs)
    return model


class TSGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras. Train & trainval sampling is stochastic, val / test is not!'
    def __init__(self, batch_size=1024, n_steps_per_epoch=50, 
                 alt_ids_list=[], set_type="train", 
                 predict=False, folder_data=None):
        self.batch_size = batch_size
        self.alt_ids_list = alt_ids_list  # number of unique ts in a set
        self.set_type = set_type
        self.predict = predict
        self.folder_data = folder_data
        
        # only one pass over dataset during val and test or predict
        if (self.set_type in ["val", "test"]) or self.predict:
            self.n_steps_per_epoch = len(alt_ids_list)
        else: self.n_steps_per_epoch = n_steps_per_epoch
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_steps_per_epoch
            
    def get_ts_batch(self, alt_id, n_series, return_all_rows=False):
        """
        Load n_series of rows from train dataset for desired ts_id.
        
        """
        
        X = np.load(self.folder_data + "X_{}_alt_id={}.npy".format(self.set_type, alt_id))
        y_ts = np.load(self.folder_data + "y_{}_alt_id={}.npy".format(self.set_type, alt_id))

        chosen_rows = np.random.choice(np.arange(len(y_ts)), n_series)
        
        # whether to return randomly sampled rows arr all data for each series
        if (self.set_type in ["val", "test"]) or self.predict or return_all_rows:
            return X, y_ts
        else:
            return X[chosen_rows], y_ts[chosen_rows]
              
    
    def __data_generation(self, index):
        """
        Loads desired data from a disk and returns a mini-batch.
        If val, test or predict -> one batch holds data for one ts, otherwise
        one batch holds sampled data from multiple ts.
        """

        if (self.set_type in ["val", "test"]) or self.predict:
            X_batch, y_batch = self.get_ts_batch(alt_id=self.alt_ids_list[index], n_series=1)
            
        else:
            sampling_info = (pd.Series(np.random.choice(self.alt_ids_list,
                                       size=self.batch_size))
                               .value_counts().sort_index())
            X_batch, y_batch = [], []

            for alt_id in sampling_info.index:
                n_series = sampling_info.loc[alt_id]  #  number of timestamps to sample

                X_ts, y_ts = self.get_ts_batch(alt_id, n_series)
                X_batch.append(X_ts)                
                y_batch.append(y_ts)

            X_batch = np.concatenate(X_batch)
            y_batch = np.concatenate(y_batch)

        return X_batch, y_batch

    
    def __getitem__(self, index):
        """
        index: consecutive batch number (0, 1, 2, ...) in one epoch.
        This method is executed at the beginning of every batch &
        returns a mini-batch of data.
        """
        # Generate data
        X_lags, y = self.__data_generation(index)
        return X_lags, y