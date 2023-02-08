import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from IPython.display import Image 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# PLOTTING
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as dates
plt.rcParams['axes.axisbelow'] = True  # grid behind plot
mpl.rcParams['figure.facecolor'] = "w"

import seaborn as sns
#sns.set(color_codes=False)

import plotly
import plotly.express as px
plotly.offline.init_notebook_mode()


# PYTHON
from itertools import product
import random
import sys
import time
import os
import datetime as dt
from dateutil.parser import parse
import itertools
from itertools import islice
import collections


# SKLEARN
import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.model_selection import * 
#train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, learning_curve, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFromModel, f_regression
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, r2_score, make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans, DBSCAN

# SCIENTIFIC COMPUTING
import scipy
import scipy.stats as st
from scipy.stats import probplot
from scipy import stats
from scipy.stats import boxcox

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import joblib
  
# OTHER
import pickle
import csv
import copy
#import graphviz
#import pydotplus

import shutil
import h5py