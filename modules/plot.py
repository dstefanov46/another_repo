import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams['figure.facecolor'] = "w"
plt.rcParams['axes.axisbelow'] = True  # grid behind plot

import plotly
import plotly.express as px
plotly.offline.init_notebook_mode()

import seaborn as sns
#sns.set(color_codes=False)

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot

import time

import os
cwd = os.path.dirname(os.getcwd())


# GET FUNCTIONS FROM OTHER MODULES
from runpy import run_path
helper_func_default = run_path(cwd + "/modules/default/helper_func_default.py")
plot_default = run_path(cwd + "/modules/default/plot_default.py")

reshape_ts = helper_func_default["reshape_ts"]

blue = plot_default["blue"]
green = plot_default["green"]
red = plot_default["red"]
grey = plot_default["grey"]
