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
import plotly.graph_objects as go

import seaborn as sns
#sns.set(color_codes=False)

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot

import time


# SEABORN COLORS        
blue = sns.color_palette()[0]
green = sns.color_palette()[2]
red = sns.color_palette()[3]
grey = sns.color_palette()[7]
    
   
def ecdf(s): return s.sort_values().values, np.linspace(0, 1, len(s))

def plot_ecdf(s, title="", xlabel="", ylabel="", fontsize=15, size=None, color=None, alpha=0.8, label=None): 
    x, y = s.sort_values().values, np.linspace(0, 1, len(s))
    plt.scatter(x, y, alpha=alpha, s=size, c=color, label=label)
    plt.xticks(fontsize=fontsize, rotation=45), plt.yticks(np.arange(0, 1.1, 0.1), fontsize=fontsize);
    plt.xlabel(xlabel, fontsize=fontsize), plt.ylabel(ylabel, fontsize=fontsize);
    plt.title(title, fontsize=fontsize);
    plt.grid()
    
    
def get_color_pal(color_name="Blue", n_colors = 2, reverse=True):
    # nujno: http://jose-coto.com/styling-with-seaborn  (tukaj imaš vse)
    # pallete_names: https://chrisalbon.com/python/seaborn_color_palettes.html
    # plot: https://matplotlib.org/api/pyplot_api.html?highlight=plot#matplotlib.pyplot.plot
    # add _r to reverse
    
    # for heatmaps use 2 lines below
    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list('Custom', get_color_pal("Reds", 6, False), 6)  # Reds
    #sns.heatmap(df_plot, cmap=cmap, linewidths=0.01, linecolor="grey");
    
    out_r = sns.color_palette(color_name + "_r", n_colors, desat=.8).as_hex()  # reverse colormap
    out = sns.color_palette(color_name, n_colors, desat=.8).as_hex()
    return out_r if reverse else out


def plotly_df(df_in, slider=False):
    """
    plots all cols of time series df
    """
    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else: 
        df = df_in.copy()

    import plotly.graph_objects as go

    title = "TS Data"

    def tuple2rgb(color_palette): return ["rgb" + str(color) for color in color_palette]

    color_list = tuple2rgb(sns.color_palette(n_colors=df.shape[1]))

    plot_data = []

    for i, col in enumerate(df.columns):

        go_scatter = go.Scatter(x=df.index,
                                y=df.loc[:, col],
                                line=go.scatter.Line(color=color_list[i], width=2),
                                opacity=0.8,
                                name=col,
                                text="")  # dodaten napis ko daš gor hoover, če rabiš dodaj
        plot_data.append(go_scatter)

    layout = go.Layout(height=800, width=900, 
                       font=dict(size=18),
                       title=title,
                       xaxis=dict(title='Timestamps',  # xlabel
                                            # Range selector with buttons
                                             rangeselector=dict(
                                                 # Buttons for selecting time scale
                                                 buttons=list([
                                                     # 1 day
                                                     dict(count=1,
                                                          label='1d',
                                                          step='day',
                                                          stepmode='todate'),
                                                     # 1 week
                                                     dict(count=7,
                                                          label='1w',
                                                          step='day',
                                                          stepmode='todate'),
                                                     # 1 month
                                                     dict(count=1,
                                                          label='1m',
                                                          step='month',
                                                          stepmode='backward'), 
                                                     # Entire scale
                                                     dict(step='all')
                                                 ])
                                             ),
                                             # Sliding for selecting time window
                                             rangeslider=dict(visible=True),
                                             # Type of xaxis
                                             type='date'),
                       # yaxis is unchanged
                       yaxis=dict(title='')  # ylabel
                       )
    if not slider: layout = None
    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()