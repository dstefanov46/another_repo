import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import psutil
from pathlib import Path
import os

x_axis = pd.date_range("2017-1-1", periods=96, freq="15min").strftime("%H:%M")

def get_x_axis(freq, periods): return pd.date_range("2017-1-1", periods=periods, freq=freq).strftime("%H:%M")

def display_time(t_1, t_2):
    time = t_2-t_1
    if time >= 60 and time < 3600:
        print("Calculation time:", round(time/60, 1), "min")
    elif time >= 3600:
        print("Calculation time:", round(time/3600, 1),  "h")
    else:
        print("Calculation time:", round(time, 1), "s")
        


def memory_usage(): 
    print("{:.3f} GB".format(psutil.Process(os.getpid()).memory_info()[-2] / 1_000_000_000))
    
    
def get_folder_size(path):
    size_GB = sum(p.stat().st_size for p in Path(path).rglob('*')) / 1e9
    print("{} size: {:.3f} GB".format(path, size_GB))

  
def reshape_ts(s):
	df = s.to_frame().assign(date=s.index.to_period("D"),
						     hour=s.index.time)
	return df.pivot_table(s.name, index="date", columns="hour")


def del_files_in_folder(folder, file_list=None):
    file_list = os.listdir(folder) if file_list == None else file_list
    for file in file_list:
        os.remove(folder + file)