import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from data_rasterize import rasterize
import os
from .global_configure import *


def get_data(path):
    data = pd.read_csv(path)
    col = data.iloc[:, 0]
    arrs = col.values
    return arrs


def drow_map(arrs, num, time):
    # GRID_SIZE = 30
    # MAP_RANGE_HALF = MAP_RANGE // 2
    arrs = np.split(arrs, GRID_SIZE, axis=0)
    plt.imshow(arrs, extent=(-MAP_RANGE_HALF, MAP_RANGE_HALF, -MAP_RANGE_HALF, MAP_RANGE_HALF), cmap=cm.rainbow,
               norm=LogNorm(),
               origin='lower')
    plt.colorbar()
    # plt.show()
    if not os.path.exists("predDataMap"):
        os.mkdir("predDataMap")
    if not os.path.exists(os.path.join("predDataMap", "excel{}".format(num))):
        os.mkdir(os.path.join("predDataMap", "excel{}".format(num)))
    plt.savefig("predDataMap/excel{}/time{}.jpg".format(num, time))
    plt.close()
    print("Prediction figure saved to predDataMap/excel{}/time{}.jpg".format(num, time))


num_excel = 81
for i in range(6, 23):
    pred_path = "result/prediction_900_time{}.csv".format(i)
    pred_arr = get_data(pred_path)
    drow_map(pred_arr, num_excel, i)

true_path = "dataset/{}.xlsx".format(num_excel)
rasterize(num_excel, true_path, True)
