import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from data_rasterize import rasterize
import os

def get_data(path):
    data = pd.read_csv(path)
    col = data.iloc[:, 0]
    arrs = col.values
    return arrs
def drow_map(arrs, num, time):
    arrs= np.split(arrs, 30, axis=0)
    plt.imshow(arrs, extent=(-750, 750, -750, 750), cmap=cm.rainbow, norm=LogNorm(), origin='lower')
    plt.colorbar()
    # plt.show()
    if not os.path.exists("predDataMap"):
        os.mkdir("predDataMap")
    if not os.path.exists(os.path.join("predDataMap", "excel{}".format(num))):
        os.mkdir(os.path.join("predDataMap", "excel{}".format(num)))
    plt.savefig("predDataMap/excel{}/time{}.jpg".format(num, time))
    plt.close()
    print("Prediction figure saved to predDataMap/excel{}/time{}.jpg".format(num, time))

for i in range(6, 23):
    pred_path = "result/prediction_900_time{}.csv".format(i)
    pred_arr = get_data(pred_path)
    drow_map(pred_arr, 81, i)

true_path = "dataset/81.xlsx"
rasterize(81, true_path, True)