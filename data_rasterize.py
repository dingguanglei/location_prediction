from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xlrd
from matplotlib.colors import LogNorm
import numpy as np
import os
from .global_configure import *


def rasterize(num_excel, path, save_fig):
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('Sheet1')
    total_cols = table.ncols
    total_rows = table.nrows
    seq_arr = []
    for n in range(total_rows):
        rows = table.row_values(n)

        X_parameter = []
        Y_parameter = []
        for i in range(total_cols // 6):
            X_parameter.append(float(rows[3 + i * 6]))
            Y_parameter.append(float(rows[4 + i * 6]))
        arr = np.zeros((1000, 2))
        arr[:, 0] = X_parameter
        arr[:, 1] = Y_parameter

        arr = arr // GRID_SIZE  # arr矩阵表示x,y坐标
        result = []
        arr = arr.tolist()
        for i in arr:
            if -15 <= i[0] < 15 and -15 <= i[1] < 15:
                x = arr.count(i)
                result.append(i + [x])
        res = np.array(sorted(list(set([tuple(t) for t in result]))))
        res = res.astype(int)

        arr = np.zeros((GRID_NUM, GRID_NUM))
        x, y, z = res[:, 0], res[:, 1], res[:, 2]
        for i in range(len(x)):
            arr[y[i] + 15, x[i] + 15] = z[i]
        seq = np.concatenate(arr, axis=0)
        seq_arr.append(seq)
        if save_fig:
            plt.imshow(arr, extent=(-MAP_RANGE_HALF, MAP_RANGE_HALF, -MAP_RANGE_HALF, MAP_RANGE_HALF), cmap=cm.rainbow,
                       norm=LogNorm(), origin='lower')
            plt.colorbar()
            # plt.show()
            if not os.path.exists("trueDataMap"):
                os.mkdir("trueDataMap")
            if not os.path.exists(os.path.join("trueDataMap", "excel{}".format(num_excel))):
                os.mkdir(os.path.join("trueDataMap", "excel{}".format(num_excel)))
            plt.savefig("trueDataMap/excel{}/time{}.jpg".format(num_excel, n + 1))
            plt.close()
            print("Raw figure saved to trueDataMap/excel{}/time{}.jpg".format(num_excel, n + 1))

    return seq_arr
