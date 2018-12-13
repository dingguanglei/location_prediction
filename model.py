import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataset import create_dataset
from torch.utils.data import DataLoader
import math
import os
import pandas as pd


# create train dataset and test dataset
def DataGen(dataset_num, seq, k):
    train_prop = 0.8
    train_dat = []
    test_dat = []
    L = len(seq) // dataset_num
    n = 0
    while n < dataset_num * train_prop:
        for i in range(L - k):
            indat = seq[(i + n * L):(i + k + n * L)]
            outdat = seq[(i + k + n * L)]
            train_dat.append((indat, [outdat]))
        n += 1
    while dataset_num * train_prop <= n < dataset_num:
        for i in range(L - k):
            indat = seq[(i + n * L):(i + k + n * L)]
            outdat = seq[(i + k + n * L)]
            test_dat.append((indat, [outdat]))
        n += 1
    return train_dat, test_dat

# you don't need this function.
# def ToVariable(x):
#     tmp = torch.FloatTensor(x)
#     return Variable(tmp)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers=4)
        self.hidden2out = nn.Linear(hidden_dim, 900)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(4, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(4, 1, self.hidden_dim)).cuda())

    def forward(self, seq):
        lstm_out, self.hidden = self.rnn(seq.view(len(seq), 1, -1), self.hidden)
        outdat_in_last_timestep = lstm_out[-1, :, :]
        outdat = self.hidden2out(outdat_in_last_timestep)
        return outdat


use_gpu = torch.cuda.is_available()
# print(use_gpu)
model = LSTM(900, 2000)
if use_gpu:
    model = model.cuda()

loss_function = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)


dataset = create_dataset("dataset")
train_data, test_data = DataGen(100, dataset, 5)


# print(len(train_data))
# print(len(test_data))
def train(epoch):
    for step, input_data in enumerate(train_data, 1):
        # seq = ToVariable(input_data[0])
        # outs = ToVariable(input_data[1])
        seq = input_data[0]
        outs = input_data[1]
        if use_gpu:
            seq = seq.cuda()
            outs = outs.cuda()
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        predict = model(seq)
        loss = loss_function(predict, outs)
        loss.backward()
        optimizer.step()
        print("step{}".format(step), ": ", loss.item())

        if step % 17 == 0:
            # print(predict)
            # 只算一个预测结果的loss？
            for i in range(len(predict[0])):
                if predict[0][i] < 0:
                    predict[0][i] = 0
                if predict[0][i] % 1 > 0.3:
                    predict[0][i] = math.ceil(predict[0][i])
                else:
                    predict[0][i] = math.floor(predict[0][i])
            # print(predict)
            loss_int = loss_function(predict, outs)
            print(loss_int.item())


def checkpoint(epoch):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(os.path.join("checkpoints", "try")):
        os.mkdir(os.path.join("checkpoints", "try"))
    model_out_path = "checkpoints/try/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format("checkpoints" + "try"))


nEpochs = 1
for epoch in range(1, nEpochs + 1):
    train(epoch)
    # if epoch % 5 == 0:
    #     checkpoint(epoch)

predDat = []
model = model.eval()
for step, data in enumerate(test_data, 1):
    # seq = ToVariable(data[0])
    # trueVal = ToVariable(data[1])
    seq = data[0]
    trueVal = data[1]
    if use_gpu:
        seq = seq.cuda()
        trueVal = trueVal.cuda()
    predDat = model(seq)
    if step < 18:
        for i in range(len(predDat[0])):
            if predDat[0][i] < 0:
                predDat[0][i] = 0
            if predDat[0][i] % 1 > 0.3:
                predDat[0][i] = math.ceil(predDat[0][i])
            else:
                predDat[0][i] = math.floor(predDat[0][i])
        loss_int = loss_function(predDat, trueVal)
        predDat = predDat[-1].data.cpu().numpy()
        print(predDat)
        print(loss_int.data[0])
        dataframe = pd.DataFrame(predDat)
        dataframe.to_csv("result/prediction_900_time{}.csv".format(step + 5), index=False, sep=',')
# predDat = predDat[-1].data.numpy()
# pred = np.split(predDat, 30, axis=0)
# print(pred)
