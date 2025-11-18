# Author: Junran Chen
# Date: 2023-Aug-30
# Function: Create data from .CSV file, and add history operation data.

import numpy as np
import pandas as pd
import torch



def create_dataset(X_data, Y_data, lookback):
    """Transform a time series into a prediction dataset

    Args:
        X_data: Feature
        Y_data: Target
        lookback: Size of window for prediction
    """
    X, y = [], []
    # print(len(Y_data))
    for i in range(lookback, len(Y_data), 1):
        feature = X_data[i - lookback+1:i+1, :]
        target = Y_data[i]  # it's tricky that 1:10, returns 1~9 actually, so here we should -1 on i.
        X.append(feature)
        y.append(target)
    # X = np.array(X)
    # y = np.array(y)
    # return torch.tensor(X), torch.tensor(y)
    return X, y


# ---------------------------Create Training set-------------------------------------------------------
Train = pd.read_csv('./Combined_Training31-Aug-2023.csv')
X_Train = Train[["SOC", "I", "T"]].values.astype('float32')
Y_Train = Train[["V"]].values.astype('float32')
del Train
tempX, tempy = create_dataset(X_Train, Y_Train, 400)
del X_Train, Y_Train
cacheX = np.array(tempX)
cachey = np.array(tempy)
cacheX = torch.tensor(cacheX)
cachey = torch.tensor(cachey)
torch.save([cacheX, cachey], './Combined_Training3-Oct-2024.pt')
# --------------------------Create Testing set-----------------------------------------------------
# Train = pd.read_csv('./Combined_Testing31-Aug-2023.csv')
# X_Train = Train[["SOC", "I", "T"]].values.astype('float32')
# Y_Train = Train[["V"]].values.astype('float32')
# del Train
# tempX, tempy = create_dataset(X_Train, Y_Train, 1)
# del X_Train, Y_Train
# cacheX = np.array(tempX)
# cachey = np.array(tempy)
# cacheX = torch.tensor(cacheX)
# cachey = torch.tensor(cachey)
# torch.save([cacheX, cachey], './Combined_Testing31-Aug-2023.pt')
# --------------------------Create Training set and split-----------------------------------------------------
# Train = pd.read_csv('./Combined_TrainingTrainSet24-Jan-2024.csv')
# X_Train = Train[["SOC", "I", "T"]].values.astype('float32')
# Y_Train = Train[["V"]].values.astype('float32')
# del Train
# tempX, tempy = create_dataset(X_Train, Y_Train, 20)
# del X_Train, Y_Train
# cacheX = np.array(tempX)
# cachey = np.array(tempy)
# cacheX = torch.tensor(cacheX)
# cachey = torch.tensor(cachey)
# torch.save([cacheX, cachey], './Combined_TrainingTrainSet24-Jan-2024.pt')
#
# Train = pd.read_csv('./Combined_TrainingValidSet24-Jan-2024.csv')
# X_Train = Train[["SOC", "I", "T"]].values.astype('float32')
# Y_Train = Train[["V"]].values.astype('float32')
# del Train
# tempX, tempy = create_dataset(X_Train, Y_Train, 900)
# del X_Train, Y_Train
# cacheX = np.array(tempX)
# cachey = np.array(tempy)
# cacheX = torch.tensor(cacheX)
# cachey = torch.tensor(cachey)
# torch.save([cacheX, cachey], './Combined_TrainingValidSet26-Jan-2024.pt')



# Author: Junran Chen
# Date: 2023-June-23
# Function: module for LSTM model
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import datetime, timedelta


class VEstimLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True) #we have out data from the datacreate method arranged in (batches,  sequence, features) 
        self.linear = nn.Linear(hidden_size, 1)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)
        # self.h_s = None
        # self.h_c = None

    def forward(self, x, h_s, h_c):
        # The h_s, h_c is defaulted to 0 every time, so only remember last 500-second data
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        y = self.linear(y)
        # y = torch.clamp(y, 0, 1)    # Clipped ReLU layer
        # y = self.LeakyReLU(y)
        return y, (h_s, h_c)
    
    # Author: Junran Chen
# Date: 2023-June-23
# Function: Train LSTM model.
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import datetime, timedelta
from VEstimLSTM_Mina import VEstimLSTM

INPUT_SIZE = 3
HIDDEN_SIZE = 10
BATCH_SIZE = 100
EPOCH = 5000
# DROPOUT = 0.2
LAYERS = 1  # LSTM hidden layers
LR = 0.00001  # Learning rate
ValidFrequency = 3  # Validation frequency
torch.manual_seed(20)  # set random seed.


# log function
def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


# ----------------------------Training the network---------------------------------------------------------
print(str(datetime.now() + timedelta(hours=0)) + ': ')
torch.manual_seed(0)  # set random seed.
generator1 = torch.Generator().manual_seed(42)
[cacheX, cachey] = torch.load('./Combined_Training29-Jan-2024.pt')  # load data set
log_string = 'INPUT_SIZE = 3, HIDDEN_SIZE = 10, BATCH_SIZE = 100, EPOCH = 5000, LR = 0.01, LAYERS = 1, ' \
             'ValidFrequency = 3'
log('./LSTM_March8.log', log_string)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # select CPU or GPU
train_set_size = int(len(cachey) * 0.8)
valid_set_size = len(cachey) - train_set_size
Valid, Train = data.random_split(cacheX, [0.3, 0.7], generator=torch.Generator().manual_seed(0))
# This is when data shuffle happened.
train_sampler = data.SubsetRandomSampler(Train.indices)
valid_sampler = data.SubsetRandomSampler(Valid.indices)
trainLoader = data.DataLoader(data.TensorDataset(cacheX, cachey),
                              batch_size=BATCH_SIZE, drop_last=True, sampler=train_sampler)
validLoader = data.DataLoader(data.TensorDataset(cacheX, cachey),
                              batch_size=BATCH_SIZE, drop_last=True, sampler=valid_sampler)
del cacheX, cachey
# model = torch.load('./LSTM_March8.model', map_location='cuda:0').get('model').cuda()
model = VEstimLSTM(input_size=INPUT_SIZE,
                   hidden_size=HIDDEN_SIZE,
                   layers=LAYERS)
model.to(device)
loss_fn = nn.MSELoss()
# optimizer = torch.load('./LSTM_Oct17.optim')
optimizer = optim.Adam(model.parameters(), lr=LR)
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    # milestones=[EPOCH // 10 * 1, EPOCH // 10 * 2, EPOCH // 10 * 3, EPOCH // 10 * 4],
    milestones=[EPOCH // 10 * 2, EPOCH // 10 * 4, EPOCH // 10 * 6, EPOCH // 10 * 8], gamma=0.1)
n_epochs = EPOCH
train_loss = []
valid_loss = []
min_valid_loss = np.inf
for epoch in range(n_epochs):
    # Training
    total_train_loss = []
    model.train()  # enter training mode
    for X_batch, y_batch in trainLoader:
        h_s = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)  # 2-layers, BATCH_SIZE-batch, 16-hidden layers.
        h_c = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        h_s = h_s.to(device)
        h_c = h_c.to(device)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
        loss = loss_fn(y_pred[:, -1, :], y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    train_loss.append(np.mean(total_train_loss))
    mult_step_scheduler.step()
    # Validation
    if epoch % ValidFrequency != 0:
        continue
    total_valid_loss = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in validLoader:
            h_s = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)  # 2-layers, BATCH_SIZE-batch, 16-hidden layers.
            h_c = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)
            h_s = h_s.to(device)
            h_c = h_c.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            test_rmse = torch.sqrt(loss_fn(y_pred[:, -1, :], y_batch))
            total_valid_loss.append(test_rmse.item())
        valid_loss.append(np.mean(total_valid_loss))
        if valid_loss[-1] < min_valid_loss:
            torch.save({'epoch': epoch, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, './LSTM_March8.model')  # save model and data
            torch.save(optimizer, './LSTM_March8.optim')  # save optimizer
            min_valid_loss = valid_loss[-1]

    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((epoch + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])
    print(str(datetime.now() + timedelta(hours=0)) + ': ')
    print(log_string)  # print the log
    log('./LSTM_March8.log', log_string)  # save the log


# Author: Junran Chen
# Date: 2023-June-23
# Function: Train LSTM model.
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import datetime, timedelta
from VEstimLSTM_Mina import VEstimLSTM

INPUT_SIZE = 3
HIDDEN_SIZE = 10
BATCH_SIZE = 100
EPOCH = 5000
# DROPOUT = 0.2
LAYERS = 1  # LSTM hidden layers
LR = 0.00001  # Learning rate
ValidFrequency = 3  # Validation frequency
torch.manual_seed(20)  # set random seed.


# log function
def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


# ----------------------------Training the network---------------------------------------------------------
print(str(datetime.now() + timedelta(hours=0)) + ': ')
torch.manual_seed(0)  # set random seed.
generator1 = torch.Generator().manual_seed(42)
[cacheX_train, cachey_train] = torch.load('./Combined_TrainingTrainSet24-Jan-2024.pt')  # load data set
[cacheX_valid, cachey_valid] = torch.load('./Combined_TrainingValidSet24-Jan-2024.pt')  # load data set
log_string = 'INPUT_SIZE = 3, HIDDEN_SIZE = 10, BATCH_SIZE = 100, EPOCH = 5000, LR = 0.01, LAYERS = 1, ' \
             'ValidFrequency = 3'
log('./LSTM_Jan31.log', log_string)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # select CPU or GPU
trainLoader = data.DataLoader(data.TensorDataset(cacheX_train, cachey_train),
                              batch_size=BATCH_SIZE, drop_last=True)
validLoader = data.DataLoader(data.TensorDataset(cacheX_valid, cachey_valid),
                              batch_size=BATCH_SIZE, drop_last=True)
del cacheX_train, cachey_train, cacheX_valid, cachey_valid
min_valid_loss = 0.017719
model = VEstimLSTM(input_size=INPUT_SIZE,
                   hidden_size=HIDDEN_SIZE,
                   layers=LAYERS)
model.to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
n_epochs = EPOCH
train_loss = []
valid_loss = []
min_valid_loss = np.inf
start_epoch = -1
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[EPOCH // 10 * 2, EPOCH // 10 * 4, EPOCH // 10 * 6, EPOCH // 10 * 8], gamma=0.1)
# --------------Continue training---------------------
# model = torch.load('./LSTM_Jan31.model', map_location='cuda:0').get('model')
# train_loss = torch.load('./LSTM_Jan31.model', map_location='cuda:0').get('train_loss')
# valid_loss = torch.load('./LSTM_Jan31.model', map_location='cuda:0').get('valid_loss')
# optimizer = optim.Adam(model.parameters(), lr=LR)
# mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[EPOCH // 10 * 2, EPOCH // 10 * 4, EPOCH // 10 * 6, EPOCH // 10 * 8], gamma=0.1)
# min_valid_loss = min(valid_loss)
# start_epoch = torch.load('./LSTM_Jan31.model', map_location='cuda:0').get('epoch')
# ------------------------------------------
for epoch in range(start_epoch + 1, n_epochs):
    # Training
    total_train_loss = []
    model.train()  # enter training mode
    for X_batch, y_batch in trainLoader:
        h_s = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)  # 2-layers, BATCH_SIZE-batch, 16-hidden layers.
        h_c = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        h_s = h_s.to(device)
        h_c = h_c.to(device)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
        loss = loss_fn(y_pred[:, -1, :], y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    train_loss.append(np.mean(total_train_loss))
    mult_step_scheduler.step()
    # Validation
    if epoch % ValidFrequency != 0:
        continue
    total_valid_loss = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in validLoader:
            h_s = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)  # 2-layers, BATCH_SIZE-batch, 16-hidden layers.
            h_c = torch.zeros(LAYERS, BATCH_SIZE, HIDDEN_SIZE)
            h_s = h_s.to(device)
            h_c = h_c.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            test_rmse = torch.sqrt(loss_fn(y_pred[:, -1, :], y_batch))
            total_valid_loss.append(test_rmse.item())
        valid_loss.append(np.mean(total_valid_loss))
        if valid_loss[-1] < min_valid_loss:
            torch.save({'epoch': epoch, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss, 'mult_step_scheduler': mult_step_scheduler}, './LSTM_Jan31.model')  # save model and data
            torch.save(optimizer, './LSTM_Jan31.optim')  # save optimizer
            min_valid_loss = valid_loss[-1]
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((epoch + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])
    print(str(datetime.now() + timedelta(hours=0)) + ': ')
    print(log_string)  # print the log
    log('./LSTM_Jan31.log', log_string)  # save the log
