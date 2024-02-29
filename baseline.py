import numpy as np
from scipy import signal
import data_loader
import matplotlib.pyplot as plt

x, y = data_loader.lab_data_loader()
b, a = signal.butter(8, [0.1, 0.8], btype='bandpass')
filtered_x = signal.filtfilt(b, a, x[:, :2000])
fft_filtered_x = np.abs(np.fft.fft(filtered_x))[:, :1000]


import torch
from torch.nn import Linear, ReLU, Softmax, Dropout, Sequential
from torch.optim import SGD
from torch.nn.functional import cross_entropy, one_hot

model = Sequential(
    Linear(1000, 300),
    ReLU(),
    Dropout(0.2),
    Linear(300, 100),
    ReLU(),
    Dropout(0.2),
    Linear(100, 6),
    Softmax()
)


def shuffle(x, y):
    idx = torch.randperm(x.shape[0])
    return x[idx, :], y[idx]

x, y = shuffle(fft_filtered_x, y[:,0])

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.int64)
# y = one_hot(y)
splitter = int(0.8*x.shape[0])
x_train = x[:splitter, :]
x_val = x[splitter:, :]
y_train = y[:splitter]
y_val = y[splitter:]

optimizer = SGD(model.parameters(), lr=1e-2)
total_epoch = 10
batch_size = 32


def train_loop(x_train, y_train, x_val, y_val):
    for epoch in range(total_epoch):
        x_train, y_train = shuffle(x_train, y_train)
        loss_train = []
        for i in range(x_train.shape[0] // batch_size):
            x_iter = x_train[i*batch_size:i*batch_size+batch_size, :]
            y_iter = y_train[i*batch_size:i*batch_size+batch_size]
            model.train()
            pred = model(x_iter)
            loss_train.append(cross_entropy(pred, y_iter))
            optimizer.zero_grad()
            loss_train[i].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_val)
            loss_val = cross_entropy(pred, y_val)
        print(f'train_loss = {sum(loss_train)/len(loss_train)}, val_loss = {loss_val}')


train_loop(x_train, y_train, x_val, y_val)
