import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from lstm import LSTM
import time

stock=pd.read_csv("amzn.csv")
open_price = stock[['Open']]
open_scaler =MinMaxScaler(feature_range=(-1, 1))
data = open_scaler.fit_transform(open_price['Open'].values.reshape(-1, 1))

def create_sequences(datafile, seq_length):
    xs, ys = [], []
    for i in range(len(datafile) - seq_length):
        xs.append(datafile[i:i+seq_length])
        ys.append(datafile[i+seq_length])
    return np.array(xs), np.array(ys)

train, test=train_test_split(data, test_size=0.2, shuffle=False)
seq_len=50
x_train, y_train = create_sequences(train, seq_len)
x_test, y_test = create_sequence(test, seq_len)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

input_dim = 1
hidden_dim = 100
layer_dim = 1
output_dim = 1
num_samples = 100

model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

import time
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
loss_val=[]

for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, open_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    loss_val.append(loss.item())
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
