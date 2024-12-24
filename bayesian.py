import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.distributions import Normal, Uniform
from lstm import LSTM

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
epsilon = 1

model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
initial_state_dict = model.state_dict()
prior_mean = 0.0
prior_std = 1.5
prior = Normal(prior_mean, prior_std)


best_state_dict = None
best_distance = float('inf')


for _ in tqdm(range(num_samples)):
    sampled_state_dict = {}
    for key, value in initial_state_dict.items():
        sampled_state_dict[key] = prior.sample(value.shape)

    model.load_state_dict(sampled_state_dict)
    model.eval()

    with torch.no_grad():
        y_pred = model(x_train).squeeze(-1).numpy()
        y_pred=open_scaler.fit_transform(y_pred.reshape(-1,1))
        distance = mean_squared_error(y_train.numpy(), y_pred)

    if distance < epsilon and distance < best_distance:
        best_state_dict = sampled_state_dict
        best_distance = distance

if best_state_dict:
    model.load_state_dict(best_state_dict)
    model.eval()

with torch.no_grad():
    y_pred_test = model(x_test).squeeze(-1).numpy()
    y_test_np = y_test.numpy()
    test_mse = mean_squared_error(y_test_np, open_scaler.fit_transform(y_pred_test.reshape(-1,1)))
    test_mae = mean_absolute_error(y_test_np, open_scaler.fit_transform(y_pred_test.reshape(-1,1)))
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(open_scaler.fit_transform(y_pred_test.reshape(-1,1)), label='Predicted', alpha=0.7)
plt.title('LSTM Model Predictions After ABC Optimization')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()

print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")


