import numpy as np
import pandas as pd
import torch
import pdb
import argparse

from step0_neural_network_library import retrain_many_networks
from step0_neural_network_library import k_fold_CV
from step0_neural_network_library import ffn

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs = 1, type = str, action = "store", dest = "file")
args = parser.parse_args()
file = (args.file)[0]

df = pd.read_csv(file, delimiter = "\t", header = 0)
X = df.loc[:, df.columns[:-1]].to_numpy()
y = df.loc[:, df.columns[-1]].to_numpy()

main_col_name = "pheno"
max_epochs = 1000 
batch_size = 1000
n_layers = 0
learning_rate = 0.02
lr_coefs = [1, 0.1, 0.01, 0.001, 0.0001]
train_indices = np.random.choice(np.arange(len(X)), size = int(0.8*len(X)), replace = False) 
test_indices = np.setdiff1d(np.arange(len(X)), train_indices)
X_train = X[train_indices] 
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
data = [X_train, X_test, y_train, y_test, 1000]
hyperparameters = [max_epochs, batch_size, learning_rate, lr_coefs]
num_layers = 2
ffn_params = [len(X_train[0]), 80, num_layers, torch.tensor(X_train).float()]
network = ffn(*ffn_params)
model_name = "LR"
is_binary = True
is_low_count_integer = True
col_name = "HF"
alpha = 1
num_attempts = 5
effectiveness, model = retrain_many_networks(data, hyperparameters, network, model_name, 
                                             is_binary, ffn, is_low_count_integer, 
                                             col_name, ffn_params, alpha, num_attempts)

notes = file.split("/")[2].split(".")
void = notes.pop()
path = "trained_networks/" + "".join(notes) + ".pth"
torch.save(model.state_dict(), path)
