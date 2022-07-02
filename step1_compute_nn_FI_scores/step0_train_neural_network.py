import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from copy import deepcopy as COPY
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

kf = KFold(n_splits = 5, shuffle = True)
kf.get_n_splits(X)
index_sets = [indices for indices in kf.split(X)]
X_train_sets = [X[indices[0]] for indices in index_sets]
y_train_sets = [y[indices[0]] for indices in index_sets]
X_test_sets = [X[indices[1]] for indices in index_sets]
y_test_sets = [y[indices[1]] for indices in index_sets]
train_test_sets = zip(X_train_sets, X_test_sets, y_train_sets, y_test_sets)

test_accuracy = []
for X_train, X_test, y_train, y_test in train_test_sets:

    hidden_size = 100
    hyperparameters = [max_epochs, batch_size, learning_rate, lr_coefs]
    num_layers = 2
    ffn_params = [len(X_train[0]), hidden_size, num_layers, torch.tensor(X_train).float()]
    network = ffn(*ffn_params)
    model_name = "neural network"
    is_binary = True
    is_low_count_integer = True
    col_name = "Hibachi Phenotype"
    alpha = 1
    num_attempts = 5

    data = [X_train, X_test, y_train, y_test, 1000]
    effectiveness, model = retrain_many_networks(data, hyperparameters, COPY(network), model_name, 
                                                 is_binary, ffn, is_low_count_integer, 
                                                 col_name, ffn_params, alpha, num_attempts)

    if effectiveness[0] > 0.9:
        test_accuracy.append(effectiveness[0])
    else:
        hidden_size = 400
        hyperparameters = [max_epochs, batch_size, learning_rate, lr_coefs]
        num_layers = 2
        ffn_params = [len(X_train[0]), hidden_size, num_layers, torch.tensor(X_train).float()]
        network = ffn(*ffn_params)
        model_name = "neural network"
        is_binary = True
        is_low_count_integer = True
        col_name = "Hibachi Phenotype"
        alpha = 1
        num_attempts = 5

        data = [X_train, X_test, y_train, y_test, 1000]
        effectiveness, model = retrain_many_networks(data, hyperparameters, COPY(network), model_name, 
                                                     is_binary, ffn, is_low_count_integer, 
                                                     col_name, ffn_params, alpha, num_attempts)
        test_accuracy.append(effectiveness[0])



hyperparameters = [max_epochs, batch_size, learning_rate, lr_coefs]
num_layers = 2
ffn_params = [len(X[0]), hidden_size, num_layers, torch.tensor(X).float()]
network = ffn(*ffn_params)
model_name = "neural network"
is_binary = True
is_low_count_integer = True
col_name = "Hibachi Phenotype"
alpha = 1
num_attempts = 5

final_data = [X, X, y, y, 1000]
effectiveness, final_model = retrain_many_networks(final_data, hyperparameters, COPY(network), model_name, 
                                                   is_binary, ffn, is_low_count_integer, 
                                                   col_name, ffn_params, alpha, num_attempts)

notes = file.split("/")[2].split(".")
void = notes.pop()
path = "trained_networks/" + "".join(notes) + ".pth"
torch.save(final_model.state_dict(), path)

mean_acc = np.mean(test_accuracy)
path = "mean_accuracies/" + "neural_network_" + "".join(notes) + "_" + str(mean_acc) + ".txt"
low_acc_file = open(path, "w")
low_acc_file.close()
