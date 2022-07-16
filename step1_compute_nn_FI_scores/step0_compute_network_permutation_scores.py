import numpy as np
import pandas as pd
import torch
import os
from scipy.stats import pearsonr
from copy import deepcopy as COPY
import pdb
import argparse
from captum.attr import IntegratedGradients
from step0_neural_network_library import ffn

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs = 1, type = str, action = "store", dest = "file")
parser.add_argument('--baseline', nargs = '?', type = str, action = "store", dest = "baseline")
args = parser.parse_args()
file = (args.file)[0]
test_inds_file = "test_fold_indices/" + "".join(file.split("/")[2].split("."))[:-3] + ".txt"
bs = args.baseline

df = pd.read_csv(file, delimiter = "\t", header = 0)
test_inds = pd.read_csv(test_inds_file, delimiter = "\t", header = 0)
all_permutation_scores = []
for i in range(5):
    
    fold = "fold" + str(i + 1)
    inds = test_inds.loc[:, fold].to_numpy()
    X = torch.tensor(df.loc[inds, df.columns[:-1]].to_numpy()).float()
    y = torch.tensor(df.loc[inds, df.columns[-1]].to_numpy()).float()

    # ffn_params = [input_dim, hidden_layer_dim, num_hidden_layers, standardization_content]
    try:
        ffn_params = [len(X[0]), 100, 2, X]
        network = ffn(*ffn_params)
        notes = file.split("/")[2].split(".")
        void = notes.pop()
        path = "trained_networks/" + "".join(notes) + "_" + fold + ".pth"
        network.load_state_dict(torch.load(path))
    except:
        ffn_params = [len(X[0]), 400, 2, X]
        network = ffn(*ffn_params)
        notes = file.split("/")[2].split(".")
        void = notes.pop()
        path = "trained_networks/" + "".join(notes) + "_" + fold + ".pth"
        network.load_state_dict(torch.load(path))

    X_copy = COPY(X)
    permutation_scores = np.zeros(len(X[0]))
    for i in range(len(X[0])):
        y_est = network(X).reshape(-1)
        shuffled_indices = np.random.choice(np.arange(len(X)), size = len(X), replace = False)
        X[:, i] = X[torch.tensor(shuffled_indices), i]
        yp_est = network(X).reshape(-1)
        X = COPY(X_copy)
        r2 = pearsonr(y.detach().numpy(), y_est.detach().numpy())[0]**2
        r2p = pearsonr(y.detach().numpy(), yp_est.detach().numpy())[0]**2
        permutation_scores[i] = r2 - r2p
    all_permutation_scores.append(permutation_scores)

permutation_scores = pd.DataFrame(np.mean(all_permutation_scores, axis = 0))
path = "permutation_scores/" + "".join(notes) + ".txt"
permutation_scores.to_csv(path, sep = "\t", header = False, index = False)