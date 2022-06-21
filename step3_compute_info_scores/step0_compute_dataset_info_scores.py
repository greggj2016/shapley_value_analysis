import numpy as np
import pandas as pd
import torch
import os
from itertools import combinations as combos
from copy import deepcopy as COPY
import pdb
import argparse
from step0_MI_library import compute_MI

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs = 1, type = str, action = "store", dest = "file")
args = parser.parse_args()
file = (args.file)[0]

df = pd.read_csv(file, delimiter = "\t", header = 0)
X = df.loc[:, df.columns[:-1]].to_numpy()
y = df.loc[:, df.columns[-1]].to_numpy()
P = len(X[0])
MI_sets = compute_MI(X, y.reshape(-1, 1))
feature_info = np.zeros(P)
subset_lists = [np.array(list(combos(range(P), i + 1))) for i in range(P)]
for i in range(len(MI_sets)):
    MI_set = np.array(MI_sets[i])
    for j in range(len(subset_lists[0])):
        feature_info_indices = np.any(subset_lists[i] == j, axis = 1)
        feature_info[j] += np.sum(MI_set[feature_info_indices]/(i + 1))

note = file.split("/")[2].split(".")
void = note.pop()
path = "info_scores/" + "".join(note) + ".txt"
pd.DataFrame(feature_info).to_csv(path, sep = "\t", header = False, index = False)