import numpy as np
import pandas as pd
import torch
import pdb
import argparse
from tqdm import tqdm
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
all_attributions = np.zeros(df[df.columns[:-1]].shape)
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

    nchunks = 500
    IG_function = IntegratedGradients(network)
    baseline = torch.zeros(X.shape)
    N = 300

    errors = torch.zeros(len(X))
    attributions = torch.zeros(X.shape)
    boundaries = np.cumsum([0] + [int(len(X)/nchunks)]*(nchunks - 1))
    boundaries = np.array(boundaries.tolist() + [int(len(X))])
    for i in range(nchunks):
        out = IG_function.attribute(X[boundaries[i]:boundaries[i + 1]], 
                                    baselines = baseline[boundaries[i]:boundaries[i + 1]],
                                    method = 'riemann_trapezoid',
                                    return_convergence_delta = True,
                                    n_steps = N)
        attributions[boundaries[i]:boundaries[i + 1]] += out[0]
        errors[boundaries[i]:boundaries[i + 1]] += out[1]

    all_attributions[inds] = attributions.detach().numpy()

shap_attributions = pd.DataFrame(all_attributions)
path = "SHAP_values/" + "".join(notes) + ".txt"
shap_attributions.to_csv(path, sep = "\t", header = False, index = False)