import numpy as np
import pandas as pd
import pdb
import argparse
import shap
import os
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from copy import deepcopy as COPY

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs = 1, type = str, action = "store", dest = "file")
parser.add_argument('--method', nargs = 1, type = str, action = "store", dest = "method")
args = parser.parse_args()
file = (args.file)[0]
test_ind_file = "../step1_compute_nn_FI_scores/test_fold_indices/" 
test_ind_file += "".join(file.split("/")[2].split("."))[:-3] + ".txt"
method = (args.method)[0]

note = file.split("/")[2].split(".")
void = note.pop()
test_path = "permutation_scores/" + method + "_" + "".join(note) + ".txt"
if os.path.exists(test_path):
    exit()

df = pd.read_csv(file, delimiter = "\t", header = 0)
X = df.loc[:, df.columns[:-1]].to_numpy()
y = df.loc[:, df.columns[-1]].to_numpy()

test_indices_df = pd.read_csv(test_ind_file, delimiter = "\t")
all_indices = np.arange(len(y))
test_indices = [ind for ind in test_indices_df.to_numpy().T]
train_indices = [np.setdiff1d(all_indices, ind) for ind in test_indices_df.to_numpy().T] 
index_sets = [inds for inds in zip(train_indices, test_indices)]
X_train_sets = [X[indices[0]] for indices in index_sets]
y_train_sets = [y[indices[0]] for indices in index_sets]
X_test_sets = [X[indices[1]] for indices in index_sets]
y_test_sets = [y[indices[1]] for indices in index_sets]

if method == "SVM":
    from sklearn.svm import SVC as model
    custom_parameter_values = {"C": 1E8, "probability": True, "tol": 1E-5}
if method == "RF":
    from sklearn.ensemble import RandomForestClassifier as model
    custom_parameter_values = {}
if method == "GB":
    from sklearn.ensemble import GradientBoostingClassifier as model
    custom_parameter_values = {"max_depth": 5, "tol": 1E-8}

test_accuracy = []
fitters = []
for i in range(5):
    fitter = model(**custom_parameter_values)
    fitter.fit(X_train_sets[i], y_train_sets[i])
    fitters.append(COPY(fitter))
    y_est = fitter.predict_proba(X_test_sets[i])[:, 1]
    test_accuracy.append(pearsonr(y_test_sets[i], y_est)[0])

mean_acc = np.mean(test_accuracy)
path = "mean_accuracies/" + method + "_" + "".join(note) + "_" + str(mean_acc) + ".txt"
low_acc_file = open(path, "w")
low_acc_file.close()

all_permutation_scores = []
shap_values = np.zeros(X.shape)
for i, acc, fitter in zip(np.arange(5), test_accuracy, fitters):

    X, y, fold =  X_test_sets[i], y_test_sets[i], test_indices[i]
    fitted_function = lambda x: fitter.predict_proba(x)[:,1]
    reference = np.zeros((100, len(X[0])))
    explainer = shap.KernelExplainer(fitted_function, reference)
    shap_values[fold] = explainer.shap_values(X, nsamples = 100)

    permutation_scores = []
    for i in range(len(X[0])):
        X_perm = COPY(X)
        np.random.shuffle(X_perm[:, i])
        y_est_perm = fitter.predict_proba(X_perm)[:, 1]
        score = acc**2 - pearsonr(y, y_est_perm)[0]**2
        permutation_scores.append(score)
    all_permutation_scores.append(permutation_scores)

permutation_scores = pd.DataFrame(np.mean(all_permutation_scores, axis = 0))
path = "permutation_scores/" + method + "_" + "".join(note) + ".txt"
permutation_scores.to_csv(path, sep = "\t", header = False, index = False)

shap_attributions = pd.DataFrame(shap_values)
path = "SHAP_values/" + method + "_" + "".join(note) + ".txt"
shap_attributions.to_csv(path, sep = "\t", header = False, index = False)