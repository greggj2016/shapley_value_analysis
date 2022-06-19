import numpy as np
import pandas as pd
import pdb
import argparse
import shap
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from copy import deepcopy as COPY

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs = 1, type = str, action = "store", dest = "file")
parser.add_argument('--method', nargs = 1, type = str, action = "store", dest = "method")
args = parser.parse_args()
file = (args.file)[0]
method = (args.method)[0]

def k_fold_CV(data, k, main_col_name, num_layers, alpha):
    X = data[0]
    y = data[1]
    kf = KFold(n_splits = k, shuffle = True)
    kf.get_n_splits(X)
    efficacies =  np.array([validate_fold(data, indices, main_col_name, 
                                          num_layers, alpha)
                            for indices in kf.split(X)]) 
    simple_model_effectiveness = efficacies[:, 0, :]
    neural_network_effectiveness = efficacies[:, 1, :]
    print(np.mean(simple_model_effectiveness, axis = 0))
    return([simple_model_effectiveness, neural_network_effectiveness])\

df = pd.read_csv(file, delimiter = "\t", header = 0)
X = df.loc[:, df.columns[:-1]].to_numpy()
y = df.loc[:, df.columns[-1]].to_numpy()

kf = KFold(n_splits = 5, shuffle = True)
kf.get_n_splits(X)
index_sets = [indices for indices in kf.split(X)]
X_train_sets = [X[indices[0]] for indices in index_sets]
y_train_sets = [y[indices[0]] for indices in index_sets]
X_test_sets = [X[indices[1]] for indices in index_sets]
y_test_sets = [y[indices[1]] for indices in index_sets]

if method == "SVM":
    from sklearn.svm import SVC as model
    custom_parameter_values = {"C": 1E8, "probability": True, "tol": 1E-8}
if method == "RF":
    from sklearn.ensemble import RandomForestClassifier as model
    custom_parameter_values = {}
if method == "GB":
    from sklearn.ensemble import GradientBoostingClassifier as model
    custom_parameter_values = {"max_depth": 5, "tol": 1E-8}

test_accuracy = []
for i in range(5):
    fitter = model(**custom_parameter_values)
    fitter.fit(X_train_sets[i], y_train_sets[i])
    y_est = fitter.predict_proba(X_test_sets[i])[:, 1]
    test_accuracy.append(pearsonr(y_test_sets[i], y_est)[0])

mean_acc = np.mean(test_accuracy)
if mean_acc <= 0.99:
    low_acc_file = open("low_accuracy_datasets.txt", "a")
    low_acc_file.write(file)
    low_acc_file.close()

permutation_scores = []
fitter = model(**custom_parameter_values)
fitter.fit(X, y)
for i in range(len(X[0])):
    X_perm = COPY(X)
    np.random.shuffle(X_perm[:, i])
    y_est_perm = fitter.predict_proba(X_perm)[:, 1]
    score = mean_acc**2 - pearsonr(y, y_est_perm)[0]**2
    permutation_scores.append(score)

fitted_function = lambda x: fitter.predict_proba(x)[:,1]
reference = np.zeros((100, len(X[0])))
explainer = shap.KernelExplainer(fitted_function, reference)
shap_values = explainer.shap_values(X, nsamples = 100)

note = file.split("/")[2].split(".")
void = note.pop()

permutation_scores = pd.DataFrame(permutation_scores)
path = "permutation_scores/" + method + "_" + "".join(note) + ".txt"
permutation_scores.to_csv(path, sep = "\t", header = False, index = False)

shap_attributions = pd.DataFrame(shap_values)
path = "SHAP_values/" + method + "_" + "".join(note) + ".txt"
shap_attributions.to_csv(path, sep = "\t", header = False, index = False)