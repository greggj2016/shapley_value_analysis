import numpy as np
import pandas as pd
import os
import pdb

if not os.path.exists("network_trainers"):
    os.mkdir("network_trainers")
if not os.path.exists("SHAP_computers"):
    os.mkdir("SHAP_computers")
if not os.path.exists("PI_computers"):
    os.mkdir("PI_computers")
if not os.path.exists("trained_networks"):
    os.mkdir("trained_networks")
if not os.path.exists("SHAP_values"):
    os.mkdir("SHAP_values")
if not os.path.exists("permutation_scores"):
    os.mkdir("permutation_scores")
if not os.path.exists("mean_accuracies"):
    os.mkdir("mean_accuracies")

data_file_names = ["../datasets/" + name for name in os.listdir("../datasets")]

def get_template(prefix):
    template = "#!/bin/bash\n"
    template += "#BSUB -J " + prefix + "\n"
    template += "#BSUB -o " + prefix + ".out\n" 
    template += "#BSUB -e " + prefix + ".err\n\n" 
    template += "source activate torch_env2\n\n" 
    return(template)

for i in range(len(data_file_names)):

    template = get_template("network_trainers/network_trainer" + str(i + 1))
    file = open("network_trainers/network_trainer" + str(i + 1) + ".sh", "w") 
    file.write(template + "python step0_train_neural_network.py --file " + data_file_names[i])
    file.close()

    template = get_template("SHAP_computers/SHAP_computer" + str(i + 1))
    file = open("SHAP_computers/SHAP_computer" + str(i + 1) + ".sh", "w") 
    file.write(template + "python step0_compute_network_shapley_values.py --file " + data_file_names[i])
    file.close()

    template = get_template("PI_computers/PI_computer" + str(i + 1))
    file = open("PI_computers/PI_computer" + str(i + 1) + ".sh", "w") 
    file.write(template + "python step0_compute_network_permutation_scores.py --file " + data_file_names[i])
    file.close()

