import numpy as np
import pandas as pd
import os
import pdb

if not os.path.exists("importance_computers"):
    os.mkdir("importance_computers")
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

    template = get_template("importance_computers/importance_computers" + str(i + 1))
    file = open("importance_computers/importance_computers" + str(i + 1) + ".sh", "w") 
    file.write(template)
    file.write("python step0_analyze_model.py --file " + data_file_names[i] + " --method SVM\n")
    file.write("python step0_analyze_model.py --file " + data_file_names[i] + " --method RF\n")
    file.write("python step0_analyze_model.py --file " + data_file_names[i] + " --method GB\n")
    file.close()

