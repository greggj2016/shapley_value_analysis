import numpy as np
import pandas as pd
import os
import pdb

pdb.set_trace()

paths = os.listdir("mean_accuracies")
acc_info = [p.split('.txt')[0].split('_') for p in paths]
acc_info = pd.DataFrame([[i[3], "0." + i[5][1:], i[7] == "corr", i[-1], float(i[-1])] for i in acc_info])
acc_info.columns = ["P", "a", "is correlated", "number of effects", "mean test efficacy"]
acc_info.to_csv("accuracy_info.txt", sep = "\t", header = True, index = False)
print(np.min(acc_info["mean test efficacy"]))
print(np.mean(acc_info["mean test efficacy"]))