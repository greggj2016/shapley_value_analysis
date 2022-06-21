import numpy as np
import pandas as pd
import os
import pdb

if not os.path.exists("info_computers"):
    os.mkdir("info_computers")
if not os.path.exists("info_scores"):
    os.mkdir("info_scores")
data_file_names = ["../datasets/" + name for name in os.listdir("../datasets")]

def get_template(prefix):
    template = "#!/bin/bash\n"
    template += "#BSUB -J " + prefix + "\n"
    template += "#BSUB -o " + prefix + ".out\n" 
    template += "#BSUB -e " + prefix + ".err\n" 
    template += "#BSUB -M 25000MB\n"
    template += '#BSUB -R "span[hosts=1] rusage[mem=25000MB]"\n'
    template += "source activate torch_env2\n\n" 
    return(template)

for i in range(len(data_file_names)):

    template = get_template("info_computers/info_computer" + str(i + 1))
    file = open("info_computers/info_computer" + str(i + 1) + ".sh", "w") 
    file.write(template + "python step0_compute_dataset_info_scores.py --file " + data_file_names[i])
    file.close()

