#!/bin/bash
#BSUB -J step1c_compute_all_shapley_values
#BSUB -o step1c_compute_all_shapley_values.out
#BSUB -e step1c_compute_all_shapley_values.err

PATH_PREFIX="SHAP_computers/SHAP_computer"
PATH_SUFFIX=".sh"
NUM_FILES=$(ls SHAP_computers | wc -l)
for ((i=1; i<=$NUM_FILES; i++))
do
   bsub < $PATH_PREFIX$i$PATH_SUFFIX
done