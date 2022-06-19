#!/bin/bash
#BSUB -J step2b_train_all_models
#BSUB -o step2b_train_all_models.out
#BSUB -e step2b_train_all_models.err

PATH_PREFIX="importance_computers/importance_computers"
PATH_SUFFIX=".sh"
NUM_FILES=$(ls importance_computers  | wc -l)
for ((i=1; i<=$NUM_FILES; i++))
do
   bsub < $PATH_PREFIX$i$PATH_SUFFIX
done