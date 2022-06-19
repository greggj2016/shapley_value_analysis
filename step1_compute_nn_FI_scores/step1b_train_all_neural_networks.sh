#!/bin/bash
#BSUB -J step1b_train_all_neural_networks
#BSUB -o step1b_train_all_neural_networks.out
#BSUB -e step1b_train_all_neural_networks.err

PATH_PREFIX="network_trainers/network_trainer"
PATH_SUFFIX=".sh"
NUM_FILES=$(ls network_trainers  | wc -l)
for ((i=1; i<=$NUM_FILES; i++))
do
   bsub < $PATH_PREFIX$i$PATH_SUFFIX
done