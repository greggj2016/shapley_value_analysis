#!/bin/bash
#BSUB -J step1d_compute_all_permutation_scores
#BSUB -o step1d_compute_all_permutation_scores.out
#BSUB -e step1d_compute_all_permutation_scores.err

PATH_PREFIX="PI_computers/PI_computer"
PATH_SUFFIX=".sh"
NUM_FILES=$(ls PI_computers | wc -l)
for ((i=1; i<=$NUM_FILES; i++))
do
   bsub < $PATH_PREFIX$i$PATH_SUFFIX
done