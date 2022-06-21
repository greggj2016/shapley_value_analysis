#!/bin/bash
#BSUB -J step3b_compute_all_info_scores
#BSUB -o step3b_compute_all_info_scores.out
#BSUB -e step3b_compute_all_info_scores.err

PATH_PREFIX="info_computers/info_computer"
PATH_SUFFIX=".sh"
NUM_FILES=$(ls info_computers | wc -l)
for ((i=1; i<=$NUM_FILES; i++))
do
   bsub < $PATH_PREFIX$i$PATH_SUFFIX
done