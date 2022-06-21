import numpy as np 
import pandas as pd
import os
from scipy.stats import spearmanr
import pdb

info_prefix = "info_scores"
PI_prefix = "../step2_compute_svm_FI_scores/permutation_scores"
SV_prefix = "../step2_compute_svm_FI_scores/SHAP_values"
def imp(path): return(pd.read_csv(path, delimiter = "\t", header = None).to_numpy())

info_scores_paths = [info_prefix + "/" + i for i in os.listdir("info_scores")]
no_corr_info_score_paths = np.sort([i for i in info_scores_paths if "no_corr" in i])
corr_info_score_paths = np.sort([i for i in info_scores_paths if "no_corr" not in i])

PI_score_paths = [PI_prefix + "/" + i for i in os.listdir(PI_prefix)]
SVM_no_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'SVM' in i and "no_corr" in i])
SVM_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'SVM' in i and "no_corr" not in i])
RF_no_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'RF' in i and "no_corr" in i])
RF_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'RF' in i and "no_corr" not in i])
GB_no_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'GB' in i and "no_corr" in i])
GB_corr_PI_score_paths = np.sort([i for i in PI_score_paths if 'GB' in i and "no_corr" not in i])

SV_score_paths = [SV_prefix + "/" + i for i in os.listdir(SV_prefix)]
SVM_no_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'SVM' in i and "no_corr" in i])
SVM_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'SVM' in i and "no_corr" not in i])
RF_no_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'RF' in i and "no_corr" in i])
RF_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'RF' in i and "no_corr" not in i])
GB_no_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'GB' in i and "no_corr" in i])
GB_corr_SV_score_paths = np.sort([i for i in SV_score_paths if 'GB' in i and "no_corr" not in i])

no_corr_info_scores = np.concatenate([imp(i) for i in no_corr_info_score_paths])
corr_info_scores = np.concatenate([imp(i) for i in corr_info_score_paths])

SVM_no_corr_PI_scores = np.concatenate([imp(i) for i in SVM_no_corr_PI_score_paths])
SVM_corr_PI_scores = np.concatenate([imp(i) for i in SVM_corr_PI_score_paths])
RF_no_corr_PI_scores = np.concatenate([imp(i) for i in RF_no_corr_PI_score_paths])
RF_corr_PI_scores = np.concatenate([imp(i) for i in RF_corr_PI_score_paths])
GB_no_corr_PI_scores = np.concatenate([imp(i) for i in GB_no_corr_PI_score_paths])
GB_corr_PI_scores = np.concatenate([imp(i) for i in GB_corr_PI_score_paths])

SVM_no_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in SVM_no_corr_SV_score_paths])
SVM_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in SVM_corr_SV_score_paths])
RF_no_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in RF_no_corr_SV_score_paths])
RF_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in RF_corr_SV_score_paths])
GB_no_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in GB_no_corr_SV_score_paths])
GB_corr_SV_scores = np.concatenate([np.mean(np.abs(imp(i)), axis = 0) for i in GB_corr_SV_score_paths])

r1 = spearmanr(GB_corr_PI_scores, corr_info_scores)[0]
r2 = spearmanr(SVM_corr_PI_scores, corr_info_scores)[0]
r3 = spearmanr(RF_corr_PI_scores, corr_info_scores)[0]

r4 = spearmanr(GB_corr_SV_scores, corr_info_scores)[0]
r5 = spearmanr(SVM_corr_SV_scores, corr_info_scores)[0]
r6 = spearmanr(RF_corr_SV_scores, corr_info_scores)[0]

r7 = spearmanr(GB_no_corr_PI_scores, no_corr_info_scores)[0]
r8 = spearmanr(SVM_no_corr_PI_scores, no_corr_info_scores)[0]
r9 = spearmanr(RF_no_corr_PI_scores, no_corr_info_scores)[0]

r10 = spearmanr(GB_no_corr_SV_scores, no_corr_info_scores)[0]
r11 = spearmanr(SVM_no_corr_SV_scores, no_corr_info_scores)[0]
r12 = spearmanr(RF_no_corr_SV_scores, no_corr_info_scores)[0]

print("correlated data PI vs info: " + str(r1)[0:8] + ", " + str(r2)[0:8] + ", " + str(r3)[0:8])
print("correlated data SV vs info: " + str(r4)[0:8] + ", " + str(r5)[0:8] + ", " + str(r6)[0:8])
print("uncorrelated data PI vs info: " + str(r7)[0:8] + ", " + str(r8)[0:8] + ", " + str(r9)[0:8])
print("uncorrelated data SV vs info: " + str(r10)[0:8] + ", " + str(r11)[0:8] + ", " + str(r12)[0:8])
