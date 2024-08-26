"""
this code compares five GRN inference methods on THP-1 time series data
"""

from evaluation import auroc_auprc
from sincerities import sincer
from wendy_alg import wendy_alg
import numpy as np
from xgbgrn import get_importances
from GENIE3 import GENIE3
import warnings
warnings.filterwarnings("ignore")

gene_list = ['BCL6', 'CEBPB', 'CEBPD', 'EGR2', 'FLI1', 'HOXA10', 'HOXA13', 
             'IRF8', 'MAFB', 'MYB', 'NFATC1', 'NFE2L1', 'PPARD', 'PPARG', 
             'PRDM1', 'RUNX1', 'SNAI3', 'TCFL5', 'TFPT', 'UHRF1']

AUROC = [0.0] * 4
AUPR = [0.0] * 4
with open('THP1/THP1_A.npy', 'rb') as f:
    A = np.load(f)
with open('THP1/THP1_data.npy', 'rb') as f:
    data = np.load(f)

all_time = [0, 1, 6, 12, 24, 48, 72, 96] # all time points
sim_num = data.shape[1] # number of cells per time point

wt0 = 0 # choose the best time points
wt1 = 5
st0 = 1
st1 = 7
nt0 = 0
nt1 = 2
gt = 4

covdyn = wendy_alg(data[wt0], data[wt1])
AUROC[0], AUPR[0] = auroc_auprc(A, covdyn)

sinc_time = all_time[st0:st1+1]
sinc_data = data[st0:st1+1]
sinc_res = sincer(sinc_data, sinc_time)
AUROC[1], AUPR[1] = auroc_auprc(A, sinc_res)

nl_time = all_time[nt0:nt1+1]
nl_data = data[nt0:nt1+1]
bulk_data = [np.average(nl_data[:, :sim_num//2, :], axis=1), 
             np.average(nl_data[:, sim_num//2:, :], axis=1)]
nl_time_full = [np.arange(len(nl_time))] * 2
nlode = get_importances(bulk_data, nl_time_full, alpha='from_data')
AUROC[2], AUPR[2] = auroc_auprc(A, nlode)

genie = GENIE3(data[gt])
AUROC[3], AUPR[3] = auroc_auprc(A, genie)

print('results for WENDY, SINCERITIES, NonlinearODEs, GENIE3')
print('AUROC: ', AUROC)
print('AUPR: ', AUPR)


