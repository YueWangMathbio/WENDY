# -*- coding: utf-8 -*-
"""
this code compares five GRN inference methods on hESC time series data
"""

from evaluation import directed_evaluation
from sincerities import sincer
from wendy_alg import wendy_alg
import numpy as np
from xgbgrn import get_importances
from GENIE3 import GENIE3
import warnings
warnings.filterwarnings("ignore")

gene_list = ['POU5F1', 'AEBP2', 'MIER1', 'SMAD2', 'ZNF652', 'ZFX', 'TERF1', 
             'SOX11', 'BBX', 'ZFP42', 'TULP4', 'ZNF471', 'ARID4B', 'ZNF483', 
             'SHOX', 'ZNF587', 'ZFP14', 'CEBPZ']

AUROC = [0.0] * 4
AUPR = [0.0] * 4

data = []
for i in range(6):
    filename = 'hESC/hESC_data%d.npy' % i
    with open(filename, 'rb') as f:
        temp = np.load(f)
        data.append(temp)
with open('hESC/hESC_A.npy', 'rb') as f:
    A = np.load(f)


all_time = [0, 12, 24, 36, 72, 96] # all time points

wt0 = 1 # choose the best time points
wt1 = 2
st0 = 0
st1 = 2
nt0 = 2
nt1 = 3
gt = 0

covdyn = wendy_alg(data[wt0], data[wt1])
covdyn = np.abs(covdyn)
AUROC[0], AUPR[0] = directed_evaluation(A, covdyn)
      
sinc_time = all_time[st0:st1+1]
sinc_data = data[st0:st1+1]
sinc_res = sincer(data, sinc_time)
sinc_res = np.abs(sinc_res)
AUROC[1], AUPR[1] = directed_evaluation(A, sinc_res)

nl_time = all_time[nt0:nt1+1]
nl_data = data[nt0:nt1+1]
bulk_data = np.zeros((2, nt1+1-nt0, data[0].shape[-1]))
for i in range(nt0, nt1+1):
    sim_num = data[i].shape[0]
    bulk_data[0, i-nt0, :] = np.average(data[i][:sim_num//2, :], axis=0)
    bulk_data[1, i-nt0, :] = np.average(data[i][sim_num//2:, :], axis=0)
nl_time_full = [np.arange(len(nl_time))] * 2
nlode = get_importances(bulk_data, nl_time_full, alpha='from_data')
nlode = np.abs(nlode)
AUROC[2], AUPR[2] = directed_evaluation(A, nlode)

genie = GENIE3(data[gt])
genie = np.abs(genie)
AUROC[3], AUPR[3] = directed_evaluation(A, genie)

print('results for WENDY, SINCERITIES, NonlinearODEs, GENIE3')
print('AUROC: ', AUROC)
print('AUPR: ', AUPR)
