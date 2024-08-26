"""
this code compares five GRN inference methods on DREAM4 time series data
"""

from evaluation import auroc_auprc
from sincerities import sincer
from wendy_alg import wendy_alg
import numpy as np
from xgbgrn import get_importances
from GENIE3 import GENIE3
from dynGENIE3 import dynGENIE3
import warnings
warnings.filterwarnings("ignore")

for dataset in range(2):
    AUROC = [0.0] * 5
    AUPR = [0.0] * 5
    for network in range(5):
        gene_num = dataset * 90 + 10 # either 10 or 100
        grn_name = 'DREAM4/DREAM4_A_' + str(gene_num) + '_' + str(network) + '.npy'
        with open(grn_name, 'rb') as f:
            A = np.load(f)
        data_name = 'DREAM4/DREAM4_data_' + str(gene_num) + '_' + str(network) + '.npy'
        with open(data_name, 'rb') as f:
            data = np.load(f)
        sim_num = data.shape[1]
        
        if gene_num == 10:
            wt0 = 9
            wt1 = 17
            st0 = 0
            st1 = 7
            nt0 = 6
            nt1 = 8
            gt = 9
            dt0 = 0
            dt1 = 17
        else:
            wt0 = 6
            wt1 = 11
            st0 = 9
            st1 = 14
            nt0 = 4
            nt1 = 11
            gt = 7
            dt0 = 0
            dt1 = 16
        
        
        covdyn = wendy_alg(data[wt0], data[wt1])
        covdyn = np.abs(covdyn)
        aurocp, auprp = auroc_auprc(A, covdyn)
        AUROC[0] += aurocp / 5
        AUPR[0] += auprp / 5
        
        sinc_time = np.linspace(st0*50, st1*50, st1-st0+1)
        sinc_data = data[st0:st1+1]
        sinc_res = sincer(sinc_data, sinc_time)
        sinc_res = np.abs(sinc_res)
        aurocp, auprp = auroc_auprc(A, sinc_res)
        AUROC[1] += aurocp / 5
        AUPR[1] += auprp / 5
        
        nl_time = np.linspace(nt0*50, nt1*50, nt1-nt0+1)
        nl_data = data[nt0:nt1+1]
        bulk_data = [nl_data[:, i, :] for i in range(sim_num)]
        nl_time_full = [np.arange(len(nl_time))] * sim_num
        nlode = get_importances(bulk_data, nl_time_full, alpha='from_data')
        nlode = np.abs(nlode)
        aurocp, auprp = auroc_auprc(A, nlode)
        AUROC[2] += aurocp / 5
        AUPR[2] += auprp / 5
        
        genie = GENIE3(data[gt])
        aurocp, auprp = auroc_auprc(A, genie)
        AUROC[3] += aurocp / 5
        AUPR[3] += auprp / 5
        
        dg_data = data[dt0:dt1+1]
        dg_time = np.linspace(dt0*50, dt1*50, dt1-dt0+1)
        dg_data_full = [dg_data[:, i, :] for i in range(sim_num)]
        dg_time_full = [dg_time for i in range(sim_num)]
        dg3 = dynGENIE3(dg_data_full, dg_time_full)
        aurocp, auprp = auroc_auprc(A, dg3)
        AUROC[4] += aurocp / 5
        AUPR[4] += auprp / 5
        
    print('results for WENDY, SINCERITIES, NonlinearODEs, GENIE3, dynGENIE3')
    print('data set DREAM4 %d genes' % gene_num)
    print('AUROC: ', AUROC)
    print('AUPR: ', AUPR)
    
    
