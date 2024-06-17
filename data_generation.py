"""
this code generate single-cell gene expression data from a nonlinear SDE.
"""

import numpy as np

# there are totally 40 GRNs in SINCERETIES paper
# 10 GRNs in E coli with 10 genes, 10 GRNs in E coli with 20 genes
# 10 GRNs in yeast with 10 genes, 10 GRNs in yeast with 20 genes

# no GRN has nonzero terms on the diagonal

def generation(network, time_points, sim_num):
    # network can take value in range(40), meaning the true GRN number
    # time_points is a list of non-negative time points, showing when
    # we want to record the expression levels
    # sim_num is the number of cells measured, i.e., the number of 
    # trajectories generated
    
    # read the true GRN
    if network < 10:
        name = 'SINC/Ecoli10genes_' + str(network % 10) + '.npy'
    elif network < 20:
        name = 'SINC/Ecoli20genes_' + str(network % 10) + '.npy'
    elif network < 30:
        name = 'SINC/Yeast10genes_' + str(network % 10) + '.npy'
    elif network < 40:
        name = 'SINC/Yeast20genes_' + str(network % 10) + '.npy'
    with open(name, 'rb') as f:
        A = np.load(f) # the GRN
    gene_num = len(A) # number of genes
    
    
    
    
    time_step = 0.01 
    tp_num = len(time_points) # number of time points
    V = 30
    beta = 1
    theta = 0.2
    sigma = 0.1
    
    data = np.zeros((tp_num, sim_num, gene_num))
    curr_time = 0.0
    curr_exp = np.random.rand(sim_num, gene_num)
    while curr_time <= time_points[-1] + 1e-6:
        for i in range(tp_num):
            if abs(curr_time - time_points[i]) < 1e-6:
                data[i, :, :] = curr_exp
        curr_time += time_step
        diff = np.zeros((sim_num, gene_num))
        for sim in range(sim_num):
            for j in range(gene_num):
                pro = beta
                for i in range(gene_num):
                    pro *= 1 + A[i, j] * curr_exp[sim, i] / (1 + curr_exp[sim, i])
                diff[sim, j] = V * time_step * pro
        diff -= V * time_step * theta * curr_exp
        diff += np.multiply(curr_exp, np.random.normal(0, \
                sigma * np.sqrt(time_step), (sim_num, gene_num)))
        curr_exp += diff      
    
    return A, data # A is the true GRN, data is the expression level data
    # of size number of time points * number of cells * number of genes
    
