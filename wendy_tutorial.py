# -*- coding: utf-8 -*-
"""
This file explains how to use WENDY method to calculate the gene regulatory
network from single-cell level gene expression data, measured at two points 
after some general interventions, where the joint distribution of these two
time points is unknown.
"""

from sklearn.covariance import GraphicalLassoCV
from wendy_solver import RegRelSolver
import numpy as np
import warnings
warnings.filterwarnings("ignore") # use this to ignore warnings

def wendy_alg(data0, data1, user_mask=None): # each of input data is numpy array or 
# similar array, with the shape of cell number * gene number
    gene_num = data0.shape[1] # number of genes
    temp = GraphicalLassoCV().fit(data0)
    invcov_0 = np.around(temp.precision_, decimals=3)
    k0 = np.linalg.inv(invcov_0) # covariance matrix at time 0
    temp = GraphicalLassoCV().fit(data1)
    invcov_1 = np.around(temp.precision_, decimals=3)
    kt = np.linalg.inv(invcov_1) # covariance matrix at time t
    lam = 0.0 # coefficient of an L2 regularizer, suggested to be 0
    weight = np.ones((gene_num, gene_num)) 
    for i in range(gene_num):
        weight[i, i] = 0.0 # declare that diagonal elements do not count in matching
    solver = RegRelSolver(k0, kt, lam, weight, user_mask) # call the solver to calculate A
    my_at = solver.fit()
    covdyn = np.around(my_at, decimals=3)
    return covdyn # the calculated GRN matrix

"""
if you have raw single-cell RNA sequencing (scRNAseq) data:
use scanpy or other packages to extract expression data at each time point.
remove genes that only appear in a few cells, and cells that only a few genes
are measured.
replace each value x by log(1+x).
for each cell (row), normalize its sum, so that each cell has the same 
total expression level. 
"""

# this is an example of using WENDY
data0 = np.load('example_data_0.npy')
data1 = np.load('example_data_1.npy')
# here are two data sets, each is a numpy array of size 100 * 10,
# meaning the expression levels of 10 genes for 100 cells 
sim_num, gene_num = data0.shape
mask = None

"""
# if you know some gene i cannot regulate some gene j from biology,
# add such forbbiden edges to the following list, and uncomment this section
# otherwise, just ignore this section
forbidden_edge = [[0, 1]] 
# if we know gene 0 cannot regulate gene 1, then we force the solved GRN A
# to have A[0, 1] = 0.0
mask = np.ones((gene_num, gene_num))
for [x, y] in forbidden_edge:
    mask[x, y] = 0.0
mask = np.isclose(mask, 1.0)
"""

grn = wendy_alg(data0, data1, user_mask=mask) # this is the calculate GRN.
# grn[i, j] means the regulation strength of gene i on gene j
# positive means activation, negative means inhibition.

print(grn)
