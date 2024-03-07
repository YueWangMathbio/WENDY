# -*- coding: utf-8 -*-
"""
This is the main function of wendy method, used in testing. 
"""

from sklearn.covariance import GraphicalLassoCV
from wendy_solver import RegRelSolver
import numpy as np

def wendy_alg(data0, data1): # each of input is numpy array or similar array, 
# with the shape of cell number * gene number
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
    solver = RegRelSolver(k0, kt, lam, weight) # call the solver to calculate A
    my_at = solver.fit()
    covdyn = np.around(my_at, decimals=3)
    return covdyn # the calculated GRN matrix