"""
this code compares five GRN inference methods on SINC time series data
using parallel computing to accelerate
"""

import functools
from itertools import product
from multiprocessing import Pool
import multiprocessing
from data_generation import generation
from evaluation import directed_evaluation
from sincerities import sincer
from wendy_alg import wendy_alg
from GENIE3 import GENIE3
import numpy as np
from xgbgrn import get_importances
from dynGENIE3 import dynGENIE3
import time
import warnings
warnings.filterwarnings("ignore")

def wendy_wrap(unused_index, network, sim_num, time_points):
    del unused_index
    A, data = generation(network, time_points, sim_num)
    t0 = 0
    t1 = 1
    wendy = wendy_alg(data[t0, data[t1]])
    auroc_dir_wendy, aupr_dir_wendy = directed_evaluation(A, wendy, print_res=False)
    return auroc_dir_wendy, aupr_dir_wendy

def wendy_rep(rep, net_num, sim_num, time_points):
    full_combination = list(product(np.arange(rep), np.arange(net_num)))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        res = p.starmap(functools.partial(wendy_wrap, sim_num=sim_num, time_points=time_points), full_combination)
    return res

def sinc_wrap(unused_index, network, sim_num, time_points):
    del unused_index
    A, data = generation(network, time_points, sim_num)
    sinc = sincer(data, time_points)
    auroc_dir_sinc, aupr_dir_sinc = directed_evaluation(A, sinc, print_res=False)
    return auroc_dir_sinc, aupr_dir_sinc

def sinc_rep(rep, net_num, sim_num, time_points):
    full_combination = list(product(np.arange(rep), np.arange(net_num)))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        res = p.starmap(functools.partial(sinc_wrap, sim_num=sim_num, time_points=time_points), full_combination)
    return res

def nlode_wrap(unused_index, network, sim_num, time_points):
    del unused_index
    A, data = generation(network, time_points, sim_num)
    bulk_data = [np.average(data[:, :sim_num//2, :], axis=1), np.average(data[:, sim_num//2:, :], axis=1)]
    nl_time = [np.arange(len(time_points))] * 2
    nlode = get_importances(bulk_data, nl_time, alpha='from_data')
    auroc_dir_nlode, aupr_dir_nlode = directed_evaluation(A, nlode, print_res=False)
    return auroc_dir_nlode, aupr_dir_nlode

def nlode_rep(rep, net_num, sim_num, time_points):
    full_combination = list(product(np.arange(rep), np.arange(net_num)))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        res = p.starmap(functools.partial(nlode_wrap, sim_num=sim_num, time_points=time_points), full_combination)
    return res

def genie_wrap(unused_index, network, sim_num, time_points):
    del unused_index
    A, data = generation(network, time_points, sim_num)
    t = 1
    genie = GENIE3(data[t])
    cov_sign = np.sign(np.cov(data[t].T))
    genie = np.multiply(genie, cov_sign)
    auroc_dir_genie, aupr_dir_genie = directed_evaluation(A, genie, print_res=False)
    return auroc_dir_genie, aupr_dir_genie

def genie_rep(rep, net_num, sim_num, time_points):
    full_combination = list(product(np.arange(rep), np.arange(net_num)))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        res = p.starmap(functools.partial(genie_wrap, sim_num=sim_num, time_points=time_points), full_combination)
    return res

def dg_wrap(unused_index, network, sim_num, time_points):
    del unused_index
    A, data = generation(network, time_points, sim_num)
    dg_data = [data[:, i, :] for i in range(sim_num)]
    dg_time = [time_points for i in range(sim_num)]
    dg = dynGENIE3(dg_data, dg_time)
    cov_sign = np.sign(np.cov(data[len(time_points)-1].T))
    dg = np.multiply(dg, cov_sign)
    auroc_dir_dg, aupr_dir_dg = directed_evaluation(A, dg, print_res=False)
    return auroc_dir_dg, aupr_dir_dg

def dg_rep(rep, net_num, sim_num, time_points):
    full_combination = list(product(np.arange(rep), np.arange(net_num)))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        res = p.starmap(functools.partial(dg_wrap, sim_num=sim_num, time_points=time_points), full_combination)
    return res

def res_process(res, rep, net_num):
    res = np.reshape(res, (rep, net_num, 2))
    res = np.average(res, axis=0)
    auroc_10 = np.mean(res[0:10, 0]) * 0.5 + np.mean(res[20:30, 0]) * 0.5
    aupr_10 = np.mean(res[0:10, 1]) * 0.5 + np.mean(res[20:30, 1]) * 0.5
    auroc_20 = np.mean(res[10:20, 0]) * 0.5 + np.mean(res[30:40, 0]) * 0.5
    aupr_20 = np.mean(res[10:20, 1]) * 0.5 + np.mean(res[30:40, 1]) * 0.5
    return auroc_10, aupr_10, auroc_20, aupr_20
    

def main():
    
    sim_num = 100 # number of cells can be 10, 30, 100
    net_num = 40 # number of networks tested
    rep = 100 # each situation is averaged over 100 repeats 
    print('sim_num= ', sim_num)
    
    
    time_points = np.linspace(0.0, 0.6, 2)
    print('time_points: ', time_points)
    start_time = time.time()    
    res = wendy_rep(rep, net_num, sim_num, time_points)
    wendy = res_process(res, rep, net_num)
    print('wendy, auroc_10, aupr_10, auroc_20, aupr_20')
    print(wendy)
    end_time = time.time()
    print(end_time-start_time)
    
    for i in range(2):
        if i == 0:
            time_points = np.linspace(0.9, 3.0, 8)
        if i == 1:
            time_points = np.linspace(0.6, 3.0, 9)
        print('time_points: ', time_points)
        start_time = time.time()            
        res = sinc_rep(rep, net_num, sim_num, time_points)
        sinc = res_process(res, rep, net_num)
        print('sinc, auroc_10, aupr_10, auroc_20, aupr_20')
        print(sinc)
        end_time = time.time()
        print(end_time-start_time)
    
    for i in range(2):
        if i == 0:
            time_points = np.linspace(0.3, 3.0, 10)
        if i == 1:
            time_points = np.linspace(0.3, 2.4, 8)
        print('time_points: ', time_points)
        start_time = time.time()    
        res = nlode_rep(rep, net_num, sim_num, time_points)
        nlode = res_process(res, rep, net_num)
        print('nlode, auroc_10, aupr_10, auroc_20, aupr_20')
        print(nlode)
        end_time = time.time()
        print(end_time-start_time)
    
    start_time = time.time()
    time_points = np.linspace(0.0, 0.3, 2)
    print('time_points: ', time_points)
    res = genie_rep(rep, net_num, sim_num, time_points)
    genie = res_process(res, rep, net_num)
    print('genie3, auroc_10, aupr_10, auroc_20, aupr_20')
    print(genie)
    end_time = time.time()
    print(end_time-start_time)
    
    start_time = time.time()
    time_points = np.linspace(0.0, 2.7, 10)
    print('time_points: ', time_points)
    res = dg_rep(rep, net_num, sim_num, time_points)
    dg = res_process(res, rep, net_num)
    print('dyngenie3, auroc_10, aupr_10, auroc_20, aupr_20')
    print(dg)   
    end_time = time.time()
    print(end_time-start_time)
    
if __name__ == "__main__":
    main()

