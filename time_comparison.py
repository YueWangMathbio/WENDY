"""
this code compares the time cost of different GRN inference methods
"""
from data_generation import generation
from sincerities import sincer
from wendy_alg import wendy_alg
import numpy as np
from xgbgrn import get_importances
import time
from GENIE3 import GENIE3
from dynGENIE3 import dynGENIE3

time_points = np.linspace(0.00, 3.00, 11) # generate data at these time points
sim_num = 100 # number of cells/simulation trajectories
n10 = list(range(10)) + list(range(20,30))
n20 = list(range(10,20)) + list(range(30,40))
nn = [n10, n20]
cn = [10, 30, 100] # number of cells
all_mean = []
all_std = []
for cc in range(2):
    net = nn[cc]
    for sim_num in cn:
        time_used = [[] for i in range(5)]
        for network in net:
        
            A, data = generation(network, time_points, sim_num)
            # A is true GRN, data is expression data
            
            
            # wendy method
            start_time = time.time()
            t0 = 0
            t1 = 1
            covdyn = wendy_alg(data[t0], data[t1])
            end_time = time.time()
            time_used[0].append(end_time-start_time)
            
            # sincerities method
            start_time = time.time()
            sinc_res = sincer(data, time_points)
            end_time = time.time()
            time_used[1].append(end_time-start_time)
            
            # nonlinearodes method
            
            start_time = time.time()               
            bulk_data = [np.average(data[:, :sim_num//2, :], axis=1), np.average(data[:, sim_num//2:, :], axis=1)]
            nl_time = [np.arange(len(time_points))] * 2
            nlode = get_importances(bulk_data, nl_time, alpha='from_data')                
            end_time = time.time()
            time_used[2].append(end_time-start_time)
            
            # genie3 method
            start_time = time.time()
            t = 1
            genie = GENIE3(data[t])
            end_time = time.time()
            time_used[3].append(end_time-start_time)
            
            # dygenie3 method
            start_time = time.time()
            dg_data = [data[:, z, :] for z in range(sim_num)]
            dg_time = [time_points for z in range(sim_num)]
            dg3 = dynGENIE3(dg_data,dg_time)
            end_time = time.time()
            time_used[4].append(end_time-start_time)
                
                    
        mean = np.round([np.mean(time_used[i]) for i in range(5)], decimals=2)
        std = np.round([np.std(time_used[i]) for i in range(5)], decimals=2)
        all_mean.append(mean)
        all_std.append(std)
        
print(all_mean) # mean of time cost
print(all_std) # standard deviation of time cost