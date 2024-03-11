"""
for three experimental data sets, use WENDY method to calculate the GRN
between each neighboring pair of time points.
then draw the GRN

notice that using the same code, the final plot might differ in different
computers
"""

import numpy as np
from wendy_alg import wendy_alg
from graphviz import Digraph

for dataset in range(3):
    add = 'Experimental data/Exp data/Data %d/' % dataset
    f = open(add+"expdata.txt", "r")
    data = f.read()
    data = data.split()
    data = np.array(data)
    data = np.reshape(data, (100, -1))
    data = data.T
    data = np.double(data) # 456/373/758 cells (rows) and 100 genes (columns)
    n_total_cell = data.shape[0]
    
    
    tt = np.mean(data>0, axis=0)
    nd = []
    major_gene = []
    for i in range(100):
        if tt[i] > 0.95:
            major_gene.append(i)
            nd.append(data[:, i])
    nd = np.array(nd).T
    gene_num = len(major_gene) # find genes we should study: express in more than
                                # 95% cells
    
    
    f = open(add+"time.txt", "r")
    time = f.read()
    time = time.split()
    time = np.array(time)
    time = np.reshape(time, (n_total_cell, 2))
    time = time[:, 0]
    time = np.int_(time) # the time each cell is measured
    
    exp = []
    tp = np.unique(time)
    n_tp = len(tp)
    for i in range(n_tp):
        left = float('inf')
        right = -float('inf')
        for j in range(len(time)):
            if time[j] == tp[i]:
                left = min(left, j)
                right = max(right, j)
        exp.append(nd[left:right+1, :]) # expression data at each time point
    
    
    
    f = open(add+"tf.txt", "r")
    tf = f.read()
    tf = tf.split()
    tf = [tf[i].upper() for i in major_gene] # name of studied genes
    
    wendy = []
    acc = np.zeros((gene_num, gene_num))
    for i in range(n_tp-1):
        temp = wendy_alg(exp[i], exp[i+1]) # use WENDY to calculate the GRN
        wendy.append(temp)
        reg = []
        acc += np.abs(temp)
    total = []
    for p in range(gene_num):
        for q in range(gene_num):
            if p == q:
                continue
            total.append([acc[p, q], p, q])
    total.sort(reverse=True, key=lambda x:x[0])
    strong = set()
    for w in total:
        if (w[2], w[1]) not in strong: # top 15 edges
            strong.add((w[1], w[2]))
        if len(strong) == 15:
            break
    gene_names = set()
    for [x,y] in strong:
        gene_names.add(tf[x])
        gene_names.add(tf[y])
    gene_names = list(gene_names)
    gs = ', '.join(gene_names)
    print(gs) # genes considered
    
    for i in range(n_tp-1): # use graphviz to draw the GRN
        dot = Digraph()
        for gn in gene_names:
            dot.node(gn, gn)
        for [x, y] in strong:
            if wendy[i][x, y] > 0.01:
                dot.edge(tf[x], tf[y], penwidth=str(10*np.abs(wendy[i][x, y])))
            elif wendy[i][x, y] < -0.01:
                dot.edge(tf[x], tf[y], arrowhead='tee', 
                         penwidth=str(10*np.abs(wendy[i][x, y])))    
            else:
                dot.edge(tf[x], tf[y], penwidth=str(0), arrowsize=str(0))
        filename = 'Experimental data/Exp GRN/GRN_data%d_time%d' % (dataset, i)
        file_ext = 'png'
        dot.render(filename, format=file_ext, view=False) # save the GRN figure
