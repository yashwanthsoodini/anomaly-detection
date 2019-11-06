import numpy as np
import random
from numpy.linalg import solve
import matplotlib.pyplot as plt
import time
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import os, os.path
import sys

def get_adjacency_matrix_from_txt(text_file_path):
    f = open(text_file_path, 'r')
    n = int(next(f).split(" ")[0])
    A = lil_matrix((n,n),dtype=float)
    for i in f.readlines():
        nodes = i.split(" ")
        n1 = int(nodes[0])
        n2 = int(nodes[1])
        A[n1,n2] = 1
        A[n2,n1] = 1
    f.close()
    return A

def get_num_nodes_from_txt(text_file_path):
    f = open(text_file_path, 'r')
    n = int(next(f).split(" ")[0])
    f.close()
    return n

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def get_s(groups,g,n):
    s = np.zeros((n,g))
    for k in range(0,g):
        for i in groups[k]:
            s[i][k] = 1
    return s

def moving_range_mean(x):
    n = len(x)
    M = []
    for i in range(1,n):
        M.append(abs(x[i]-x[i-1]))
    return np.sum(M)/(n-1)

def get_sparse_diag(diagonal):
    n = len(diagonal)
    D = lil_matrix((n,n),dtype=float)
    for i in range(0,n):
        D[i,i] = diagonal[i]
    return D

dataset_name = str(sys.argv[1])
dataset_path = "datasets/" + dataset_name+"/"
dataset_file_suffix = "_"+dataset_name+".txt"
num_graphs_by_dataset = {'voices':38, 'autonomous':733, 'enron_by_day':749, 'p2p-Gnutella':8}
num_graphs = num_graphs_by_dataset[dataset_name]
sim_timeline = []

for graph_num in range(0,num_graphs-1):
    start = time.time()
    A1 = get_adjacency_matrix_from_txt(dataset_path+str(graph_num)+dataset_file_suffix)
    A2 = get_adjacency_matrix_from_txt(dataset_path+str(graph_num+1)+dataset_file_suffix)
#     D1 = get_sparse_diag(np.array(A1.sum(0)).flatten())
#     D2 = get_sparse_diag(np.array(A2.sum(0)).flatten())
    D1 = diags(np.array(A1.sum(0)).flatten(),dtype=float)
    D2 = diags(np.array(A2.sum(0)).flatten(),dtype=float)
    n = get_num_nodes_from_txt(dataset_path+str(graph_num)+dataset_file_suffix)
    I = eye(n)
    e = 0.9
    nodes = list(range(0,n))
    num_groups = max(int(n/100),10)
    groups = partition(nodes,num_groups)
    s = np.array(get_s(groups,num_groups,n))
    I = csr_matrix(I)
    D1 = csr_matrix(D1)
    D2 = csr_matrix(D2)
    A1 = csr_matrix(A1)
    A2 = csr_matrix(A2)
#     s = csc_matrix(s)
    S1 = spsolve((I+(D1*e**2)-(A1*e)),s)
    S2 = spsolve((I+(D2*e**2)-(A2*e)),s)
    d = np.sum(np.square(np.sqrt(S1)-np.sqrt(S2)))
    sim = 1/(1+d)
    sim_timeline.append(sim)

np.savetxt(dataset_name+'_time_series.txt',sim_timeline)
median = np.median(sim_timeline)
mean = moving_range_mean(sim_timeline)
upper_threshold = median + 3*mean
lower_threshold = median - 3*mean

fig = plt.figure()
plt.axhline(y=upper_threshold,linestyle='--',color='brown')
plt.axhline(y=lower_threshold,linestyle='--',color='black')
plt.style.use('default')
plt.plot(sim_timeline,'.')
fig.savefig(dataset_name+'_time_series.png')
