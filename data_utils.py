import numpy as np
from sklearn.preprocessing import MinMaxScaler

def binarize_adj_mat(adj_mat):
    adj_mat = np.where(adj_mat < 1, 0., 1.)
    return adj_mat

def make_undirected(adj_mat):
    pos = np.where(adj_mat > 0)
    for (i, j) in zip(pos[0], pos[1]):
        val = (adj_mat[i, j] + adj_mat[j, i])/2
        adj_mat[i, j] = val
        adj_mat[j, i] = val
    return adj_mat

def get_all_label_distribution(truth):
    i = 0
    all_label_dist = np.zeros(np.shape(truth)[1])
    for q in truth.T :
        all_label_dist[i] = np.count_nonzero(q)
        i += 1
    #all_label_dist = all_label_dist/ sum(all_label_dist)
    return all_label_dist