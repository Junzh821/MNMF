import numpy as np
from sklearn import preprocessing
from sklearn.metrics import pairwise
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers
import data_utils as du
from sklearn.preprocessing import normalize

def RWR_all(tadj_mat, alpha=0.1, iters=10):
    adj_mat = np.copy(tadj_mat)
    # adj_mat = du.binarize_adj_mat(np.copy(tadj_mat))
    result = np.zeros(np.shape(adj_mat))
    trans_prob = normalize(adj_mat, norm='l2', axis=1)
    for i in range(iters) :
        result = np.add(result, alpha * np.power((1 - alpha), i) * k_step_graph(trans_prob, steps=i, binarize = False, normalize = False))
    result = preprocessing.normalize(result, norm='l2', axis=1)
    return result

def ERWR_all(tadj_mat, alpha=0.1, iters=10):
    adj_mat = np.copy(tadj_mat)
    # adj_mat = du.binarize_adj_mat(np.copy(tadj_mat))
    adj_mat = k_step_graph(adj_mat, steps=2, binarize = False, normalize = False) # because we normalize it in RWR_all
    result = RWR_all(adj_mat, alpha, iters=iters)
    return result

def k_step_graph(tadj_mat, steps=2, binarize = True, normalize = True):
    adj_mat = np.copy(tadj_mat)
    if binarize :
        adj_mat = du.binarize_adj_mat(np.copy(tadj_mat))
    result = adj_mat
    steps -= 1
    for k in range(steps):
        result = np.dot(result, adj_mat)
    if normalize :
        result = preprocessing.normalize(result, norm='l2', axis=1)
    return result

def label_sim_graph(n_labelled, tadj_mat, labels, train_ids):
    G = nx.from_numpy_matrix(tadj_mat)
    adj_mat = np.eye(tadj_mat.shape[0])
    for i in range(n_labelled):
        for j in range(n_labelled):
            if i != j :
                if nx.has_path(G, i, j):
                    if np.intersect1d(np.where(labels[i, :])[0], np.where(labels[j, :])[0]).shape[0] > 0:
                        adj_mat[i, j] = 1
    return adj_mat

def attr_sim_graph(view):
    adj_mat = pairwise.cosine_similarity(view)
    adj_mat = normalize(adj_mat, axis=1, norm='l2')
    return adj_mat

def prediction_sim_graph(n_ids, attributes, labels, train_ids):
    pred_ids = np.logical_not(train_ids)
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    attributes = normalize(attributes, axis=1, norm='l2')
    clf.fit(attributes[train_ids, :], labels)
    predictions = clf.decision_function(attributes)
    adj_mat = pairwise.cosine_similarity(predictions)
    adj_mat = normalize(adj_mat, axis=1, norm='l2')
    return adj_mat

def integrate_latent_graphs(n_ids, latent_graphs, labels, train_ids, w_lambda=0.1):
    m = len(latent_graphs)
    u = np.zeros((m, 1))
    I = np.eye(m, m)
    K = np.zeros((m, m))
    YY = np.matmul(labels, labels.T)
    for i in range(m):
        u[i] = np.trace(np.matmul(YY, latent_graphs[i][train_ids, :]))
        for j in range(m):
            K[i, j] = np.trace(np.matmul(latent_graphs[i][train_ids, :], latent_graphs[j][train_ids, :].T))

    P = matrix(2 * (K + w_lambda*I), tc='d')
    q = matrix(-2 * u, tc='d')
    G = matrix(-1 * np.eye(m, m), tc='d')
    h = matrix(np.zeros((m, 1)), tc='d')
    A = matrix(np.ones((1, m)), tc='d')
    b = matrix(1.0)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.round(np.array(sol['x']).reshape(m), 2)

    adj_mat = np.zeros((n_ids, n_ids), dtype=float)
    for i in range(m):
        adj_mat = np.add(adj_mat, w[i]*latent_graphs[i])

    return adj_mat, w