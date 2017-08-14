import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import pairwise
from sklearn.decomposition.nmf import _initialize_nmf
from numpy.linalg import norm
import math
from scipy.linalg import sqrtm

def get_proximity_matrix(X,eta) :
    # Sparse proximity matrix
    #S = [i.toarray() + eta * pairwise.cosine_similarity(i.toarray()) for i in X]
    #S = [csr_matrix(i) for i in S]
    S = [(i + eta * pairwise.cosine_similarity(i))/2 for i in X]
    return S

def get_modularity_matrix(X):
    n = []
    #T = X[0].toarray()
    T = X[0]
    B = np.zeros((T.shape))
    for i in T :
        n.append(np.count_nonzero(i)) # Stores the degree of each node
    n = np.array(n)
    for a in range(T.shape[0]) :
        for b in range(T.shape[0]) :
            B[a][b]=n[a]*n[b]
    #B = B/(2*sum(n))
    B = B/(sum(n))
    return B

def initialize_factor_matrices(S, n, rank, z, k, init, dtype, logger):
    logger.debug('Initializing U')
    M = []
    U = np.zeros((n, rank))
    if init == 'random':
        U = np.array(np.random.rand(n, rank), dtype=dtype)
        for i in range(z):
            M.append(np.array(np.random.rand(n, rank), dtype=dtype))
    elif init == 'nndsvd':
        # combined_S = csr_matrix((n, n), dtype=dtype)
        combined_S = np.zeros((n, n), dtype=dtype)
        for i in S:
            combined_S = combined_S + i

        _, U = _initialize_nmf(combined_S, rank, 'nndsvd')
        if issparse(U):
            U = U.toarray()
        else:
            U = np.array(U)

        for i in range(z):
            P, _ = _initialize_nmf(S[i], rank, 'nndsvd')
            if issparse(P):
                P = P.toarray()
            else:
                P = np.array(P)
            M.append(P)
    else:
        raise 'Unknown init option ("%s")' % init
    H = np.array(np.random.rand(n, k), dtype=dtype)
    # H = csr_matrix(H)
    C = np.array(np.random.rand(k, rank), dtype=dtype)
    # C = csr_matrix(C)
    # U = csr_matrix(U)
    # M = [csr_matrix(i) for i in M]
    logger.debug('Initialization completed')
    return M, U.T, C, H

def __LS_updateM_L2(S, M, U, alpha, lmbda):
    """Update step for M with LS"""
    if len(S) == 0:
        return M
    eps = np.finfo(np.float).eps
    UtU = np.dot(U.T, U)
    for i in range(len(S)):
        M[i][M[i] < eps] = eps
        row_mean = np.nanmean(M[i],axis=1)
        #Find indices that you need to replace
        inds = np.where(np.isnan(M[i]))    
        #Place column means in the indices. Align the arrays using take
        M[i][inds]=np.take(row_mean,inds[0])
        
        M[i] = M[i] * ( alpha * np.dot(S[i],U) / (alpha * np.dot(M[i],UtU) + lmbda * M[i]))

        M[i][M[i] < eps] = eps
        row_mean = np.nanmean(M[i],axis=1)
        #Find indices that you need to replace
        inds = np.where(np.isnan(M[i]))    
        #Place column means in the indices. Align the arrays using take
        M[i][inds]=np.take(row_mean,inds[0])
    return M

def __LS_updateU_L2(S, M, U, H, C, alpha, beta, lmbda):
    """Update step for U with LS"""
    eps = np.finfo(np.float).eps
    U[U < eps] = eps
    row_mean = np.nanmean(U,axis=1)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(U))
    #Place column means in the indices. Align the arrays using take
    U[inds] = np.take(row_mean,inds[0])

    n, rank = U.shape
    nominator = np.zeros((U.shape), dtype=U.dtype)
    dnominator = np.zeros((U.shape), dtype=U.dtype)
    CtC = np.dot(C.T, C)
    for i in range(len(S)):
        MtM = np.dot(M[i].T, M[i])
        nominator += alpha * np.dot(S[i].T, M[i])
        dnominator += alpha * np.dot(U,MtM)
    nominator += beta * np.dot(H,C)
    dnominator += beta * np.dot(U,CtC)
    U = U * ((nominator) / (dnominator + lmbda * U))

    U[U < eps] = eps
    row_mean = np.nanmean(U,axis=1)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(U))    
    #Place column means in the indices. Align the arrays using take
    U[inds] = np.take(row_mean,inds[0])
    return U

def __LS_updateC_L2(H, U, C, beta, lmbda) :
    eps = np.finfo(np.float).eps
    UtU = np.dot(U.T, U)
    C[C < eps] = eps
    row_mean = np.nanmean(C,axis=1)
    #Find indices that you need to replace
    inds = np.where(np.isnan(C))    
    #Place column means in the indices. Align the arrays using take
    C[inds]=np.take(row_mean,inds[0])
        
    C = C * ( beta * np.dot(H.T , U) / (beta * np.dot(C,UtU) + lmbda * C))
    
    C[C < eps] = eps
    row_mean = np.nanmean(C,axis=1)
    #Find indices that you need to replace
    inds = np.where(np.isnan(C))    
    #Place column means in the indices. Align the arrays using take
    C[inds]=np.take(row_mean,inds[0])
    return C

def __LS_updateH_L2(H, U, C, X, B, beta, gamma, zeta) :
    eps = np.finfo(np.float).eps
    H[H < eps] = eps
    row_mean = np.nanmean(H,axis=1)
    #Find indices that you need to replace
    inds = np.where(np.isnan(H))    
    #Place column means in the indices. Align the arrays using take
    H[inds]=np.take(row_mean,inds[0])
    
    HtH = np.dot(H.T, H)
    HHtH = np.dot(H, HtH)
    BH = np.dot(B, H)
    small_delta = 2 * gamma * (BH * BH) + (16 * zeta * HHtH) * (2 * gamma * np.dot(X[0],H) + 2 * beta * np.dot(U,C.T) + (4 * zeta - 2 * beta)*H)
    small_delta_2 = (np.add((2 * gamma * BH),(8 * zeta * HHtH))) ** 2
    #H = H * np.sqrt( (-2 * gamma * BH + np.sqrt(small_delta)) / (8 * zeta * HHtH))
    H = H * np.sqrt( (-2 * gamma * BH + np.sqrt(small_delta_2)) / (8 * zeta * HHtH))

    H[H < eps] = eps
    row_mean = np.nanmean(H, axis=1)
    # Find indices that you need to replace
    inds = np.where(np.isnan(H))
    # Place column means in the indices. Align the arrays using take
    H[inds] = np.take(row_mean, inds[0])
    return H

def __LS_compute_fit(S, M, U, H, C, X, B, alpha, beta, gamma, zeta, lmbda):
    """Compute fit for factorization"""
    fitSMUt = 0
    fitHUCt = 0
    fitHtHI = 0

    normS = [norm(s.data) ** 2 for s in S]
    sumNorm = sum(normS)
    for i in range(len(M)):
        MUt = np.dot(M[i], U.T)
        fitSMUt += norm(S[i] - MUt) ** 2

    UCt = np.dot(U, C.T)
    normH = norm(H.data) ** 2
    fitHUCt += norm(H - UCt) ** 2

    HtH = np.dot(H.T,H)
    normHtH = norm(HtH.data) ** 2
    fitHtHI += norm(HtH - np.eye(H.shape[1])) ** 2

    x = - gamma * np.trace(np.dot(np.dot(H.T,X[0]),H))
    y = gamma * np.trace(np.dot(np.dot(H.T,B),H))

    normM = [norm(m.data) ** 2 for m in M]
    l2_reg = norm(U) ** 2 + norm(C) ** 2 + normM

    #print 'fitSMUt', (fitSMUt)
    #print 'sumNorm', (sumNorm)

    return (alpha * ( fitSMUt / sumNorm ) + beta * ( fitHUCt / normH ) + zeta * (fitHtHI / normHtH) + x + y + lmbda * l2_reg)

def factorize(config, D, X, Y, Y_train, train_ids, val_ids, logger):
    # ---------- Get the parameter values-----------------------------------------
    alpha = float(config.ALPHA)
    beta = float(config.BETA)
    gamma = float(config.GAMMA)
    theta = float(config.THETA)
    lmbda = float(config.LAMBDA)
    eta = float(config.ETA)
    zeta = float(config.ZETA)
    phi = float(config.PHI)
    init = config.INIT
    maxIter = int(config.MAX_ITER)
    rank = int(config.L_COMPONENTS)
    costF = config.COST_F
    if costF == 'LS':
        conv = float(config.CONV_LS)
    elif costF == 'KL':
        conv = float(config.CONV_KL)
    else:
        conv = float(config.CONV_MUL)
    
    n = np.shape(X[0])[0] 
    m = np.shape(D[0])[0]
    q, _ = Y.shape
    dtype = np.float32 
    z = len(X)
    k = int(config.K)
    # ---------- Construct proximity matrix-----------------------------------------
    S = get_proximity_matrix(X,eta)
    #S_log = "Results/S.log" # S
    B = get_modularity_matrix(X)
    #np.savetxt(s_log, S)
    # Creation of class prior vector from training data
    i = 0
    train_label_dist = np.zeros(q)
    for j in Y_train:  # (q x n')
        train_label_dist[i] = np.count_nonzero(j)
        i += 1
    train_label_dist = train_label_dist / sum(train_label_dist)
    # Creation of penalty matrix from label matrix and training data
    W = np.copy(Y.T)
    unlabelled_ids = np.logical_not(train_ids)
    n_unlabelled = np.count_nonzero(unlabelled_ids)
    W[unlabelled_ids, :] = np.zeros((n_unlabelled, q))
    # tmp = W[train_ids, :]
    # tmp[tmp == 1] = 0.01
    # tmp[tmp == 0] = 1
    W[train_ids, :] = np.ones((n - n_unlabelled, q))
    W = W.T
    mod_Y = W * Y
    # ---------- Initialize factor matrices-----------------------------------------
    M, U, C, H = initialize_factor_matrices(S, n, rank, z, k, init, dtype, logger)
    #initial_u_log = "Results/initial_u.log" # B
    #np.savetxt(initial_u_log, U)
    
    #  ------ compute factorization -------------------------------------------
    fit = fitchange = fitold = f = prev_fitchange = 0
    exectimes = []
    best_val_fit_c = np.inf
    best_train_fit_c = np.inf
    best_result_c = {'U': U, 'H': H}

    for iter in range(maxIter):
        tic = time.time()

        fitold = fit
        if costF == 'LS':
            M = __LS_updateM_L2(S, M, U, alpha, lmbda)
            #print 'M : ', np.count_nonzero(M)
            U = __LS_updateU_L2(S, M, U, H, C, alpha, beta, lmbda)
            #print 'U : ', np.count_nonzero(U)
            C = __LS_updateC_L2(H, U, C, beta, lmbda)
            # compute fit value
            H = __LS_updateH_L2(H, U, C, X, B, beta, gamma, zeta)
            fit = __LS_compute_fit(S, M, U, H, C, X, B, alpha, beta, gamma, zeta, lmbda)

        if (iter % 5 == 0):
            from main_algo import tune_model, tune_model_using_classifier
            logger.debug("\n***************************************************")
            logger.debug("Train Fit at step {%3d}: {%0.5f }" % (iter, fit))
            performance_train_c = tune_model_using_classifier(config, U, Y.T, train_ids, train_ids)
            logger.debug("Train accuracy: {%0.5f } , Train Loss: {%0.5f }" % ( performance_train_c['accuracy'], performance_train_c['cross_entropy'] ))
            performance_val_c = tune_model_using_classifier(config, U, Y.T, train_ids, val_ids)
            logger.debug("Validation accuracy: {%0.5f } , Validation Loss: {%0.5f }" % (
                performance_val_c['accuracy'], performance_val_c['cross_entropy']))
            if performance_val_c['cross_entropy'] <= best_val_fit_c:
                if performance_val_c['cross_entropy'] < best_val_fit_c:
                    best_val_fit_c = performance_val_c['cross_entropy']
                    best_result_c['U'] = U
                    best_result_c['H'] = H
                elif performance_val_c['cross_entropy'] <= best_val_fit_c and performance_train_c['cross_entropy'] < best_train_fit_c:
                    #print("Changed in first")
                    best_val_fit_c = performance_val_c['cross_entropy']
                    best_train_fit_c = performance_train_c['cross_entropy']
                    best_result_c['U'] = U
                    best_result_c['H'] = H
            logger.debug("***************************************************\n")

        fitchange = abs(fitold - fit)
        toc = time.time()
        exectimes.append(toc - tic)

        logger.debug('M-NMF ::: [%3d] fit: %0.5f | delta: %7.1e | secs: %.5f'
                   % (iter, fit, fitchange, exectimes[-1]))
        if iter > maxIter or fitchange < conv:
            # best_result['Q'] = Q
            # best_result['U'] = U
            # best_result['H'] = H
            # best_result_c['Q'] = Q
            # best_result_c['U'] = U
            # best_result_c['H'] = H
            if iter > maxIter or fitchange < conv:
                from main_algo import tune_model, tune_model_using_classifier
                performance_train_c = tune_model_using_classifier(config, best_result_c['U'], Y.T, train_ids, train_ids)
                performance_val_c = tune_model_using_classifier(config, best_result_c['U'], Y.T, train_ids, val_ids)
                performance_train_c_1 = tune_model_using_classifier(config, U, Y.T, train_ids, train_ids)
                performance_val_c_1 = tune_model_using_classifier(config, U, Y.T, train_ids, val_ids)
                if performance_val_c_1['cross_entropy'] <= performance_val_c['cross_entropy']:
                    if performance_val_c_1['cross_entropy'] < performance_val_c['cross_entropy']:
                        best_result_c['U'] = U
                        best_result_c['H'] = H
                    elif performance_val_c_1['cross_entropy'] <= performance_val_c['cross_entropy'] and \
                                    performance_train_c_1[
                                        'cross_entropy'] < performance_train_c['cross_entropy']:
                        best_result_c['U'] = U
                        best_result_c['H'] = H
                logger.debug("***************************************************\n")
            break
    
    #return U, Q
    return best_result_c
