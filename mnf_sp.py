import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import pairwise
from sklearn.decomposition.nmf import _initialize_nmf
from numpy.linalg import norm
import math
from scipy.linalg import sqrtm

def initialize_factor_matrices(S, Y, n, rank, k, init, dtype, logger) :
    # np.random.seed(0)
    logger.debug('Initializing U')
    if init == 'random':
        U = np.array(np.random.rand(n, rank), dtype=dtype)
        M = np.array(np.random.rand(n, rank), dtype=dtype)
    elif init == 'nndsvd':
        M, U = _initialize_nmf(S, rank, 'nndsvd')
        if issparse(U) and issparse(M):
            U = U.toarray()
            M = M.toarray()
        else:
            U = np.array(U)
            M = np.array(M)
    else:
        raise 'Unknown init option ("%s")' % init
    q = np.shape(Y)[0]
    Q = np.array(np.random.rand(q, rank), dtype=dtype)
    H = np.array(np.random.rand(n, k), dtype=dtype)
    C = np.array(np.random.rand(k, rank), dtype=dtype)
    logger.debug('Initialization completed')
    return M, U.T, C, H, Q

def __LS_updateM_L2(S, M, U, alpha, lmbda):
    eps = np.finfo(np.float).eps
    UtU = np.dot(U.T, U)

    numerator = alpha * 2 * S * U
    denominator = alpha * 2 * np.dot(M, UtU) + lmbda * 2 * M
    denominator[denominator == 0] = eps

    M = M * (numerator / denominator)

    M[M < eps] = eps
    row_mean = np.nanmean(M, axis=1)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(row_mean, inds[0])
    return M

def __LS_updateU_L2(S, M, U, H, C, W, Y, Q, A, d, alpha, beta, theta, lmbda, phi):
    eps = np.finfo(np.float).eps

    nominator = np.zeros((U.shape), dtype=U.dtype)
    dnominator = np.zeros((U.shape), dtype=U.dtype)
    MtM = np.dot(M.T, M)
    CtC = np.dot(C.T, C)
    nominator += alpha * 2 * S.T * M
    dnominator += alpha * 2 * np.dot(U, MtM) + lmbda * 2 * U
    nominator += beta * 2 * H * C
    dnominator += beta * 2 * np.dot(U, CtC)
    dnominator[dnominator == 0] = eps
    U = U * (nominator / dnominator)

    U[U < eps] = eps
    row_mean = np.nanmean(U, axis=1)
    inds = np.where(np.isnan(U))
    U[inds] = np.take(row_mean, inds[0])
    return U

def __LS_updateC_L2(H, U, C, beta, lmbda) :
    eps = np.finfo(np.float).eps
    UtU = np.dot(U.T, U)

    numerator = beta * 2 * H.T * U
    denominator = beta * 2 * np.dot(C, UtU) + lmbda * 2 * C
    denominator[denominator == 0] = eps

    C = C * (numerator / denominator)

    C[C < eps] = eps
    row_mean = np.nanmean(C, axis=1)
    inds = np.where(np.isnan(C))
    C[inds] = np.take(row_mean, inds[0])
    return C

def __LS_updateH_L2(H, U, C, X, B, beta, gamma, zeta, lmbda) :
    eps = np.finfo(np.float).eps

    HtH = H.T * H
    HHtH = H * HtH
    BH = B * H
    small_delta_1 = 2 * gamma * BH.multiply(BH) + (16 * zeta * HHtH).multiply(2 * gamma * X * H + csr_matrix(2 * beta * np.dot(U,C.T)) + (4 * zeta - 2 * beta)* H)
    small_delta_2 = ((2 * gamma * BH) + (8 * zeta * HHtH)).power(2)
    H = H.multiply(np.sqrt((-2 * gamma * BH + np.sqrt(small_delta_1)) / (8 * zeta * HHtH + lmbda * 2 * H)))
    # H = H.multiply(np.sqrt( (-2 * gamma * BH + np.sqrt(small_delta_2)) / (8 * zeta * HHtH + lmbda * 2 * H)))

    H = H.toarray()
    H[H < eps] = eps
    row_mean = np.nanmean(H, axis=1)
    inds = np.where(np.isnan(H))
    H[inds] = np.take(row_mean, inds[0])
    return csr_matrix(H)

def __LS_updateQ_L2(Y, U, W, Q, theta, lmbda):
    eps = np.finfo(np.float).eps
    mod_Y = Y.multiply(W)
    QUt = np.dot(Q, U.T)

    numerator = theta * 2 * mod_Y * U
    denominator = theta * 2 * np.dot((W * QUt), U) + lmbda * 2 * Q
    denominator[denominator == 0] = eps

    Q = Q * (numerator / denominator)

    Q[Q < eps] = eps
    row_mean = np.nanmean(Q, axis=1)
    inds = np.where(np.isnan(Q))
    Q[inds] = np.take(row_mean, inds[0])
    return Q

def __LS_compute_fit(S, M, U, H, C, X, B, W, Y, Q, A, d, alpha, beta, gamma, zeta, theta, lmbda, phi):
    fitSMUt = fitHUCt = fitHtHI = fitWYQUt = 0
    QUt = np.dot(Q, U.T)
    UQt = np.dot(U, Q.T)

    normS = norm(S.data) ** 2
    MUt = np.dot(M, U.T)
    fitSMUt += norm(S - MUt) ** 2

    UCt = np.dot(U, C.T)
    normH = norm(H.data) ** 2
    fitHUCt += norm(H - UCt) ** 2

    HtH = H.T * H
    normHtH = norm(HtH.data) ** 2
    fitHtHI += norm(HtH - np.eye(H.shape[1])) ** 2

    z = - gamma * np.trace(((H.T * X) * H).A) + gamma * np.trace(((H.T * B) * H).A)
    l2_reg = norm(U) ** 2 + norm(C) ** 2 + norm(M) ** 2 + norm(H.data) ** 2

    return (alpha * (fitSMUt / normS) + beta * (fitHUCt / normH) + zeta * (fitHtHI / normHtH) + z + lmbda * l2_reg)


def factorize(config, dataset, S, B, D, X, Y, Y_train, train_ids, val_ids, test_ids, logger):
    # ---------- Get the parameter values-----------------------------------------
    eta = float(config.ETA)
    alpha = float(config.ALPHA)
    beta = float(config.BETA)
    theta = float(config.THETA)
    gamma = float(config.GAMMA)
    phi = float(config.PHI)
    delta = float(config.DELTA)
    lmbda = float(config.LAMBDA)
    zeta = float(config.ZETA)
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

    n = np.shape(X)[0]
    m = np.shape(D)[0]
    q, _ = Y.shape
    dtype = np.float32
    k = int(config.K)

    # Creation of class prior vector from training data
    train_label_dist = np.sum(Y_train.T, axis=0) / np.sum(Y_train)

    # Creation of penalty matrix from label matrix and training data
    W = np.copy(Y.T.toarray())
    unlabelled_ids = np.logical_not(train_ids)
    n_unlabelled = np.count_nonzero(unlabelled_ids)
    W[unlabelled_ids, :] = np.zeros((n_unlabelled, q))
    W[train_ids, :] = np.ones((n - n_unlabelled, q))
    W = W.T

    # ---------- Initialize factor matrices-----------------------------------------
    M, U, C, H, Q = initialize_factor_matrices(S, Y, n, rank, k, init, dtype, logger)
    H = csr_matrix(H)

    #  ------ compute factorization -------------------------------------------
    fit = fitchange = fitold = f = prev_fitchange = 0
    exectimes = []
    best_result = {'Q': Q, 'U': U, 'H': H, 'i': 0}
    for iter in range(maxIter):
        tic = time.time()
        fitold = fit
        if costF == 'LS':
            A = np.dot(H.toarray(), H.T.toarray())
            d = np.diag(np.sum(A, axis=1))
            M = __LS_updateM_L2(S, M, U, alpha, lmbda)
            if beta != 0:
                C = __LS_updateC_L2(H, U, C, beta, lmbda)
                H = __LS_updateH_L2(H, U, C, X, B, beta, gamma, zeta, lmbda)
            U = __LS_updateU_L2(S, M, U, H, C, W, Y, Q, A, d, alpha, beta, theta, lmbda, phi)
            fit = __LS_compute_fit(S, M, U, H, C, X, B, W, Y, Q, A, d, alpha, beta, gamma, zeta, theta, lmbda, phi)

        if fitold != 0.0:
            fitchange = abs(fitold - fit)/abs(fitold)
        else :
            fitchange = abs(fitold - fit)

        toc = time.time()
        exectimes.append(toc - tic)
        logger.debug('MNF  ::: [%3d] fit: %0.5f | delta: %7.1e | secs: %.5f'
                   % (iter, fit, fitchange, exectimes[-1]))

        if iter > maxIter or fitchange < conv:
            break

    best_result = {'Q': Q, 'U': U, 'H': H, 'i': iter}
    return best_result, best_result, best_result