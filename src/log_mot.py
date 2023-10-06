import numpy as np
from scipy.special import logsumexp

def ipot(C, mu, epi, tau, iter_in, iter_out):
    """ Inexact proximal point iteartion for multi-marginal optimal transport in the log domain
    
    Args:
        C (np.array): cost matrix, shape = [n1, ..., nK]
        mu ([np.array]): list of marginal distributions each with shape [ni, 1]
        epi (float): convergence threshold
        tau (float): soft update ratio
        iter_in (int): maximum number of inner iterations
        iter_out (int): maximum number of outer iterations

    Returns:
        t (np.array): transport plan matrix , shape = [n1, ..., nK]
    """
    K = len(mu)  # number of distributions
    N = [] # number of supports in each distribution
    u = [] # scale vectors
    S = (np.ones((K,K)) - 2 * np.eye(K)).astype(np.int32).tolist()
    log_T = np.zeros(N)        
    for i in range(K):
        mu[i] = np.log(mu[i]) # log of marginal distributions
        log_T = log_T + np.reshape(mu[i],S[i])    # initial transport plan
        N.append(len(mu[i]))
        u.append(np.log(np.ones(N[i])/N[i]))     # scale vectors with shape = [ni,1], log domain
    t = np.exp(log_T)
    for i in range(iter_out):
        Q = - C / epi + log_T
        for j in range(iter_in):
            P = []  # marginal distributions
            U = np.zeros(N)
            for k in range(K): # compute marginal distribution using logsumexp
                ax = list(range(k)) + list(range(k+1,K))
                P.append(logsumexp(log_T, axis = tuple(ax)))
                u[k] = tau * u[k] + (1 - tau) * (u[k] + mu[k] - P[k])   # update u
                U = U + np.reshape(u[k],S[k])   # compute U as the Kronecker product of uj (log domain)
            log_T = Q + U
    t = np.exp(log_T)
    return t

def multi_FGW(C, A, mu, lp=0.1, alpha = 0.5, tau = 0.5, iter_in = 5, iter_out = 50, eps = 1e-5):
    """ Regularized inexact proximal point iteartion for multi-marginal optimal transport in the log domain
    
    Args:
        C (np.array): cost matrix, shape = [n1, ..., nK]
        A ([np.array]): a list of adjacency matrices, each with shape = [ni, ni]
        mu ([np.array]): list of marginal distributions each with shape [ni, 1]
        lp (float): weight for priximal regularizer
        alpha (float): weight for GW term, (1-alpha) * WD + alpha * GW
        tau (float): soft update ratio
        iter_in (int): maximum number of inner iterations
        iter_out (int): maximum number of outer iterations
        eps: error threshold fot termination

    Returns:
        t (np.array): transport plan matrix , shape = [n1, ..., nK]
    """
    K = len(mu)  # number of distributions
    N = [] # number of supports in each distribution
    u = [] # scale vectors
    S = (np.ones((K,K)) - 2 * np.eye(K)).astype(np.int32).tolist()  # shape matrix
    W = A.copy()
    for k in range(K): # row-normalized adjacency
        W[k] = A[k] + np.eye(len(A[k]))
        W[k] = A[k] / (np.sum(A[k], axis = 1).reshape(-1,1) + 1e-5)
    log_T = np.zeros(N)
    
    for k in range(K):
        mu[k] = np.log(mu[k]) # log of marginal distributions
        log_T = log_T + np.reshape(mu[k],S[k])    # initial transport plan
        N.append(len(mu[k]))
        u.append(np.log(np.ones(N[k])/N[k]))     # scale vectors with shape = [ni,1], log domain
    
    res = np.inf
    i = 0
    t = np.exp(np.clip(log_T, a_min = -np.inf, a_max = 70))
    W_list = [np.inf]
    res_list = []
    while i < iter_out and res > eps:
        ## compute multi-margianl GW
        t_prev = t
        P_i = compute_P_i(log_T)
        P_ij = compute_P_ij(log_T)
        L = compute_L(A, P_i, P_ij)

        Q = - ((1-alpha) * C + alpha * L - lp * log_T) / lp
        W_list.append(np.sum(((1-alpha) * C + alpha * L) * t))
        for j in range(iter_in):
            P_i = compute_P_i(log_T)
            U = np.zeros(N)
            for k in range(K): # compute marginal distribution using logsumexp
                u[k] = u[k] + mu[k] - P_i['{}'.format(k)]   # update u
                U = U + np.reshape(u[k],S[k])   # compute U as the Kronecker product of uj (log domain)
            log_T = (1 - tau) * log_T + tau * (Q + U)
        
        t =np.exp(np.clip(log_T, a_min = -np.inf, a_max = 70))
        res = np.abs(W_list[-1] - W_list[-2])
        res_list.append(res)
        i = i + 1
    return t

def compute_P_i(log_T):  # compute single marginal
    K = len(log_T.shape)
    P_i = {}
    for i in range(K):
        tmp_s = list(range(K))
        tmp_s.remove(i)
        P_i['{}'.format(i)] = logsumexp(log_T, axis = tuple(tmp_s))
    return P_i

def compute_P_ij(log_T): # compute pair marginal
    K = len(log_T.shape)
    P_ij = {}   # pair marginal
    for i in range(K):
        tmp_s = list(range(K))
        tmp_s.remove(i)
        for j in range(i+1,K):
            tmp_ss = tmp_s.copy()
            tmp_ss.remove(j)
            P_ij['{},{}'.format(i,j)] = logsumexp(log_T, axis = tuple(tmp_ss))
    return P_ij

def compute_L(A, P_i, P_ij):
    K = len(A)
    s = []
    for i in range(K):
        s.append(len(A[i]))
    L = np.zeros(s)
    for i in range(K):
        tmp_s = np.ones(K, dtype = np.int16)
        tmp_s[i] = s[i]
        L += (K-1) * np.dot(A[i]**2, np.exp(np.clip(P_i['{}'.format(i)], a_min = -np.inf, a_max = 70))).reshape(tmp_s)
        for j in range(i+1, K):
            tmp_ss = tmp_s.copy()
            tmp_ss[j] = s[j]
            L += -2 * np.dot(np.dot(A[i], np.exp(np.clip(P_ij['{},{}'.format(i,j)], a_min = -np.inf, a_max = 70))), A[j].T).reshape(tmp_ss)
    return L

def compute_N(log_T, A):
    K = len(log_T.shape)
    log_neg_T = np.exp(np.clip(log_T, a_min = -np.inf, a_max = 70))
    for k in range(K):
        log_neg_T = np.tensordot(log_neg_T, A[k], axes = (0,1))
    log_neg_T = np.log(np.clip(log_neg_T, a_min = 1e-10, a_max = np.inf))
    return log_neg_T