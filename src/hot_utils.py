import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import ot
import time

def get_rwr(A, H, beta = 0.15, iter_rwr = 50, eps = 1e-5):
    """ unified random walk with restart on separated graphs

    Args:
        A ([np.array]]): a list of K adjacency matrices each with shape [ni, ni]
        H (np.array): anchor node sets, shape = [m, K]
        beta (float): rwr restart probability
        iter_rwr (int): maximum number of rwr interation
        eps: error threshold

    Returns:
        r (np.array): list of rwr score each with shape [ni, m]
    """
    m, K = H.shape
    r = []
    for i in range(K):
        n = A[i].shape[0]
        T = A[i] / np.sum(A[i],1)
        e = np.zeros((n,m))
        e[H[:,i],range(m)] = 1
        s = np.zeros((n,m))
        j = 0
        res = np.inf
        while j < iter_rwr and res > eps:
            s_old = s
            s = (1 - beta) * np.dot(T,s) + beta * e
            res = np.max(abs(s - s_old))
            j = j + 1
        s = np.array(s)
        r.append(s)
    return r


def get_hits_multi(T, KList, test):
    """ Get Hits@K metrics

    Args:
        T (np.array): soft alignment score, shape = [n1, ..., nK]
        KList (list): list of Ks
        test (list): test set, shape=[n, K]

    Returns:
        hit_pair (np.array): pair-wise Hits@K
        hit_high (np.array): high-order Hits@K
    """
    ts = time.time()
    s = T.shape
    n = len(test)
    base_ind = test[:,0]
    test = np.delete(test, obj = 0, axis = 1)
    nK = len(KList)
    hit_pair = np.zeros(nK)
    hit_high = np.zeros(nK)
    for i in range(n):
        t = np.take(T, base_ind[i], axis = 0)  # take alignment score along dim, shape = [n1, ..., n_dim-1, n_dim+1, ..., nK]
        max_list = np.array(np.unravel_index(t.argsort(axis = None)[::-1], s[1:])).T
        for j in range(nK):
            if len(np.where((max_list[:KList[j],:] == test[i]).all(axis = 1))[0]) != 0:
                hit_high[j:] += 1
                hit_pair[j:] += 1
                break
            elif len(np.where((max_list[:KList[j],:] == test[i]).any(axis = 1))[0]) != 0:
                hit_pair[j] += 1     
    hit_pair = hit_pair / n
    hit_high = hit_high / n
    print('Time for Hits@K: %.2fs'%(time.time() - ts))
    return hit_pair, hit_high

def get_hits_c(T, KList, test, cList, max_n):
    """ Get Hits@K metrics for cluster MOT (currently only support 3 graphs)

    Args:
        T (np.array): soft alignment score, shape = [n1, ..., nK]
        KList (list): list of Ks
        test (list): test set inside the cluster, shape = [n, K]
        cList: list of nodes in the same cluster for different graphs, shape = [K,-1]
        max_n (int): number of nodes in the graph

    Returns:
        hit_pair (np.array): pair-wise Hits@K
        hit_high (np.array): high-order Hits@K
    """
    s = T.shape
    n,K = np.shape(test)
    base_ind = test[:,0]
    test = np.delete(test, obj = 0, axis = 1)
    nK = len(KList)
    hit_pair = np.zeros(nK)
    hit_high = np.zeros(nK)
    mrr = 0
    for i in range(n):
        inside = False
        for j in range(K-1):
            if test[i,j] in cList[j+1]:
                inside = True
                break
        if not inside:
            mrr += 1 / max_n
            continue
        ind = np.where(cList[0]==base_ind[i])
        t = np.take(T, ind, axis = 0)  # take alignment score along dim, shape = [n1, ..., n_dim-1, n_dim+1, ..., nK]
        max_list = np.array(np.unravel_index(t.argsort(axis = None)[::-1], s[1:]))
        temp_list = np.empty((0,np.shape(max_list)[1]), dtype = np.int32)
        for j in range(1,K):
            temp_list = np.vstack((temp_list,cList[j][max_list[j-1,:]]))
        max_list = temp_list.T.astype(np.int32)
        
        hh_ind = np.where((max_list == test[i]).all(axis = 1))[0]
        ph_ind = np.where((max_list == test[i]).any(axis = 1))[0]
        if len(hh_ind) != 0:
            mrr += 1 / (hh_ind.min() + 1)
            for j in range(nK):
                if KList[j] > hh_ind.min():
                    hit_high[j:] += 1
                    break
        if len(ph_ind) != 0:
            for j in range(nK):
                if KList[j] > ph_ind.min():
                    hit_pair[j:] += 1
                    break

    return hit_pair, hit_high, mrr


def get_intra_cost(R_list, X_list, A_list, alpha):
    """intra-cost matrices

    Args:
        R_list ([np.array]): list of rwr score matrices each with shape [ni, m]
        X_list ([np.array]): list of node attributes each with shape [ni, d]
        alpha (float): balancing weight between rwr and attribute
    
    Returns:
        C_list ([np.array]): list of intra-cost matrix, each with shape = [ni,ni]
    """
    C_list = []
    K = len(R_list)  # number of graphs
    is_plain = (len(X_list)==0)
    for i in range(K):
        rwr_C = cos(R_list[i],R_list[i])
        if is_plain:
            C_list.append(np.clip(rwr_C,a_min=0,a_max=np.inf)*(1-A_list[i]))
        else:
            attri_C = cos(X_list[i],X_list[i])
            C_list.append(np.clip((1-alpha)*attri_C + alpha*rwr_C,a_min=0,a_max = np.inf)*A_list[i])
    return C_list


def get_cross_cost(R_list, X_list, p = 2):
    """cross-cost tensor

    Args:
        R_list ([np.array]): list of rwr score matrices each with shape [ni, m]
        X_list ([np.array]): list of node attributes each with shape [ni, d]
    
    Returns:
        C (np.array): cross-cost tensor, shape = [n1, ..., nK]
    """
    
    ts = time.time()
    K = len(R_list)  # number of graphs
    N = []
    for i in range(K):
        N.append(len(R_list[i]))
    C = np.zeros(N) # aggregated cost matrix
    is_plain = len(X_list) == 0
    if is_plain:
        m = R_list[0].shape[1]
    else:
        m = R_list[0].shape[1] + X_list[0].shape[1]
    
    for i in range(K):  # normalization
        R_list[i] = R_list[i]/np.power(np.sum(np.power(R_list[i], p), axis = 1), 1/p).reshape(-1,1)
    
    X = {}
    if is_plain:
        for i in range(K):
            X[i] = R_list[i]
    else:
        for i in range(K):
            X_list[i] = X_list[i]/np.power(np.sum(np.power(X_list[i], p), axis = 1), 1/p).reshape(-1,1)
            X[i] = np.concatenate((R_list[i], X_list[i]), axis = 1)

    for i in range(K):
        for j in range(i+1,K):
            shape = np.ones(len(N), dtype = np.int16)
            shape[i] = N[i]
            shape[j] = N[j]
            C += np.reshape(ot.utils.dist(X[i],X[j],p=p), shape)

    return C


def eval_cluster_acc(c, gnd, K):
    n_hit = 0
    n = 0
    num_cluster = len(c)
    for i in range(num_cluster):
        base_ind = 0
        for k in range(K):
            if len(c[k][i]) < len(c[base_ind][i]):
                base_ind = k
        for ele in c[base_ind][i]:
            gnd_index = np.where(gnd[:,base_ind]==ele)[0]
            if len(gnd_index) == 0:
                continue
            gnd_index = gnd[gnd_index].squeeze()
            hit = True
            for k in range(K):
                if gnd_index[k] not in c[k][i]:
                    hit=False
                    break
            if hit:
                n_hit+=1
            n+=1
    return n_hit/n
            
        
        