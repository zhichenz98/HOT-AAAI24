import numpy as np
import time
import ot
from log_mot import multi_FGW
from hot_utils import get_cross_cost, get_intra_cost, get_hits_c


def get_cluster_fgw(r, x, A_list, mu_list, num_c):
    """ find clusters

    Args:
        r ([np.array]): a list of rwr score (np.array), each with shape = [n, d]
        x ([np.array]): a list of node attributes (np.array), each with shape = [n, m]
        A_list ([np.array]): a list of adjacency matrices, each with shape = [n, n]
        mu_list ([np.array]): a list of marginal distribution, each with shape = [n,]
        num_c (int): number of clusters

    Returns:
        c ([dict]): a list of dictionaries, each dictionary stores cluster-node correspondence in a graph, key: cluster index, value: node index in the cluster
        cr ([dict]): a list of dictionaries, each dictionary stores node-rwr correspodence in a graph, key: cluster index, value: node rwr score (np.array)
        cx ([dict]): a list of dictionaries, each dictionary stores node-attri correspodence in a graph, key: cluster index, value: node attribute (np.array)
    """
    K = len(A_list)
    Ys = []
    norm_r = []
    norm_x = []
    if len(x) == 0:
        for i in range(K):
            ind = np.where(np.sum(r[i]*r[i], axis = 1) == 0)
            r[i][ind] = np.ones(len(r[i][0]))
            norm_r.append(r[i] / np.sqrt(np.sum(r[i]*r[i], axis = 1)).reshape(-1,1))
            Ys.append(norm_r[i])
    else:
        for i in range(K):
            ind = np.where(np.sum(r[i]*r[i], axis = 1) == 0)
            r[i][ind] = np.ones(len(r[i][0]))
            norm_r.append(r[i] / np.sqrt(np.sum(r[i]*r[i], axis = 1)).reshape(-1,1))
            norm_x.append(x[i] / np.sqrt(np.sum(x[i]*x[i], axis = 1)).reshape(-1,1))
            Ys.append(np.concatenate((norm_r[i], norm_x[i]), axis = 1))
    
    # T_list['T'] ([np.array]) is the transport plan from G->G_i, each with shape [num_c,n_i]
    _, _, T_list = ot.gromov.fgw_barycenters(num_c, Ys, A_list, mu_list, alpha = 0.5, log = True)
    
    c = []  # record cluster info, [dict(k:#c, v:#node)]
    cr = [] # record clustered rwr
    cx = [] # record clustered node attributes
    ca = [] # record clustered adjcency matrix

    for i in range(K):
        ### Assign nodes based on transport plan
        l = np.argmax(T_list['T'][i], axis = 0)
        ind = {}
        temp_r = {}
        temp_x = {}
        temp_a = {}
        for j in range(num_c):
            ind[j] = np.where(l == j)[0]
            temp_r[j] = r[i][ind[j],:]
            temp_a[j] = A_list[i][np.ix_(ind[j], ind[j])]
            if len(x) != 0: # attributed network
                temp_x[j] = x[i][ind[j],:]
        c.append(ind)
        cr.append(temp_r)
        if len(x) != 0:
            cx.append(temp_x)
        ca.append(temp_a)
    return c, cr, cx, ca


def hot(r, X_list, A_list, mu_list, test, alpha = 0.5, num_c = 10, lp =1e-3, K_list = [1,5,10,30,50]):
    """ Hierarchical Optimal Transport

    Args:
        r ([np.array]): an array of rwr score, shape = [K, n, d]
        X_list ([np.array]): a list of node attributes (np.array), each with shape = [n, m]
        test (np.array): test dataset, shape = [n_test, K]
        alpha (float): weight between rwr cost and attribute cost
        num_c (int): number of clusters
        lp (float): convergence threshold
        K_list (list): top-K
        
    Returns:
        hit_pair: a list of pairwise Hits@K
        hit_high: a list of high-order Hits@K
        mrr: a list of MRR
        run_time: running time
    """
    K = len(r)  # number of distributions
    # get clusters c, corresponding rwr cr, and node attribute cx
    max_n = len(A_list[0])  # number of nodes in the graph
    ts = time.time()
    # calculate intra-cost of different networks
    C_list = get_intra_cost(r, X_list, A_list, alpha)
    c, cr, cx, ca = get_cluster_fgw(r, X_list, C_list, mu_list, num_c)
    
    hit_pair = 0
    hit_high = 0
    mrr = 0
    run_time = 0
    
    for i in range(num_c):  # execute MOT in each cluster
        r = []  # rwr score for nodes in cluster i
        x = []  # node attributes for nodes in cluster i
        mu = [] # marginal distribution for nodes in cluster i
        c_list = []  # node index in clutser i
        A = []
        for j in range(K):
            r.append(cr[j][i])
            mu.append(np.ones(len(r[-1]))/len(r[-1]))
            c_list.append(c[j][i])
            if len(cx) != 0:    # attributed network
                x.append(cx[j][i])
            A.append(ca[j][i])
        C = get_cross_cost(r, x)
        ts = time.time()
        T = multi_FGW(C, A, mu, lp = lp, alpha = alpha)
        run_time += time.time() - ts
        ind = np.nonzero(c_list[0][:, None] == test[:,0])[1] # find test cases in cluster i
        c_test = test[ind]
        ph, hh, tmp_mrr = get_hits_c(T, K_list, c_test, c_list, max_n)
        hit_pair += ph
        hit_high += hh
        mrr += tmp_mrr
    hit_pair = hit_pair / len(test)
    hit_high = hit_high / len(test)
    mrr = mrr / len(test)
    return hit_pair, hit_high, mrr, run_time

