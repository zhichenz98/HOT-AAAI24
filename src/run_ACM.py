## run experiments on ACM
import numpy as np
import ot
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from hot_utils import get_rwr
from hot import hot

## set parameters
K = 3   # number of graphs
n = 1000 # number of nodes
attributed = True # use node attributes or not
num_cluster = int(n/50)   # number of clusters
alpha = 0.5 # alpha * rwr + (1-alpha) * attribute
lp = 0.1   # weight for proximal regularizer
K_list = [1, 5, 10, 30, 50]  # Hits@K
pkl_path = "dataset/ACM/K{}n{}.pkl".format(K,n)
with open(pkl_path, 'rb') as f:
    dic = pickle.load(f, encoding = 'bytes')
E_list = dic['E_list']
N_list = dic['N_list']
if attributed:
    X_list = np.array(dic['X_list'])
else:
    X_list = []
H = dic['H']
test = dic['test']
A_list = []
for i in range(K):  # generate adjacency matrix from edges
    A = np.zeros((N_list[i], N_list[i]))
    A[E_list[i][0], E_list[i][1]] = 1
    A_list.append(A)
for i in range(len(A_list)):
    print('G%d: #nodes: %d, #edge: %d'%(i+1, len(A_list[i]), np.sum(A_list[i])/2))

mu_list = []    # list of marginal distributions
for i in range(K):
    mu_list.append(ot.utils.unif(N_list[i]))

## in-cluster MOT (run 10 different splits and report the mean/var)
gt = np.concatenate((H,test), axis = 0)
ph = []
hh = []
mrr = []
t = []
skf = StratifiedKFold(n_splits = 10, shuffle = True)
for test_index,train_index in tqdm(skf.split(gt,np.ones(len(gt)))):
    H = gt[train_index]
    test = gt[test_index]
    r = get_rwr(A_list, H)
    tmp_ph, tmp_hh, tmp_mrr, tmp_t = hot(r, X_list, A_list, mu_list, test, alpha, num_cluster, lp, K_list= K_list)
    ph.append(tmp_ph)
    hh.append(tmp_hh)
    mrr.append(tmp_mrr)
    t.append(tmp_t)
avg_ph = 100 * np.mean(ph, axis = 0)
std_ph = 100 * np.std(ph, axis = 0)
avg_hh = 100 * np.mean(hh, axis = 0)
std_hh = 100 * np.std(hh, axis = 0)
avg_mrr = 100 * np.mean(mrr, axis = 0)
std_mrr = 100 * np.std(mrr, axis = 0)
print('Pairwise Hits:')
for i in range(len(K_list)):
    print('Hits@{}: {:.3}±{:.2}'.format(K_list[i], avg_ph[i], std_ph[i]))
print('\nHigh-order Hits:')
for i in range(len(K_list)):
    print('Hits@{}: {:.3}±{:.2}'.format(K_list[i], avg_hh[i], std_hh[i]))
print('MRR: {:.3}±{:.2}'.format(avg_mrr, std_mrr))
print(r'Time: {:.2f}±{:.2f}'.format(np.mean(t), np.std(t)))