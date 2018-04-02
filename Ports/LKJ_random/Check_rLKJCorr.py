"""
Random Correlation matrix (LKJ 2009) output checking

Created on Wed Aug  2 09:09:02 2017

@author: junpenglao
"""
import numpy as np
from scipy import stats

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return 1
        except np.linalg.linalg.LinAlgError:
            return 0
    else:
        return 0


n = 10
eta = 1.
size = 1000
P = lkj_random(n, eta, size)
k=0
for i, p in enumerate(P):
    k+=is_pos_def(p)
print("{0} % of the output matrix is positive definite.".format(k/size*100))

import matplotlib.pylab as plt 
# Off diagnoal element 
C= P.transpose((1, 2, 0))[np.triu_indices(n, k=1)].T 
fig, ax = plt.subplots()  
ax.hist(C.flatten(), 100, normed=True) 
 
beta = eta - 1 + n/2 
C2 = 2 * stats.beta.rvs(size=C.shape, a=beta, b=beta)-1 
ax.hist(C2.flatten(), 100, normed=True, histtype='step', label='Beta() distribution') 
plt.legend(loc='upper right', frameon=False); 

