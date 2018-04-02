"""
Random Correlation matrix (LKJ 2009)
ported R code from @rmcelreath, original see 
https://github.com/rmcelreath/rethinking/blob/master/R/distributions.r#L165-L184

Created on Wed Aug  2 09:09:02 2017

@author: junpenglao
"""
import numpy as np
from scipy import stats

def lkj_random(n, eta, size=None):
    size = size if isinstance(size, tuple) else (size,) 
    beta = eta - 1 + n/2
    r12 = 2 * stats.beta.rvs(a=beta, b=beta, size=size) - 1
    P = np.eye(n)[:,:,np.newaxis] * np.ones(size)
    P = np.transpose(P, (2, 0 ,1))
    P[:, 0, 1] = r12
    P[:, 1, 1] = np.sqrt(1 - r12**2)
    if n > 2:
        for m in range(1, n-1):
            beta -= 0.5
            y = stats.beta.rvs(a=(m+1) / 2., b=beta, size=size)
            z = stats.norm.rvs(loc=0, scale=1, size=(m+1, ) + size)
            z = z/np.sqrt(np.einsum('ij,ij->j', z, z))
            P[:, 0:m+1, m+1] = np.transpose(np.sqrt(y) * z)
            P[:, m+1, m+1] = np.sqrt(1-y)
    
    C = np.einsum('...ji,...jk->...ik', P, P)
    return C