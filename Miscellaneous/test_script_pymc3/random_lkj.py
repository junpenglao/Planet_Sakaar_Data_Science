#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:09:02 2017

@author: jlao
"""

import numpy as np
from scipy import stats


n = 3
eta=.9

beta0 = eta - 1 + n/2
shape = n * (n-1) // 2
triu_ind = np.triu_indices(n, 1)
beta = np.array([beta0 - k/2 for k in triu_ind[0]])
# partial correlations sampled from beta dist.
P0 = np.ones((n, n))
P0[triu_ind] = stats.beta.rvs(a=beta, b=beta).T
P0[np.tril_indices(n, -1)] = P0[triu_ind]
# scale partial correlation matrix to [-1, 1]
P0 = (P0 - .5) * 2
r_triu = []
#%%
P = P0.copy()

for k, i in zip(triu_ind[0], triu_ind[1]):
    p = P[k, i]
    for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
        p = p * np.sqrt((1 - P[l, i]**2) *
                        (1 - P[l, k]**2)) + P[l, i] * P[l, k]
    P[k, i] = p
    P[i, k] = p
P1 = P.copy()
#%%
P = P0.copy()
P2 = np.eye(n)
for j in range(n):
    for i in range(j+1, n):
        sumsqs = 0
        for ij in range(j):
            sumsqs += P2[i, ij]**2
        P2[i,j] = P[i,j]*np.sqrt(1-sumsqs)
    sumsqs = 0
    for ij in range(j-1):
        sumsqs += P2[j, ij]**2
    P2[j,j] = np.sqrt(1-sumsqs)
P2 = np.copy(P2@P2.T)
print(np.sum(P1-P2))
#%%
tau = np.linalg.inv(P1)

partial = np.eye(n)
for i in range(n):
    for j in range(i+1, n):
        ptemp=-tau[i,j]/np.sqrt(tau[i,i]*tau[j,j])
        partial[i,j]=ptemp
        partial[j,i]=ptemp
print(partial)
#%%
import numpy as np
from scipy import stats

def lkj_random(n, eta, size=None):
    beta0 = eta - 1 + n/2
    shape = n * (n-1) // 2
    triu_ind = np.triu_indices(n, 1)
    beta = np.array([beta0 - k/2 for k in triu_ind[0]])
    # partial correlations sampled from beta dist.
    P = np.ones((n, n) + (size,))
    P[triu_ind] = stats.beta.rvs(a=beta, b=beta, size=(size,) + (shape,)).T
    # scale partial correlation matrix to [-1, 1]
    P = (P-.5)*2
    
    for k, i in zip(triu_ind[0], triu_ind[1]):
        p = P[k, i]
        for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
            p = p * np.sqrt((1 - P[l, i]**2) *
                            (1 - P[l, k]**2)) + P[l, i] * P[l, k]
        P[k, i] = p
        P[i, k] = p

    return np.transpose(P, (2, 0 ,1))
#%%
def lkj_random2(n, eta, size=None):
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
#%%
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
P = lkj_random2(n, eta, size)
k=0
for i, p in enumerate(P):
    k+=is_pos_def(p)
print("{0} % of the output matrix is positive definite.".format(k/size*100)) 
#%%
import matplotlib.pylab as plt
# Off diagnoal element
C= P.transpose((1, 2, 0))[np.triu_indices(n, k=1)].T
fig, ax = plt.subplots()

ax.hist(C.flatten(), 100, normed=True)

beta = eta - 1 + n/2
C2 = 2 * stats.beta.rvs(size=C.shape, a=beta, b=beta)-1
ax.hist(C2.flatten(), 100, normed=True, histtype='step', label='Beta() distribution')
plt.legend(loc='upper right', frameon=False);
#%%
n = 5
eta=1.
size=100

beta0 = eta - 1 + n/2
shape = n * (n-1) // 2

lkj_random(shape, n, eta, size).shape

#%%
def broadcast_shapes(*args):
    """Return the shape resulting from broadcasting multiple shapes.
    Represents numpy's broadcasting rules.

    Parameters
    ----------
    *args : array-like of int
        Tuples or arrays or lists representing the shapes of arrays to be broadcast.

    Returns
    -------
    Resulting shape or None if broadcasting is not possible.
    """
    x = list(np.atleast_1d(args[0])) if args else ()
    for arg in args[1:]:
        y = list(np.atleast_1d(arg))
        if len(x) < len(y):
            x, y = y, x
        x[-len(y):] = [j if i == 1 else i if j == 1 else i if i == j else 0
                       for i, j in zip(x[-len(y):], y)]
        if not all(x):
            return None
    return tuple(x)


def infer_shape(shape):
    try:
        shape = tuple(shape or ())
    except TypeError:  # If size is an int
        shape = tuple((shape,))
    except ValueError:  # If size is np.array
        shape = tuple(shape)
    return shape


def reshape_sampled(sampled, size, dist_shape):
    dist_shape = infer_shape(dist_shape)
    repeat_shape = infer_shape(size)

    if np.size(sampled) == 1 or repeat_shape or dist_shape:
        return np.reshape(sampled, repeat_shape + dist_shape)
    else:
        return sampled


def replicate_samples(generator, size, repeats, *args, **kwargs):
    n = int(np.prod(repeats))
    print(n)
    if n == 1:
        samples = generator(size=size, *args, **kwargs)
    else:
        samples = np.array([generator(size=size, *args, **kwargs)
                            for _ in range(n)])
        samples = np.reshape(samples, tuple(repeats) + tuple(size))
    return samples


def generate_samples(generator, *args, **kwargs):
    """Generate samples from the distribution of a random variable.

    Parameters
    ----------
    generator : function
        Function to generate the random samples. The function is
        expected take parameters for generating samples and
        a keyword argument `size` which determines the shape
        of the samples.
        The *args and **kwargs (stripped of the keywords below) will be
        passed to the generator function.

    keyword arguments
    ~~~~~~~~~~~~~~~~

    dist_shape : int or tuple of int
        The shape of the random variable (i.e., the shape attribute).
    size : int or tuple of int
        The required shape of the samples.
    broadcast_shape: tuple of int or None
        The shape resulting from the broadcasting of the parameters.
        If not specified it will be inferred from the shape of the
        parameters. This may be required when the parameter shape
        does not determine the shape of a single sample, for example,
        the shape of the probabilities in the Categorical distribution.

    Any remaining *args and **kwargs are passed on to the generator function.
    """
    dist_shape = kwargs.pop('dist_shape', ())
    size = kwargs.pop('size', None)
    broadcast_shape = kwargs.pop('broadcast_shape', None)
    params = args + tuple(kwargs.values())

    if broadcast_shape is None:
        broadcast_shape = broadcast_shapes(*[np.atleast_1d(p).shape for p in params
                                             if not isinstance(p, tuple)])
    if broadcast_shape == ():
        broadcast_shape = (1,)

    args = tuple(p[0] if isinstance(p, tuple) else p for p in args)
    for key in kwargs:
        p = kwargs[key]
        kwargs[key] = p[0] if isinstance(p, tuple) else p

    if np.all(dist_shape[-len(broadcast_shape):] == broadcast_shape):
        prefix_shape = tuple(dist_shape[:-len(broadcast_shape)])
    else:
        prefix_shape = tuple(dist_shape)

    repeat_shape = infer_shape(size)
    
    print(broadcast_shape)
    print(prefix_shape)
    print(size)
    
    print(repeat_shape)
    if broadcast_shape == (1,) and prefix_shape == ():
        if size is not None:
            samples = generator(size=size, *args, **kwargs)
        else:
            samples = generator(size=1, *args, **kwargs)
    else:
        if size is not None:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        repeat_shape + prefix_shape,
                                        *args, **kwargs)
        else:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        prefix_shape,
                                        *args, **kwargs)
    return reshape_sampled(samples, size, dist_shape)

#%%
import pymc3 as pm
from pymc3.distributions.distribution import draw_values
with pm.Model() as model:
    lkj=pm.LKJCorr('lkj', n=5, eta=1.)

n, eta = draw_values([lkj.distribution.n, lkj.distribution.eta], point=model.test_point)
testlkj=lkj.distribution
size=100
samples = generate_samples(testlkj._random, n, eta,
                           broadcast_shape=(size,))