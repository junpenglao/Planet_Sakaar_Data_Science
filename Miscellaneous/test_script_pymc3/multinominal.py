#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:30:53 2017

@author: laoj
"""
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.distribution import Discrete, draw_values, generate_samples, infer_shape
from pymc3.distributions.dist_math import bound, logpow, factln, Cholesky
from pymc3.math import tround

#%% n scaler, p 1D
#n = 183
n = np.array([[106],
 [143],
 [102],
 [116],
 [183],
 [150]])
p = np.array([[ 0.21245365,  0.41223126,  0.37531509],
 [ 0.13221011,  0.50537169,  0.3624182 ],
 [ 0.08813779,  0.54447146,  0.36739075],
 [ 0.18932804,  0.4630365,   0.34763546],
 [ 0.11006472,  0.49227755,  0.39765773],
 [ 0.17886852,  0.41098834, 0.41014314]])

# p = np.array([ 0.21245365,  0.41223126,  0.37531509])
n = tt.as_tensor_variable(n)
p = tt.as_tensor_variable(p)
n = np.squeeze(n)
n = tt.shape_padright(n) if n.ndim == 1 else tt.as_tensor_variable(n)
n.ndim
n * p
#%%
n = np.array([[106],
 [143],
 [102],
 [116],
 [183],
 [150]])

#n = 183
p = np.array([[ 0.21245365,  0.41223126,  0.37531509],
 [ 0.13221011,  0.50537169,  0.3624182 ],
 [ 0.08813779,  0.54447146,  0.36739075],
 [ 0.18932804,  0.4630365,   0.34763546],
 [ 0.11006472,  0.49227755,  0.39765773],
 [ 0.17886852,  0.41098834, 0.41014314]])
#p = np.array([[ 0.21245365,  0.41223126,  0.37531509]])

#n = tt.as_tensor_variable(n)
p = tt.as_tensor_variable(p)

#%%
class Multinomial(Discrete):

    def __init__(self, n, p, *args, **kwargs):
        super(Multinomial, self).__init__(*args, **kwargs)

        p = p / tt.sum(p, axis=-1, keepdims=True)
        n = np.squeeze(n) # works also if n is a tensor

        if len(self.shape) > 1:
            m = self.shape[-2]
            try:
                assert n.shape == (m,)
            except (AttributeError, AssertionError):
                n = n * tt.ones(m)
            self.n = tt.shape_padright(n)
            self.p = p if p.ndim > 1 else tt.shape_padleft(p)
        elif n.ndim == 1:
            self.n = tt.shape_padright(n)
            self.p = p if p.ndim > 1 else tt.shape_padleft(p)
        else:
            # n is a scalar, p is a 1d array
            self.n = tt.as_tensor_variable(n)
            self.p = tt.as_tensor_variable(p)

        self.mean = self.n * self.p
        mode = tt.cast(tt.round(self.mean), 'int32')
        diff = self.n - tt.sum(mode, axis=-1, keepdims=True)
        inc_bool_arr = tt.abs_(diff) > 0
        mode = tt.inc_subtensor(mode[inc_bool_arr.nonzero()],
                                diff[inc_bool_arr.nonzero()])
        self.mode = mode

    def _random(self, n, p, size=None):
        original_dtype = p.dtype
        # Set float type to float64 for numpy. This change is related to numpy issue #8317 (https://github.com/numpy/numpy/issues/8317)
        p = p.astype('float64')
        # Now, re-normalize all of the values in float64 precision. This is done inside the conditionals
        if size == p.shape:
            size = None
        if (p.ndim == 1) and (n.ndim == 0):
            p = p / p.sum()
            randnum = np.random.multinomial(n, p.squeeze(), size=size)
        else:
            p = p / p.sum(axis=1, keepdims=True)
            if n.shape[0] > p.shape[0]:
                randnum = np.asarray([
                    np.random.multinomial(nn, p.squeeze(), size=size)
                    for nn in n
                ])
            elif n.shape[0] < p.shape[0]:
                randnum = np.asarray([
                    np.random.multinomial(n.squeeze(), pp, size=size)
                    for pp in p
                ])
            else:
                randnum = np.asarray([
                    np.random.multinomial(nn, pp, size=size)
                    for (nn, pp) in zip(n, p)
                ])
        return randnum.astype(original_dtype)

    def random(self, point=None, size=None):
        n, p = draw_values([self.n, self.p], point=point)
        samples = generate_samples(self._random, n, p,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, x):
        n = self.n
        p = self.p

        return bound(
            tt.sum(factln(n)) - tt.sum(factln(x)) + tt.sum(x * tt.log(p)),
            tt.all(x >= 0),
            tt.all(tt.eq(tt.sum(x, axis=-1, keepdims=True), n)),
            tt.all(p <= 1),
            tt.all(tt.eq(tt.sum(p, axis=-1), 1)),
            tt.all(tt.ge(n, 0)),
            broadcast_conditions=False
        )

Multinomial.dist(1,np.ones(3)/3,shape=(6, 3)).mode.eval()
#%%
Multinomial.dist(n,p,shape=(6, 3)).p.eval()
#%%
Multinomial.dist(n,p,shape=(6, 3)).n.eval()
#%%
Multinomial.dist(n,p,shape=(6, 3)).mean.eval()
#%%
Multinomial.dist(n,p,shape=(6, 3)).random()
#%%
counts =np.asarray([[19, 50, 37],
         [21, 67, 55],
         [11, 53, 38],
         [17, 54, 45],
         [24, 93, 66],
         [27, 53, 70]])
Multinomial.dist(n,p,shape=(6, 3)).logp(x=counts).eval()
#%%
with pm.Model() as model:
    like = Multinomial('obs_ABC', n, p, observed=counts, shape=counts.shape)
#%%
paramall = (
        [[.25, .25, .25, .25], 4, 2],
        [[.25, .25, .25, .25], (1, 4), 3],
        # 3: expect to fail
        # [[.25, .25, .25, .25], (10, 4)],
        [[.25, .25, .25, .25], (10, 1, 4), 5],
        # 5: expect to fail
        # [[[.25, .25, .25, .25]], (2, 4), [7, 11]],
        [[[.25, .25, .25, .25],
         [.25, .25, .25, .25]], (2, 4), 13],
        [[[.25, .25, .25, .25],
         [.25, .25, .25, .25]], (2, 4), [17, 19]],
        [[[.25, .25, .25, .25],
         [.25, .25, .25, .25]], (1, 2, 4), [23, 29]],
        [[[.25, .25, .25, .25],
         [.25, .25, .25, .25]], (10, 2, 4), [31, 37]],
       )
for p, shape, n in paramall:
    with pm.Model() as model:
        m = Multinomial('m', n=n, p=np.asarray(p), shape=shape)
    print(m.random().shape)
#%%
counts =np.asarray([[19, 50, 37],
         [21, 67, 55],
         [11, 53, 38],
         [17, 54, 45],
         [24, 93, 66],
         [27, 53, 70]])
n = np.array([[106],
 [143],
 [102],
 [116],
 [183],
 [150]])
sparsity=1 #not zero
beta=np.ones(counts.shape) #input for dirichlet

with pm.Model() as model:
    theta=pm.Dirichlet('theta',beta/sparsity, shape = counts.shape)
    transition=pm.Multinomial('transition',n,theta,observed=counts)
    trace=pm.sample(1000)
#%%
import numpy as np
import pymc3 as pm
import theano.tensor as tt

def norm_simplex(p):
    """Sum-to-zero transformation."""
    return (p.T / p.sum(axis=-1)).T

def ccmodel(beta, x):
    """Community composition model."""
    return norm_simplex(tt.exp(tt.dot(x, tt.log(beta))))

class DirichletMultinomial(pm.Discrete):
    """Dirichlet Multinomial Model

    """
    def __init__(self, alpha, *args, **kwargs):
        super(DirichletMultinomial, self).__init__(*args, **kwargs)
        self.alpha = alpha

    def logp(self, x):
        alpha = self.alpha
        n = tt.sum(x, axis=-1)
        sum_alpha = tt.sum(alpha, axis=-1)

        const = (tt.gammaln(n + 1) + tt.gammaln(sum_alpha)) - tt.gammaln(n + sum_alpha)
        series = tt.gammaln(x + alpha) - (tt.gammaln(x + 1) + tt.gammaln(alpha))
        result = const + tt.sum(series, axis=-1)
        return result

def as_col(x):
    if isinstance(x, tt.TensorVariable):
        return x.dimshuffle(0, 'x')
    else:
        return np.asarray(x).reshape(-1, 1)

def as_row(x):
    if isinstance(x, tt.TensorVariable):
        return x.dimshuffle('x', 0)
    else:
        return np.asarray(x).reshape(1, -1)


n, k, r = 25, 10, 2

x = np.random.randint(0, 1000, size=(n, k))
y = np.random.randint(0, 1000, size=n)
design = np.vstack((np.ones(25), np.random.randint(2, size=n))).T

with pm.Model() as model:
    # Community composition
    pi = pm.Dirichlet('pi', np.ones(k), shape=(r, k))
    comp = pm.Deterministic('comp', ccmodel(pi, design))

    # Inferred population density of observed taxa (hierarchical model)
    rho = pm.Normal('rho', shape=r)
    tau = pm.Lognormal('tau')
    dens = pm.Lognormal('dens', tt.dot(design, rho), tau=tau, shape=n)

    # Community composition *with* the spike
    expected_recovery = as_col(1 / dens)
    _comp = norm_simplex(tt.concatenate((comp, expected_recovery), axis=1))  

    # Variability
    mu = pm.Lognormal('mu')

    # Data
    obs = DirichletMultinomial('obs', _comp * mu,
                               observed=tt.concatenate((x, as_col(y)), axis=1))
    
    pm.sample(1000)