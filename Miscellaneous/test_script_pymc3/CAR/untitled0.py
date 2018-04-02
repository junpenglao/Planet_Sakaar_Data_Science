#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:01:39 2017

@author: laoj
"""
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.distributions import distribution
import scipy
import theano
import theano.tensor as tt

data_df = pd.read_csv('dataset.csv')
adjacency = np.genfromtxt("adjacency.csv", delimiter=",")
adjacency2 = np.genfromtxt("adjacency2.csv", delimiter=",")

y = data_df['ncrimes'].values.astype('float32')
offset = data_df['population'].astype('float32')

class CAR3(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) with 2nd order neighbors
    """

    def __init__(self, adjacency, tau, rho, adjacency2, rho2, *args, **kwargs):
        super(CAR3, self).__init__(*args, **kwargs)
        n, m = adjacency.shape
        self.n = n

        adjacency_sparse = scipy.sparse.csr_matrix(adjacency)
        self.adjacency = theano.sparse.as_sparse_variable(adjacency_sparse)
        adjacency_sparse2 = scipy.sparse.csr_matrix(adjacency2)
        self.adjacency2 = theano.sparse.as_sparse_variable(adjacency_sparse2)

        self.n_neighbors = tt.as_tensor_variable(adjacency.sum(1))
        self.n_neighbors2 = tt.as_tensor_variable(adjacency2.sum(1))
        self.mean = tt.zeros(n)
        self.median = self.mean
        self.tau = tt.as_tensor_variable(tau)
        self.rho = tt.as_tensor_variable(rho)
        self.rho2 = tt.as_tensor_variable(rho2)

    def logp(self, x):
        priorvardenom = 1 - (self.rho + self.rho2) + self.rho * self.n_neighbors + self.rho2 * self.n_neighbors2
        #priorvar = self.tau * priorvardenom
        priorvar = tt.sqrt(priorvardenom/self.tau)

        Wx = theano.sparse.dot(self.adjacency, x.reshape((self.n, 1)))
        Wx2 = theano.sparse.dot(self.adjacency2, x.reshape((self.n, 1)))
        mu_w = (self.rho * tt.sum(Wx, axis=1) + self.rho2*tt.sum(Wx2, axis=1)) / priorvardenom

        return pm.Normal.dist(mu=mu_w, tau=priorvar).logp(x)
#%%
with pm.Model() as pooled_model:
    b0 = pm.Normal('intercept', mu=0, tau=0.001)

    rho = pm.Uniform('rho', 0, 1)
    rho2 = pm.Uniform('rho2', 0, 1)
    tau = pm.HalfCauchy('tau', 2.5)
    mu_phi = CAR3('mu_phi', adjacency=adjacency, rho=rho, adjacency2=adjacency2, rho2=rho2, tau=tau, shape=len(y))
    #phi2 = pm.Deterministic('phi', mu_phi - tt.mean(mu_phi))

    mu = tt.exp(b0 + mu_phi  + tt.log(offset))

    ncrimes = pm.Poisson('ncrimes', mu=mu, observed=y)

    pooled_trace = pm.sample(1000, njobs=4, tune=1000)  # , , tune=1000

pm.traceplot(pooled_trace);

#%%
W1, W2 = adjacency, adjacency2
D1 = np.diag(W1.sum(axis=1))
D2 = np.diag(W2.sum(axis=1))
n_neighbors1 = tt.as_tensor_variable(W1.sum(1))
n_neighbors2 = tt.as_tensor_variable(W2.sum(1))
with pm.Model() as model:
    b0 = pm.Normal('intercept', mu=0, sd=100)
    
    # Priors for spatial random effects 1
    tau1 = pm.HalfCauchy('tau1', 2.5)
    rho1 = pm.Uniform('rho1', lower=0, upper=1)
    phi1 = pm.MvNormal('phi1', mu=0, tau=tau1*(D1 - rho1*W1), shape=(1, len(y)))
    
    # Priors for spatial random effects 1
    tau2 = pm.HalfCauchy('tau2', 2.5)
    rho2 = pm.Uniform('rho2', lower=0, upper=1)
    phi2 = pm.MvNormal('phi2', mu=0, tau=tau2*(D2 - rho2*W2), shape=(1, len(y)))
    
    mu = tt.exp(phi1.T + phi2.T) + b0 + offset[:, np.newaxis]

    ncrimes = pm.Poisson('ncrimes', mu=mu, observed=y)
    pooled_trace = pm.sample(1000, njobs=4)
#%%
njobs=4
from pymc3.step_methods.hmc import quadpotential
with model:
    approx = pm.fit(
            n=200000, method='advi', 
            progressbar=True,
            obj_optimizer=pm.adagrad_window,
        )
    start = approx.sample(draws=njobs)
    start = list(start)
    stds = approx.gbij.rmap(approx.std.eval())
    cov = model.dict_to_array(stds) ** 2
    mean = approx.gbij.rmap(approx.mean.get_value())
    mean = model.dict_to_array(mean)
    weight = 50
    potential = quadpotential.QuadPotentialDiagAdapt(
        model.ndim, mean, cov, weight)
    step = pm.NUTS(potential=potential)
    pooled_trace = pm.sample(1000, step=step, njobs=njobs, tune=1000)
    
pm.traceplot(pooled_trace);