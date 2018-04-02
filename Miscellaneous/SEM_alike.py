#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Mar 22 14:56:24 2017

@author: laoj
'''

#%%
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import theano.tensor as tt
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

# Simulation data
# latent factor
Nd = 3
N = 200

sigma1 = 1.5
sigma2 = .78
sigma3 = 1.1

lambda1 = 1/np.square(sigma1)
lambda2 = 1/np.square(sigma2)
lambda3 = 1/np.square(sigma3)

r12 = -.5
r13 = .75
r23 = r12*r13

jitter = 1e-6
K = np.stack([[lambda1**-1, r12*sigma1*sigma2, r13*sigma1*sigma3],
              [r12*sigma1*sigma2, lambda2**-1, r23*sigma2*sigma3],
              [r13*sigma1*sigma3,r23*sigma2*sigma3,lambda3**-1]]) + jitter*np.eye(Nd)
#K = np.array([[  3.40, -2.75, -2.00],
#              [ -2.75,  5.50,  1.50],
#              [ -2.00,  1.50,  1.25]])

latent = stats.multivariate_normal.rvs(np.zeros(Nd), K, size=N)

df_lat = pd.DataFrame(data=latent)
sns.pairplot(df_lat)
#%%
Nd1 = 5
X1 = stats.norm.rvs(loc=0,scale=.5,size=(Nd1,N))
W = stats.norm.rvs(loc=0,scale=1,size=(Nd1,Nd1))
K2 = W*W.transpose() + np.diagflat(np.ones((Nd1,1))*Nd1)
L = np.linalg.cholesky(K2)
mu1 = np.asarray([1.,2.,1.5,5.,3.])
factor1 = latent[:,0]
# X = stats.multivariate_normal.rvs(np.zeros(Nd), K, size=N)
obs1 = np.transpose(np.outer(mu1,factor1) + np.dot(L,X1))

var2 = .1
mu2 = np.asarray([1.])
factor2 = latent[:,1]
obs2 = np.transpose(np.outer(mu2,factor2) + 
                    stats.norm.rvs(loc=0,scale=var2,size=(1,N)))

var3 = 2.5
mu3 = np.asarray([1.2,2.1])
factor3 = latent[:,2]
obs3 = np.transpose(np.outer(mu3,factor3) + 
                    stats.norm.rvs(loc=0,scale=var3,size=(2,N)))

df = pd.DataFrame(data=np.hstack([obs1,obs2,obs3]))
sns.pairplot(df)
#%%
import numpy as np

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False

from theano import tensor
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.tensor.slinalg import Solve

solve_upper_triangular = Solve(A_structure='upper_triangular', lower=False)
class QR_Chol(Op):
    """
    (mostly) Copy from theano:
    
    Incomplete QR Decomposition.
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr and return a single matrix.
    """
    __props__ = ('mode',)

    def __init__(self, mode):
        self.mode = mode

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input should be a matrix."
        z = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [z])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        assert x.ndim == 2, "The input should be a matrix."
        w, u, v = scipy.linalg.svd(x, 1, 1)
        tmp = np.dot(np.diag(np.sqrt(u)), w.T)
        r = scipy.linalg.qr(tmp, mode=self.mode)
        z[0] = r[0].T
        
    def grad(self, inputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.
        Symbolic expression for reverse-mode Cholesky gradient taken from [0]_
        References
        ----------
        .. [0] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527
        """

        x = inputs[0]
        dz = gradients[0]
        chol_x = self(x)

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return tensor.tril(mtx) - tensor.diag(tensor.diagonal(mtx) / 2.)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            return solve_upper_triangular(
                outer.T, solve_upper_triangular(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz)))

        return [tensor.tril(s + s.T) - tensor.diag(tensor.diagonal(s))]
        
def qr_chol(a, mode="r"):
    return QR_Chol(mode)(a)

#%% PyMC3 model - directly on latent - v1
jitter = 1e-6
with pm.Model() as model:
    # r∼Uniform(−1,1)
    r =  pm.Uniform('r',lower=-1, upper=1, shape=Nd)
    
    # μ1,μ2∼Gaussian(0,.001)
    mu = pm.Normal('mu', mu=0, tau=.001, shape=Nd)
    
    # σ1,σ2∼InvSqrtGamma(.001,.001)
    lam = pm.Gamma('lambda', alpha=.001, beta=.001, shape=Nd)
    sd = pm.Deterministic('sigma', 1/np.sqrt(lam))
    
    cov = pm.Deterministic('cov', 
                           tt.stacklists([[lam[0]**-1, r[0]*sd[0]*sd[1], r[1]*sd[0]*sd[2]],
                                          [r[0]*sd[0]*sd[1], lam[1]**-1, r[2]*sd[1]*sd[2]],
                                          [r[1]*sd[0]*sd[2], r[2]*sd[1]*sd[2], lam[2]**-1]]))
    
    # chol = qr_chol(cov)
    
    lat_factor = pm.MvNormal('latent', mu=mu, cov=cov, observed=latent)
    trace_lat = pm.sample(2000, njobs=4)
#    start = pm.find_MAP()
#    step = pm.NUTS(scaling=start)
#    trace_lat = pm.sample(2000, step=step, start=start, njobs=2)
#%% PyMC3 model - directly on latent - v2
# In order to convert the upper triangular correlation values to a complete
# correlation matrix, we need to construct an index matrix:
n_elem = int(Nd * (Nd - 1) / 2)
tri_index = np.zeros([Nd, Nd], dtype=int)
tri_index[np.triu_indices(Nd, k=1)] = np.arange(n_elem)
tri_index[np.triu_indices(Nd, k=1)[::-1]] = np.arange(n_elem)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1, shape=Nd)

    # We can specify separate priors for sigma and the correlation matrix:
    sd = pm.Uniform('sigma', shape=Nd)
    lam = pm.Deterministic('lambda', 1/tt.sqr(sd))
    
    corr_triangle = pm.LKJCorr('r', eta=2, n=Nd)
    corr_matrix = corr_triangle[tri_index]
    corr_matrix = tt.fill_diagonal(corr_matrix, 1)

    cov = tt.diag(sd).dot(corr_matrix.dot(tt.diag(sd)))
    # chol = qr_chol(cov)
    
    lat_factor = pm.MvNormal('latent', mu=mu, cov=cov, observed=latent)
    
    trace_lat = pm.sample(2000, njobs=4)
#    start = pm.find_MAP()
#    trace_lat = pm.sample(2000, start=start, njobs=2)
#%% PyMC3 model - directly on latent - v3
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1, shape=Nd)

    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_cov', n=Nd, eta=1, sd_dist=sd_dist)
    # compute the covariance matrix
    chol = pm.expand_packed_triangular(Nd, packed_chol, lower=True)
    cov = tt.dot(chol, chol.T)
    
    # Extract the standard deviations etc
    sd = pm.Deterministic('sd',tt.sqrt(tt.diag(cov)))
    lam = pm.Deterministic('lambda', (tt.diag(cov))**-1)
    corr = tt.diag(sd**-1).dot(cov.dot(tt.diag(sd**-1)))
    r = pm.Deterministic('r', corr[np.triu_indices(Nd, k=1)])
    
    lat_factor = pm.MvNormal('latent', mu=mu, chol=chol, observed=latent)
    trace_lat = pm.sample(2000, njobs=4)
#%%
pm.traceplot(trace_lat,
             varnames=['r','lambda'],
             lines={'r':[r12,r13,r23],
                    'lambda':[lambda1,lambda2,lambda3]})
#pm.plot_posterior(trace_lat,varnames=['r','lambda'])
plt.show()
#%%
accept = trace_lat.get_sampler_stats('mean_tree_accept')
print('The accept rate is: %.5f' % (accept.mean()))
diverge = trace_lat.get_sampler_stats('diverging')
print('Effective samples')
print(pm.diagnostics.effective_n(trace_lat))
print('Diverge of the trace')
print(diverge.nonzero())
energy = trace_lat['energy']
energy_diff = np.diff(energy)
sns.distplot(energy - energy.mean(), label='energy')
sns.distplot(energy_diff, label='energy diff')
plt.legend()
plt.show()
#%% PyMC3 model - from latent to observation
import theano
factor1 = theano.shared(latent[:,0])
with pm.Model() as model_obs1:
    # observation 1
    mu_ob1 = pm.Normal('mu_ob1', mu=0, sd=1, shape=obs1.shape[1])
    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist1 = pm.HalfCauchy.dist(beta=2.5)
    chol_cov1 = pm.LKJCholeskyCov('chol_cov1', n=Nd1, eta=2, sd_dist=sd_dist1)
    # compute the covariance matrix
    cholosb1 = pm.expand_packed_triangular(Nd1, chol_cov1, lower=True)
    cov_ob1 = pm.Deterministic('cov_ob1', tt.dot(cholosb1, cholosb1.T))
    
    mu_1 = tt.transpose(tt.outer(mu_ob1, factor1))
    obs_rv1 = pm.MvNormal('obs_rv1', mu=mu_1, cov=cov_ob1,
                          observed=obs1)
    trace_obs1 = pm.sample(2000, njobs=2)

burnin=0
pm.traceplot(trace_obs1[burnin:],varnames=['mu_ob1'])
pm.plot_posterior(trace_obs1[burnin:],varnames=['mu_ob1'])
plt.show()
#%% PyMC3 model - on observation
with pm.Model() as model_obs:
    mu = pm.Normal('mu', mu=0, sd=1, shape=Nd)

    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_cov', n=Nd, eta=2, sd_dist=sd_dist)
    # compute the covariance matrix
    chol = pm.expand_packed_triangular(Nd, packed_chol, lower=True)
    cov = tt.dot(chol, chol.T)
    
    # Extract the standard deviations etc
    sd = pm.Deterministic('sd',tt.sqrt(tt.diag(cov)))
    lam = pm.Deterministic('lambda', (tt.diag(cov))**-1)
    corr = tt.diag(sd**-1).dot(cov.dot(tt.diag(sd**-1)))
    r = pm.Deterministic('r',corr[np.triu_indices(Nd, k=1)])
    
#    v = pm.Normal('v', mu=0.0, sd=1.0, shape=(Nd,N))
#    lat_factor = pm.Deterministic('latent', 
#                                  tt.transpose(tt.outer(mu,tt.ones(N)) + tt.dot(chol, v)))
    lat_factor = pm.MvNormal('latent', mu=mu, cov=cov, observed=latent)
    
    # hyper parameters for observations
    # mu_hyper = pm.HalfCauchy('mu_hyper', beta=5)
    
    # observation 1
    mu_ob1 = pm.Normal('mu_ob1', mu=0, sd=10, shape=obs1.shape[1])
    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist1 = pm.HalfCauchy.dist(beta=2.5)
    chol_cov1 = pm.LKJCholeskyCov('chol_cov1', n=Nd1, eta=2, sd_dist=sd_dist1)
    # compute the covariance matrix
    cholosb1 = pm.expand_packed_triangular(Nd1, chol_cov1, lower=True)
    cov_ob1 = pm.Deterministic('cov_ob1', tt.dot(cholosb1, cholosb1.T))
    
    mu_1 = tt.transpose(tt.outer(mu_ob1, lat_factor[:,0]))
    obs_rv1 = pm.MvNormal('obs_rv1', mu=mu_1, cov=cov_ob1,
                          observed=obs1)
    
    # observation 2
    mu_ob2 = pm.Normal('mu_ob2', mu=0, sd=10, shape=obs2.shape[1])
    sigma2 = pm.Gamma('sigma2', alpha=1, beta=1, shape=obs2.shape[1])
    mu_2 = tt.transpose(tt.outer(mu_ob2, lat_factor[:,1]))
    obs_rv2 = pm.Normal('obs_rv2', mu=mu_2, sd=sigma2,
                          observed=obs2)
    
    # observation 3
    mu_ob3 = pm.Normal('mu_ob3', mu=0, sd=10, shape=obs3.shape[1])
    sigma3 = pm.Gamma('sigma3', alpha=1, beta=1, shape=obs3.shape[1])
    mu_3 = tt.transpose(tt.outer(mu_ob3, lat_factor[:,2]))
    obs_rv3 = pm.Normal('obs_rv3', mu=mu_3, sd=sigma3,
                          observed=obs3)
#%% PyMC3 model - on observation
with pm.Model() as model_obs:
    mu = pm.Normal('mu', mu=0, sd=1, shape=Nd)

    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_cov', n=Nd, eta=2, sd_dist=sd_dist)
    # compute the covariance matrix
    chol = pm.expand_packed_triangular(Nd, packed_chol, lower=True)
    cov = tt.dot(chol, chol.T)
    
    # Extract the standard deviations etc
    sd = pm.Deterministic('sd',tt.sqrt(tt.diag(cov)))
    lam = pm.Deterministic('lambda', (tt.diag(cov))**-1)
    corr = tt.diag(sd**-1).dot(cov.dot(tt.diag(sd**-1)))
    r = pm.Deterministic('r',corr[np.triu_indices(Nd, k=1)])
    
    lat_factor = pm.MvNormal('latent', mu=mu, chol=chol, shape=(N,Nd))
    
    # hyper parameters for observations
    mu_hyper = pm.HalfNormal('mu_hyper', sd=5)
    
    # observation 1
    mu_ob1 = pm.Normal('mu_ob1', mu=mu_hyper, sd=100, shape=obs1.shape[1])
    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    sd_dist1 = pm.HalfCauchy.dist(beta=2.5)
    chol_cov1 = pm.LKJCholeskyCov('chol_cov1', n=Nd1, eta=2, sd_dist=sd_dist1)
    # compute the covariance matrix
    cholosb1 = pm.expand_packed_triangular(Nd1, chol_cov1, lower=True)
    cov_ob1 = pm.Deterministic('cov_ob1', tt.dot(cholosb1, cholosb1.T))
    
    mu_1 = tt.transpose(tt.outer(mu_ob1, lat_factor[:,0]))
    obs_rv1 = pm.MvNormal('obs_rv1', mu=mu_1, chol=cholosb1,
                          observed=obs1)
    
    # observation 2
    mu_ob2 = pm.Normal('mu_ob2', mu=mu_hyper, sd=100, shape=obs2.shape[1])
    sigma2 = pm.Gamma('sigma2', alpha=1, beta=1, shape=obs2.shape[1])
    mu_2 = tt.transpose(tt.outer(mu_ob2, lat_factor[:,1]))
    obs_rv2 = pm.Normal('obs_rv2', mu=mu_2, sd=sigma2,
                          observed=obs2)
    
    # observation 3
    mu_ob3 = pm.Normal('mu_ob3', mu=mu_hyper, sd=100, shape=obs3.shape[1])
    sigma3 = pm.Gamma('sigma3', alpha=1, beta=1, shape=obs3.shape[1])
    mu_3 = tt.transpose(tt.outer(mu_ob3, lat_factor[:,2]))
    obs_rv3 = pm.Normal('obs_rv3', mu=mu_3, sd=sigma3,
                          observed=obs3)
#%%
with model_obs:
    trace_obs = pm.sample(3000, njobs=2, tune=1000)
#%%
import theano
with model_obs:
    # ADVI
    s = theano.shared(pm.floatX(1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    # ADVI has nearly converged
    pm.fit(n=20000, method=inference)
    # It is time to set `s` to zero
    s.set_value(0)
    approx = inference.fit(n=10000)
    trace_obs = approx.sample(2000)
    
    elbos1 = -inference.hist
#%%
burnin=0
pm.traceplot(trace_obs,
             varnames=['r','lambda','mu_hyper'],
             lines={'r':[r12,r13,r23],
                    'lambda':[lambda1,lambda2,lambda3]})
plt.show()
#%%
pm.traceplot(trace_obs[burnin:],
             varnames=['mu_ob1','mu_ob2','mu_ob3',
                                 'sigma2','sigma3'],
             lines={'mu_ob1':mu1, 'mu_ob2':mu2, 'mu_ob3':mu3})
#pm.plot_posterior(trace_obs[burnin:],varnames=['mu_ob1','mu_ob2','mu_ob3',
#                                 'sigma2','sigma3'])
plt.show()