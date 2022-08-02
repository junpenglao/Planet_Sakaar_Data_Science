import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pymc3 as pm

import theano
import theano.tensor as tt

#load data
train_df = pd.read_csv('redwoods.csv')
#scale it so it is a bit easier to work with 
xloc = train_df['redwoodfull.x']*100
yloc = train_df['redwoodfull.y']*100

#discretize spatial data
D = 2 #dimension
num_bins = 64
hist, xedges, yedges = np.histogram2d(yloc, xloc, 
                                      bins=np.linspace(0,100,num_bins+1))
xcenters = xedges[:-1] + np.diff(xedges)/2
ycenters = yedges[:-1] + np.diff(yedges)/2

f, ax = plt.subplots(1, 2, figsize=(8, 4),)

lengthscale_ = 7
sns.kdeplot(xloc,yloc,
            bw=lengthscale_,
            cmap="viridis", shade=True, ax=ax[0])
ax[0].scatter(xloc, yloc, color='r', alpha=.25)
ax[0].set_xlim(0,100)
ax[0].set_ylim(0,100)

ax[1].imshow(hist,
             cmap='viridis', origin='lower')
ax[1].axis('off')
#%%
#input/output
xv, yv = np.meshgrid(xcenters, ycenters)
x_data = np.vstack((yv.flatten(),xv.flatten())).T
y_data = hist.flatten()
#%% pymc3 minibatch setup 
# Not suitable for 2D mapping problem, overestimated lengthscale
batchsize = 10
Xbatch = pm.Minibatch(x_data, batchsize**2)
Ybatch = pm.Minibatch(y_data, batchsize**2)
#%% set up minibatch
data = hist
batchsize = 10
z1, z2 = batchsize, batchsize
s1, s2 = np.shape(data)
yshared = theano.shared(data)
x1shared = theano.shared(ycenters[:,np.newaxis].repeat(64,axis=1))
x2shared = theano.shared(xcenters[:,np.newaxis].T.repeat(64,axis=0))

ixs1 = pm.tt_rng().uniform(size=(1,), low=0, high=s1-z1-1e-10).astype('int64')
ixs2 = pm.tt_rng().uniform(size=(1,), low=0, high=s2-z2-1e-10).astype('int64')
range1 = tt.arange(ixs1.squeeze(),(ixs1+z1).squeeze())
range2 = tt.arange(ixs2.squeeze(),(ixs2+z2).squeeze())
Ybatch = yshared[range1][:,range2].flatten()
Xbatch1 = x1shared[range1][:,range2].flatten()
Xbatch2 = x2shared[range1][:,range2].flatten()
Xbatch = tt.stack((Xbatch1,Xbatch2)).T

import theano
theano.config.compute_test_value = 'off'
#%%
with pm.Model() as model:
    #hyper-parameter priors
    # weakly informative prior
#    l = pm.HalfCauchy('l', beta=3.)
    # informative prior
    l = pm.Gamma('l', alpha=5, beta=1, 
                 transform=pm.distributions.transforms.LogExpM1())
    
    eta = pm.HalfCauchy('eta', beta=3.)
    cov_func = eta**2 * pm.gp.cov.Matern32(D, ls=l*np.ones(D))
    
    #Gaussian Process
    gp = pm.gp.Latent(cov_func=cov_func)
    f = gp.prior('f', X=Xbatch, shape=batchsize**2)
    
    obs = pm.Poisson('obs', mu=tt.exp(f), observed=Ybatch, total_size=y_data.shape)

    approx = pm.fit(20000,
                    method='fullrank_advi',
                    callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    trace = approx.sample(1000)
pm.traceplot(trace, varnames=['l','eta']);
#%%
with model:
    group_1 = pm.Group([l,eta], vfam='fr')  # latent1 has full rank approximation
    group_other = pm.Group(None, vfam='mf')  # other variables have mean field Q
    approx = pm.Approximation([group_1, group_other])
    pm.KLqp(approx).fit(100000,
                    callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    trace = approx.sample(1000)
#%% prediction
#nx=50
#x = np.linspace(0, 100, nx)
#y = np.linspace(0, 100, nx)
#xv, yv = np.meshgrid(x, y)
#x_pred = np.vstack((yv.flatten(),xv.flatten())).T

# add the GP conditional to the model, given the new X values
with pm.Model() as predi_model:
    #hyper-parameter priors
    l = pm.HalfNormal('l', sd=.1)
    eta = pm.HalfCauchy('eta', beta=3.)
    cov_func = eta**2 * pm.gp.cov.Matern32(D, ls=l*np.ones(D))
    
    #Gaussian Process
    gp = pm.gp.Latent(cov_func=cov_func)
    f = gp.prior('f1', X=x_data)
    
    obs = pm.Poisson('obs1', mu=tt.exp(f), observed=y_data)
    f_pred = gp.conditional('f_pred', x_data)
    # Sample from the GP conditional distribution
    pred_samples = pm.sample_ppc(trace, vars=[f_pred], samples=100)
ftrace = np.mean(pred_samples['f_pred'], axis=0)
ftrace = np.reshape(ftrace, (num_bins, num_bins))
latent_rate = np.exp(ftrace)
#%%
f, ax = plt.subplots(1, 3, figsize=(12, 4), )
sns.kdeplot(xloc,yloc,
            bw=.3,
            cmap="viridis", shade=True, ax=ax[0])
ax[0].scatter(xloc, yloc, color='r', alpha=.25)
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)

ax[1].imshow(hist,
             cmap='viridis', origin='lower')
ax[1].axis('off')
#%%
ax[2].imshow(latent_rate,
             cmap='viridis', origin='lower', interpolation='gaussian')
ax[2].scatter(xloc/2, yloc/2, color='r', alpha=.25)
ax[2].axis('off')
plt.tight_layout();
