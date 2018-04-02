#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:03:37 2017

@author: laoj
"""
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats
#from sampled import sampled
import seaborn as sns
import theano.tensor as tt
import theano
from pymc3.distributions.dist_math import bound
#%%
def tt_donut_pdf(scale):
    """Compare to `donut_pdf`"""
    def logp(x):
         return -tt.square((1 - x.norm(2)) / scale)
    return logp

with pm.Model():
    """Gets samples from the donut pdf, and allows adjusting the scale of the donut at sample time."""
    pm.DensityDist('donut', logp=tt_donut_pdf(.01), shape=2, testval=[0, 1])
    tr = pm.sample(init=None)
#%%
def w1(z):
    return tt.sin(2.*np.pi*z[0]/4.)
def w2(z):
    return 3.*tt.exp(-.5*(((z[0]-1.)/.6))**2)
def w3(z):
    return 3.*(1+tt.exp(-(z[0]-1.)/.3))**-1
def pot1(z):
    z = z.T
    return .5*((z.norm(2, axis=0)-2.)/.4)**2 - tt.log(tt.exp(-.5*((z[0]-2.)/.6)**2) + tt.exp(-.5*((z[0]+2.)/.6)**2))
def pot2(z):
    z = z.T
    return .5*((z[1]-w1(z))/.4)**2 + 0.1*tt.abs_(z[0])
def pot3(z):
    z = z.T
    return -tt.log(tt.exp(-.5*((z[1]-w1(z))/.35)**2) + tt.exp(-.5*((z[1]-w1(z)+w2(z))/.35)**2)) + 0.1*tt.abs_(z[0])
def pot4(z):
    z = z.T
    return -tt.log(tt.exp(-.5*((z[1]-w1(z))/.4)**2) + tt.exp(-.5*((z[1]-w1(z)+w3(z))/.35)**2)) + 0.1*tt.abs_(z[0])

z = tt.matrix('z')
z.tag.test_value = pm.floatX([[0., 0.]])
pot1f = theano.function([z], pot1(z))
pot2f = theano.function([z], pot2(z))
pot3f = theano.function([z], pot3(z))
pot4f = theano.function([z], pot4(z))

def contour_pot(potf, ax=None, title=None, xlim=5, ylim=5):
    grid = pm.floatX(np.mgrid[-xlim:xlim:100j,-ylim:ylim:100j])
    grid_2d = grid.reshape(2, -1).T
    cmap = plt.get_cmap('inferno')
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 9))
    pdf1e = np.exp(-potf(grid_2d))
    contour = ax.contourf(grid[0], grid[1], pdf1e.reshape(100, 100), cmap=cmap)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return ax

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
contour_pot(pot1f, ax[0], 'pot1');
contour_pot(pot2f, ax[1], 'pot2');
contour_pot(pot3f, ax[2], 'pot3');
contour_pot(pot4f, ax[3], 'pot4');
fig.tight_layout()
#%%
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
z = tt.matrix('z')
z.tag.test_value = pm.floatX([[0., 0.]])
pot1f = theano.function([z], pot1(z))
contour_pot(pot1f, ax[0], 'pot1');
#%%
def cust_func(z):
    return bound(-pot1(z), z>-5, z<5)
    
with pm.Model():
    """Gets samples from the donut pdf, and allows adjusting the scale of the donut at sample time."""
    pm.DensityDist('donut', logp=cust_func, shape=(2,), testval=[0, 2])
    tr = pm.sample(10000, init=None, tune=0, start=dict(pot1=np.array([2, 0])))
    
plt.plot(tr['donut'][:,0],tr['donut'][:,1],'o');
#%%
def cust_logp(z):
    return -pot3(z)
    #return bound(-pot4(z), z>-5, z<5)

with pm.Model() as pot3m:
    pm.DensityDist('pot_func', logp=cust_logp, shape=(2,))
    
with pot3m:
    traceNUTS = pm.sample(2500, init=None, njobs=2)
    
formula = 'planar*16'
with pot3m:
    inference = pm.NFVI(formula, jitter=1.)
inference.fit(25000, obj_optimizer=pm.adam(learning_rate=.01), obj_n_mc=200)
traceNF = inference.approx.sample(5000)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
contour_pot(pot3f, ax[0], 'pot3');
ax[1].scatter(traceNUTS['pot_func'][:,0],traceNUTS['pot_func'][:,1],c='r',alpha=.02);
ax[1].set_xlim(-5,5)
ax[1].set_ylim(-5,5)
ax[1].set_title('NUTS')
ax[2].scatter(traceNF['pot_func'][:,0],traceNF['pot_func'][:,1],c='b',alpha=.02);
ax[2].set_xlim(-5,5)
ax[2].set_ylim(-5,5)
ax[2].set_title('NF with '+formula)