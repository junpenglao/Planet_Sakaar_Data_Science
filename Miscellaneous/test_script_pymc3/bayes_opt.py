#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 10:32:46 2017

@author: jlao
"""
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy.ma import masked_values
from skopt import gp_minimize
import theano
from scipy import optimize
#%%
# Import data, filling missing values with sentinels (-999)
test_scores = pd.read_csv(pm.get_data('test_scores.csv')).fillna(-999)

# Extract variables: test score, gender, number of siblings, previous disability, age,
# mother with HS education or better, hearing loss identified by 3 months
# of age
(score, male, siblings, disability,
    age, mother_hs, early_ident) = test_scores[['score', 'male', 'siblings',
                                                'prev_disab', 'age_test',
                                                'mother_hs', 'early_ident']].astype(float).values.T

with pm.Model() as model:
    # Impute missing values
    sib_mean = pm.Exponential('sib_mean', 1.)
    siblings_imp = pm.Poisson('siblings_imp', sib_mean,
                              observed=siblings)

    p_disab = pm.Beta('p_disab', 1., 1.)
    disability_imp = pm.Bernoulli(
        'disability_imp', p_disab, observed=masked_values(disability, value=-999))

    p_mother = pm.Beta('p_mother', 1., 1.)
    mother_imp = pm.Bernoulli('mother_imp', p_mother,
                              observed=masked_values(mother_hs, value=-999))

    s = pm.HalfCauchy('s', 5., testval=5)
    beta = pm.Laplace('beta', 0., 100., shape=7, testval=.1)

    expected_score = (beta[0] + beta[1] * male + beta[2] * siblings_imp + beta[3] * disability_imp +
                      beta[4] * age + beta[5] * mother_imp + beta[6] * early_ident)

    observed_score = pm.Normal(
        'observed_score', expected_score, s, observed=score)

#%%
with model:
    start1 = pm.find_MAP()
    
#%%
with pm.Model() as model:
    p = pm.Beta('p', 1., 1.)
    obs = pm.Binomial('obs', p=p, n=5, observed=2)
#%%
with pm.Model() as model:
    mu = pm.Normal('m', 0., 100.)
    sd = pm.HalfCauchy('sd', 5.)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=np.random.randn(100)+5.)
    inference = pm.ADVI()

testval = np.concatenate([inference.approx.shared_params['mu'].get_value(), 
                          inference.approx.shared_params['rho'].get_value()])
    
NEG_ELBO = theano.function([], inference.objective(5000))
def objective(point):
    mu = point[:len(point)//2]
    rho = point[len(point)//2:]
    inference.approx.shared_params['mu'].set_value(mu)
    inference.approx.shared_params['rho'].set_value(rho)
    return np.asscalar(NEG_ELBO())

r1=optimize.fmin_powell(objective, testval)

inference2 = pm.NFVI(flow='scale-loc', model=model)# pm.ADVI(model=model)
inference2.fit(n=50000)
best = np.concatenate([inference2.approx.shared_params[0]['loc'].get_value(), 
                       inference2.approx.shared_params[1]['log_scale'].get_value()])
#%%
r2=gp_minimize(objective, [(0,10), (-1,1), (-5,5), (-5,5)])
#%%
import theano
localrv = inference.objective.obj_params
munew = theano.shared(np.ones(localrv[0].eval().shape))
rhonew = theano.shared(np.ones(localrv[1].eval().shape))
pointfunc = theano.clone(inference.objective(nmc=10), 
                         replace = {localrv[0]:munew, localrv[1]:rhonew})
#%%
from pymc3.blocking import DictToArrayBijection, ArrayOrdering
from pymc3.model import modelcontext, Point
from pymc3.theanof import inputvars
from numpy import isfinite, nan_to_num, logical_not

start = model.test_point
vars = model.cont_vars
vars = inputvars(vars)

start = Point(start, model=model)
bij = DictToArrayBijection(ArrayOrdering(vars), start)

logp = bij.mapf(model.fastlogp)

def allfinite(x):
    return np.all(isfinite(x))

def nan_to_high(x):
    return np.where(isfinite(x), x, 1.0e100)

def logp_o(point):
    return np.asscalar(nan_to_high(-logp(point)))

r = optimize.fmin_bfgs(logp_o, bij.map(start))

#%%
res = gp_minimize(logp_o, dimensions=[(-2.0, 2.0)])
#%%
def gp_minimize2(func, *args, **kwargs):
    def func2(x):
        return np.asscalar(func(x))
    return gp_minimize(func2, dimensions=[(-100.0, 100.0) for i in 
                                          range(len(model.bijection.map(model.test_point)))], 
        *args, **kwargs)['x']

with model:
    start1 = pm.find_MAP()
    start2 = pm.find_MAP(fmin=gp_minimize2)
    #start3 = pm.fit()
#%%
def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

x=np.linspace(-1,1, 1000)
y = np.array([f([x1]) for x1 in x])
plt.figure()
plt.plot(x,y)
#%%
res = gp_minimize(f, dimensions=[(-2.0, 2.0)])

x=np.linspace(-1,1, 1000)
y = np.array([f([x1]) for x1 in x])
plt.plot(x,y, alpha=.1)
plt.scatter(res['x_iters'], res['func_vals'], color='r',marker='o')
plt.scatter(res['x'], res['fun'], color='k',marker='o')