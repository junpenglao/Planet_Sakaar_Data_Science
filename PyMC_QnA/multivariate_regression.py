import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

multivariate_data = pd.read_csv("fake_data_multivariate_regression.csv")

x_vars  = ["x0","x1","x2","x3","x4","x5"]
y_vars  = ["y1","y2","y3","y4"]

num_obs = len(multivariate_data)

n_xvar = len(x_vars)

y_data  = multivariate_data[y_vars].values
x_data  = multivariate_data[x_vars].values

from sklearn import preprocessing

scaler_x = preprocessing.StandardScaler()
x_data[:,1:] = scaler_x.fit_transform(x_data[:,1:])

scaler_y = preprocessing.StandardScaler()
y_data = scaler_y.fit_transform(y_data)

dim = len(y_vars)

# Used to extract the correlation terms
cor_term_indices = np.triu_indices(dim,k=1)

with pm.Model() as multivariate_regression:
    # Uninformed priors for the covariance matrix using Cholesky factors
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_fact', n=dim, eta=1, sd_dist=sd_dist)
    
    # Transform the packed cholesky factor (array) into an unpacked factor (matrix)
    chol = pm.expand_packed_triangular(dim, packed_chol, lower=True)
    
    # Generate covariance matrix from Cholesky factor
    cov_mtx = tt.dot(chol, chol.T)

    # Generate the standard deviations from the cholesky factor
    sd  = pm.Deterministic('sd',tt.sqrt(tt.diag(cov_mtx)))

    # Tensor manipulation used to extract the correlation terms
    cor_mtx = tt.diag(sd**-1).dot(cov_mtx).dot(tt.diag(sd**-1))
    cor_terms  = pm.Deterministic('cor_terms',cor_mtx[cor_term_indices])
    
    # Uninformed priors on the betas
    betas = pm.Normal('betas', mu=0.0, sd=10, shape=(n_xvar,dim))
    
    # Creating the center of the multivariate distribution for each observation
    y = pm.MvNormal('y', mu=tt.dot(x_data, betas), chol=chol, observed=y_data)
    
    trace = pm.sample(1000, njobs=4)
#%%