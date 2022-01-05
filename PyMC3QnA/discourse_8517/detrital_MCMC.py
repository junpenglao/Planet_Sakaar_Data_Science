# -detrital_MCMC.py *- coding: utf-8 -*-

#%% Import general packages

import os
import numpy as np
import scipy.io
import pymc3 as pm
from pymc3 import Model, Uniform, sample, Metropolis
import theano.tensor as tt
import arviz as az
from arviz import plot_trace

# RANDOM_SEED = 50
path = 'G:\Shared drives\Ben_Greg\Thesis\ZHe_data\Benjamin\Shilong Basin'
os.chdir(path)


#%% Import basin information --- Check proper working directory (were catchment data is)

dem = scipy.io.loadmat('Shillong_DEM.mat')   
Z = dem['Z']/1000 # convert meters to kilometers for consistency with RVs


#%% Import thermochronology data --- Check proper working directory (were detrital data is)

filename = 'Avdeev_detrital.txt'
thermo_file = np.loadtxt(filename, skiprows = 1, dtype = str)
thermo_age = thermo_file[:,0]
thermo_age = np.array(thermo_age).astype('float64')


#%% Define priors for model --- Requires installation of pymc3

with Model() as detrital_thermo:
    slope_break = Uniform('slope_break', lower = 0, upper = 60)
    erosion_0 = Uniform('erosion_0', lower = 0.01, upper = 1)
    erosion_1 = Uniform('erosion_1', lower = 0.01, upper = 1)
    closure_depth = Uniform('closure_depth', lower = -1, upper = 4)
    std_dev = Uniform('std_dev', lower = 0.05, upper = 0.5)
    #alpha = Uniform('alpha', lower = 0, upper = 1) # isotherm topo-deflection coeff
    closure_depth_1 = closure_depth + slope_break*(erosion_0 - erosion_1)
    

#%% Bedrock cooling age

with detrital_thermo:
    # changed minus to plus sign in age calculation; error in Avdeev's?
    b_0 = ((Z + closure_depth) / erosion_0)
    b_1 = ((Z + closure_depth_1) / erosion_1)
    b = tt.switch((b_0 < slope_break), b_0, b_1) 
    
    basin_size = tt.sum(~tt.isnan(b))

    age = tt.switch(tt.isnan(b), 0, b)
    sigma_age = age * std_dev
     
    
#%% Define p_D as a mixture of normal distributions ['manually' is faster]
# See https://online.stat.psu.edu/stat414/lesson/26/26.1


#%% PDF of detrital sample -- Version 1: Manual Normal Mixture (faster)

with detrital_thermo:
       
    w = 1/basin_size
    
    mean = tt.sum(age) * w
    std = tt.sqrt(tt.sum(tt.sqr(sigma_age * w)))  # what about covariance? 
        
    p_D = pm.Normal('p_D', mu = mean, sd = std, observed = thermo_age)


#%% PDF of detrital sample -- Version 2: PyMC3 Normal Mixture (slower)
'''
with detrital_thermo:

    weigths = np.ones(Z.shape)/basin_size

    p_D = pm.NormalMixture('p_D', w = tt.flatten(weigths), mu = tt.flatten(age), 
                           sigma = tt.flatten(sigma_age), 
                           observed = thermo_age)
'''       

#%% Check model definitions

pm.model_to_graphviz(detrital_thermo)

    
#%% Sampling the posterior

with detrital_thermo:
    
    trace = sample(draws = 5000, tune = 1000, cores = 1, step = Metropolis(),
                   chains = 1, return_inferencedata = True)


#%% Posterior outputs

with detrital_thermo:
    az.summary(trace, round_to=2)
    #plot_trace(trace, var_names = "slope_break")
    plot_trace(trace)

#%% Avdeev's results for Shillong Basin

slope_break = 9.00
closure_depth = -0.50
erosion_0 = 0.23
erosion_1 = 0.03
std_dev = 0.30
