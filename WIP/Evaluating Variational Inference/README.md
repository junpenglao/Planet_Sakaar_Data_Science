# R code for the manuscript "Yes, but Did It Work: Evaluating Variational Inference"
The paper  will be submitted to to ICML 2018.

While it’s always possible to compute a variational approximation to a posterior distribution,it’s difficult to work out when this approximation is good. We propose two diagnostic algorithms to alleviate this problem. The Pareto-smoothed importance sampling (PSIS) diagnostic gives a goodness of fit measurement for joint distributions through the shape parameter in the fitted distribution of density ratios. It shrinks the variational estimation error at the same time. The variational simulation-based calibration (VSBC) assesses the average performance of point estimates.

The R_code folder contains source code for Figure 1-8 in the paper, where we experimentally illustrates the benefits of proposed diagnoistics. 
