### R code for CGR test in horseshoe logstic regression. I run it on cluster.
#### cluster settings. ignore it if you are on a PC.
.libPaths("")
args <- commandArgs(trailingOnly = TRUE)
print(Sys.getenv("SLURM_ARRAY_JOB_ID"))
arrayid <- as.integer(args[1])
set.seed(as.integer(arrayid))

library(rstan)
library(LaplacesDemon)
#library(invgamma)
options(mc.cores = parallel::detectCores())
setwd("")

load(file="horse_shoe_cgr.RData") ## compile stanmodel first, you can run stanmodel=stanmodel <- stan_model('glm_bernoulli_rhs.stan') instead
datafile <- 'leukemia.RData'
load(datafile,verbose=T)
 # load data
x <- scale(x)
d <- NCOL(x)
n <- NROW(x)

scale_icept=10
slab_scale=5
slab_df=4
tau0 <- 1/(d-1) * 2/sqrt(n) # should be a reasonable scale for tau (see Piironen&Vehtari 2017, EJS paper)
scale_global=tau0

I=20
prob=matrix(NA,I,213900)

for( i in 1:i){
  set.seed(i+arrayid*I*2)  # to make sure a new seed every time
  z = rnorm(d,0,1)
  lambda = rhalfcauchy(n=d,scale=25)
  tau = rhalfcauchy(n=1, scale=scale_global)
  caux = rinvgamma(1,shape=0.5*slab_df, scale=0.5*slab_df)
  beta0 = rnorm(1, 0,scale_icept)

  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * (lambda)^2 / (c^2 + tau^2* (lambda)^2) );
  beta = z * lambda_tilde*tau;
  f = beta0 + x%*%beta;
  y = rbinom(n,size=1,prob= invlogit(f))

  parameter0=c(beta0, beta, z,  log(tau),log(lambda),log(caux) )
data <- list(n=n, d=d, x=x, y=as.vector(y), scale_icept=10, scale_global=tau0,
               slab_scale=5, slab_df=4)

  fit_advi <- vb(stanmodel, data=data,iter=40000,output_samples=5000,tol_rel_obj=0.001,eta = 0.3, adapt_engaged=F)  #### it really takes a while.
  vi_sample=extract(fit_advi)
  rm(fit_advi)
  trans_parameter=cbind(vi_sample$beta0, vi_sample$beta, vi_sample$z, log(vi_sample$tau),log(vi_sample$lambda),log(vi_sample$caux) )
  rm(vi_sample)
  vi_parameter_mean=apply(trans_parameter, 2, mean)
  vi_parameter_sd=apply(trans_parameter, 2, sd)
  rm(trans_parameter)
  prob[i,]=pnorm( parameter0, vi_parameter_mean,vi_parameter_sd)
  rm(vi_parameter_mean,vi_parameter_sd)
}
save(prob,file=paste("arg_", arrayid, ".RData", sep=""))
rm(list=ls())

