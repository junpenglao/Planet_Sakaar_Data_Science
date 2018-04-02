######## I rewrote ADVI for logistic regression. It is more flexible than rstan ######################
# Elbo and gradient
Entropy=function(w){
  return(1+log(2*pi)+sum(w) )   ## the entropy of standard Gaussian , only depending on w= log(sigma)
}
elbo_wo_entropy=function(eta, mu, w,x, y){   # calculate the log likelihood of a standard gaussian variable: eta,
  sigma=exp(w)
  zeta=sigma*(eta)+mu   # the transformed parameter is a scaled mean_filed gaussian, with mean mu and sd sigma. It is the approximation of beta
  p=invlogit( x %*% zeta )
  return (sum(   log(p)*y+log(1-p)*(1-y)) )   # likelihood of logit regression
}

lppd=function(n_sample,mu, w,x,y)    # log-point-wise-predicitve-density using the normal approximation. used for out of sample prediction accuracy evaluation.
{
  elbo_sample=c()
  for(sample_index in 1:n_sample)
    elbo_sample[sample_index]=elbo_wo_entropy(eta=rnorm(K,0,1), mu, w,x,y)
  return (mean(elbo_sample) )
}

elbo=function(n_sample,mu, w,x,y ){  ## sample  n_elbo standard gaussian variables to evaluate the elbo, as a stochastic approximation
  elbo_sample=rep(NA, n_sample)
  for(sample_index in 1:n_sample)
    elbo_sample[sample_index]=elbo_wo_entropy(eta=rnorm(K,0,1), mu, w,x, y)
  return (mean(elbo_sample)+Entropy(w) )
}



elbo_grad=function(eta,mu,w,x,y){ #the gradient of objective function, evaluated by  a one-sample stochastic approximation -- each time sample 1 standard normal eta.
  grad_mu=grad_w=rep(NA,K)
  sigma=exp(w)
  zeta=sigma*(eta)+mu
  p=invlogit(x%*%zeta)
  grad_mu=t(x)%*%(y-p)
  grad_w=sigma*eta*grad_mu+1
  return(c(grad_mu,grad_w))
}
# test if it works: elbo_grad(eta=c(0,0), mu=c(1,1),w=c(0,0),x,y)

#### 1.  initialize: choose a better scale_factor, skip if you want pick it manually
scale_factor_vec=c(0.01,0.1,1,10,100)
scale_factor_warm_up=function(scale_factor_vec=scale_factor_vec, iter_warmup=50, batch_size=10, x, y){
  sample_data_index=sample(1:n,batch_size)     ## sample a mini-batch
  elbo_scale_compare=rep(NA,length(scale_factor_vec))
  for(j in  1:length(scale_factor_vec)){
    scale_factor=scale_factor_vec[j]
    iter=iter_warmup
    mu_iter=matrix(NA, iter+1, K)
    w_iter=matrix(NA, iter+1, K)
    mu_iter[1,]=rep(0, K)
    w_iter[1,]=rep(1, K)
    rho=matrix(NA,iter, 2*K)   #leanring rate. 2* the number of parameters
    s=matrix(NA,iter, 2*K)
    for(i in 1:iter)
    {
      eta=rep(NA,K)
      eta=rnorm(K, 0,1)
      mu=mu_iter[i,]
      w=w_iter[i,]
      grad_vec=elbo_grad(eta, mu, w, x, y)*n/batch_size
      if(i==1) s[i,]= 0.1* grad_vec^2 else
        s[i,]=0.1* grad_vec^2+ (0.9)* s[i-1,]
      rho[i,]=scale_factor*i^(-1/2+1e-16)/(1+sqrt(s[i,]))
      mu=mu+rho[i,1:K]*grad_vec[1:K ]
      w=w+rho[i,(1:K)+K]*grad_vec[(1:K)+K]
      mu_iter[i+1,]=mu
      w_iter[i+1,]=w
    }
    elbo_scale_compare[j]=elbo(1000,mu,w,x,y)
  }
  scale_factor=scale_factor_vec[which.max(elbo_scale_compare)]
  return(scale_factor)
}
### 2.


posterior_sample=function (mu, w, S=1000, random_seed=NULL)
{
  if(!is.null(random_seed))
    set.seed(random_seed)
  beta=matrix(NA, S, length(mu) )
  for( i in 1:length(mu)){
    beta[,i]=rnorm (S, mean=mu[i], sd=exp(w[i]))
  }
  return( beta )  #posterior beta: S*K matrix
}
#beta_sample_vi=posterior_sample(mu=mu, w=w, S=S)

log_proposal_lik_one_sample=function(beta_one_sample, mu,w ){
  return(sum (dnorm(beta_one_sample,mean=mu,sd=exp(w),log = TRUE)))
}
log_proposal_density=function(beta_sample_vi, mu, w){   ## calculate the proposal density
  return(apply(beta_sample_vi, 1, log_proposal_lik_one_sample, mu=mu, w=w)   )
}
# calculate the model log_density

log_density_target=function(beta_one_sample,x,y){
  p=invlogit( x %*% beta_one_sample )
  return(sum( log(p)*y+log(1-p)*(1-y)))
} ##  the posterior density : likelihood*prior


log_importance_ratio=function(mu, w, S=1000, x=x, y=y,vi_sample=NULL){
  if(is.null(vi_sample))
    vi_sample=posterior_sample(mu=mu, w=w, S=S)
  if(nrow(vi_sample)!=S )
    print("warning: mis-specify the length of samples")
  log_proposal=log_proposal_density(vi_sample, mu=mu, w=w)
  log_target=apply(vi_sample,1,log_density_target, x=x, y=y)
  return(log_target-log_proposal)
}


ip_weighted_average=function(lw, x){
  ip_weights=exp(lw-min(lw))
  return(  t(ip_weights)%*%x /sum(ip_weights) )
}

#
# mu= advi_sample1$mu
# w= advi_sample1$w
#
# ip_ratio=log_importance_ratio(mu=mu, w=w, S=10000, x=x, y=y)
# print(psislw(lw=ip_ratio)$pareto_k)






advi=function(iter= 10000,  iter_skip=200, scale_factor=NULL, x=x, y=y,K=K,  n_elbo_sample=100,is.elbo=TRUE ,is.test=FALSE, is.k_hat=TRUE,S=1000,silent=FALSE){
  if (is.null(scale_factor))
    scale_factor=scale_factor_warm_up(iter_warmup=50, scale_factor_vec=c(0.01,0.1,1,10,100),   batch_size=10, x=x, y=y)
  mu_iter=matrix(NA, iter+1, K)
  w_iter=matrix(NA, iter+1, K)
  k_iter=rep(NA, iter)
  first_moment_estimation=array(NA, c(iter, 3,K)  )   ## three methods: proposal, is, psis
  second_moment_estimation=array(NA, c(iter,3, K)  )   ## three methods: proposal, is, psis

  mu_iter[1,]=c(0,0)   #initialize from standard gaussian
  w_iter[1,]=c(1,1)
  elbo_iter=rep(NA, iter)
  rho=matrix(NA,iter, 2*K)
  s=matrix(NA,iter, 2*K)
  test_lpd=rep(NA, iter)
  elbo_iter_MC=matrix(NA, 2*K,iter)
  for(i in 1:iter)
  {
    eta=rep(NA,K)
    eta=rnorm(K, 0,1)
    mu=mu_iter[i,]
    w=w_iter[i,]
    grad_vec=elbo_grad(eta, mu, w, x, y)
    if(i==1) s[i,]= 0.1* grad_vec^2 else
      s[i,]=0.1* grad_vec^2+ (0.9)* s[i-1,]
    rho[i,]=scale_factor*i^(-1/2+1e-16)/(1+sqrt(s[i,]))
    mu=mu+rho[i,1:K]*grad_vec[1:K ]
    w=w+rho[i,(1:K)+K]*grad_vec[(1:K)+K]
    mu_iter[i+1,]=mu
    w_iter[i+1,]=w

    if(is.elbo==TRUE)
      if(i <=10 | i %% iter_skip ==0 | i %% iter_skip ==1  ){
        elbo_iter[i]=elbo(n_sample=n_elbo_sample,mu=mu, w=w,x=x,y=y )
      }
    if(i <=10 | i %% iter_skip ==0  ){
      if(is.test==TRUE)
        test_lpd[i]=lppd(n_sample=n_elbo_sample, mu=mu, w=w,x=x_test,y=y_test )/ length(y_test)
      if(is.k_hat==TRUE)
      {
        vi_sample=posterior_sample(mu=mu, w=w, S=S)
        ip_ratio=log_importance_ratio(mu=mu, w=w, S=S, x=x, y=y,vi_sample=vi_sample)
        vi_sample=vi_sample[complete.cases(ip_ratio),] ## remove NAs, (when the proposal is too far away)
        ip_ratio=ip_ratio[complete.cases(ip_ratio)]
        options(warn=-1) ## You don't want receive ten million warning during the loop.
        psis_estimation=psislw(lw=ip_ratio)
        k_iter[i]=psis_estimation$pareto_k
        psis_lw=psis_estimation$lw_smooth
        first_moment_estimation[i,1,]=mu
        second_moment_estimation[i,1,]=mu^2+ (exp(w))^2
        first_moment_estimation[i,2,]=ip_weighted_average(lw=ip_ratio, x=vi_sample)
        second_moment_estimation[i,2,]=ip_weighted_average(lw=ip_ratio, x=vi_sample^2)
        first_moment_estimation[i,3,]=ip_weighted_average(lw=psis_lw, x=vi_sample)
        second_moment_estimation[i,3,]=ip_weighted_average(lw=psis_lw, x=vi_sample^2)
      }
    }
    if(silent==FALSE){
      if(i %% 1000 ==0){
        cat( paste ("iter:", i) )
        cat("\n")
      }
    }
  }
  return(list(mu=mu, w=w, mu_iter=mu_iter,w_iter=w_iter,elbo_iter=elbo_iter,test_lpd=test_lpd,k_iter=k_iter, first_moment_estimation=first_moment_estimation,  second_moment_estimation=second_moment_estimation ))
}
