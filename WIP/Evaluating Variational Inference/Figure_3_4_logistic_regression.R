############################################
##  Figure 3-4 #############################
####  Logistic regression ADVI #############
###### eventually fail when cor increases
library(arm)
library(rstan)
#setwd("")
source("functions.R")
## 1.  do exact sampling
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
stan_code='
data {
int<lower=0> n;
int<lower=0> K;
matrix[n, K] x;
int<lower=0,upper=1> y[n];
}
parameters {
vector[K] beta;
}
model {
y ~ bernoulli_logit(x*beta);
}
'

##### 2. modified ADVI  function ###########

advi=function(iter= 10000,  iter_skip=200, scale_factor=NULL, x=x, y=y,K=K,   S=1000 ){
  if (is.null(scale_factor))
    scale_factor=scale_factor_warm_up(iter_warmup=50, scale_factor_vec=c(0.01,0.1,1,10,100),   batch_size=10, x=x, y=y)
  mu_iter=matrix(NA, iter+1, K)
  w_iter=matrix(NA, iter+1, K)
  first_moment_estimation=array(NA, c( 3,K)  )   ## three methods: proposal, is, psis
  second_moment_estimation=array(NA, c( 3, K)  )   ## three methods: proposal, is, psis
  mu_iter[1,]=c(0,0)   #initialize from standard gaussian
  w_iter[1,]=c(1,1)
  rho=matrix(NA,iter, 2*K)
  s=matrix(NA,iter, 2*K)
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
  }

  final_elbo  = elbo(n_sample=n_elbo_sample,mu=mu, w=w,x=x,y=y )
  final_test_lpd=lppd(n_sample=n_elbo_sample, mu=mu, w=w,x=x_test,y=y_test )/ length(y_test)

  vi_sample=posterior_sample(mu=mu, w=w, S=S)
  ip_ratio=log_importance_ratio(mu=mu, w=w, S=S, x=x, y=y,vi_sample=vi_sample)
  vi_sample=vi_sample[complete.cases(ip_ratio),] ## remove NAs, (when the proposal is too far away)
  ip_ratio=ip_ratio[complete.cases(ip_ratio)]
  options(warn=-1) ## You don't want receive ten million warning during the loop.
  psis_estimation=psislw(lw=ip_ratio)
  k_final=psis_estimation$pareto_k
  psis_lw=psis_estimation$lw_smooth
  first_moment_estimation[1,]=mu
  second_moment_estimation[1,]=mu^2+ (exp(w))^2
  first_moment_estimation[2,]=ip_weighted_average(lw=ip_ratio, x=vi_sample)
  second_moment_estimation[2,]=ip_weighted_average(lw=ip_ratio, x=vi_sample^2)
  first_moment_estimation[3,]=ip_weighted_average(lw=psis_lw, x=vi_sample)
  second_moment_estimation[3,]=ip_weighted_average(lw=psis_lw, x=vi_sample^2)

  return(list(mu=mu, w=w, mu_iter=mu_iter,w_iter=w_iter,elbo_final=final_elbo,test_final=final_test_lpd,k_final=k_final, first_moment_estimation=first_moment_estimation,  second_moment_estimation=second_moment_estimation ))
}

log_density_target_mean=function(beta_one_sample,x,y){
  p=invlogit( x %*% beta_one_sample )
  return(mean( log(p)*y+log(1-p)*(1-y), na.rm = T   ))
} ##  the posterior density : likelihood*prior


#1.5 a large varience
cov_vector=c(0, 0.1, 0.2, 0.5,0.7, 0.8,0.9,1.3,1.4,1.6,1.7,1.9,2,2.3,2.41,2.437,2.5,2.6,3,4,8)
###  c(2.41,2.437,2.5) manucally picked up, for they lead to results around k=0.7
Rep_number=200
iter= 6000
J=length(cov_vector)
K=2
k_rep=elbo_rep=test_rep=matrix(NA,J ,Rep_number)
rmse_array=bias_array=std_array=array(NA,c(J,2,3,K))
cor_vec=c()
optimal_lpd=c()
for( j  in  1:J)
{
  cov_sd=  cov_vector[j]
  set.seed(1000)
  #  generate ground truth, a logistic regression with K parameters and no intercept
  b=rep(NA, K)
  n=100
  b[1]=1
  b[2]=2
  n=50
  n_test=100
  x_full=matrix(NA, n+n_test, K)
  for(k in 1:K)
    x_full[,k] = rnorm(n+n_test,0,1)
  x_full =  (x_full-rnorm(n+n_test,0,sd=cov_sd))/sqrt(1^2+cov_sd^2)
  x=x_full[1:n,]
  x_test=x_full[n+1:n_test,]
  p_full=invlogit(x_full %*% b)
  y_full=rbinom(n=n+n_test,size=1,prob=p_full)
  y=y_full[1:n]
  y_test=y_full[n+1:n_test]
  fit_sample=stan(model_code=stan_code, data=list(N=n,K=K, y=y,x=x), control =list(adapt_delta=0.93)  )
  fit_sample_extract=extract(fit_sample)
  beta_sample=fit_sample_extract[["beta"]]
  lp_sample=fit_sample_extract[["lp__"]]
  cor_vec[j]=cor(beta_sample[,2],beta_sample[,1] )
  temp=apply( beta_sample, 1,log_density_target_mean,x=x_test,y=y_test)
  optimal_lpd[j]=mean(temp)*n_test
  moment_est=array(NA,c(Rep_number,2, 3, K))
  for(sim_replicate in c(1:Rep_number)){
    advi_sample1=advi(iter= iter, iter_skip=200,scale_factor=0.1,x=x, y=y,K=K,  S=100000 )
    k_rep[j,sim_replicate]= advi_sample1$k_final
    elbo_rep[j,sim_replicate]=advi_sample1$elbo_final
    test_rep[j,sim_replicate]=advi_sample1$test_final
    moment_est[sim_replicate, 1,,]  =advi_sample1$first_moment_estimation
    moment_est[sim_replicate, 2,,] =advi_sample1$second_moment_estimation
    if(sim_replicate %% 20 ==0){
      cat( paste (paste ("index", j),      paste (", replicate:", sim_replicate) ))
      cat("\n")
    }
  }

  moment_true=array(NA,c(2, K))
  moment_true[1,]=colMeans(beta_sample)
  moment_true[2,]=colMeans(beta_sample^2)
  for(i in 1:2)
    for(jj in 1:3)
      for(k in 1:K){
        rmse_array[j,i,jj,k]= sqrt( mean( (moment_est[,i,jj,k]- moment_true[i,k])^2 ) )
        bias_array[j,i,jj,k]=mean( (moment_est[,i,jj,k]- moment_true[i,k])  )
        std_array[j,i,jj,k]=sd( (moment_est[,i,jj,k])  )
      }
  print(j)
}

#save(rmse_array,bias_array,std_array,k_rep,elbo_rep,test_rep,optimal_lpd,cor_vec, file= "cov_new_1.RData" )
#load( file="cov_new.RData" )

k_vec=apply(k_rep, 1, mean)
elbo_vec=apply(elbo_rep, 1, mean)
test_vec=apply(test_rep, 1, mean)

k_vec_25=apply(k_rep, 1, quantile,0.25)
k_vec_75=apply(k_rep, 1, quantile,0.75)
k_vec_025=apply(k_rep, 1, quantile,0.025)
k_vec_975=apply(k_rep, 1, quantile,0.975)


test_vec_25=apply(test_rep, 1, quantile,0.25, na.rm=T)*n
test_vec_75=apply(test_rep, 1, quantile,0.75, na.rm=T)*n
test_vec_025=apply(test_rep, 1, quantile,0.025, na.rm=T)*n
test_vec_975=apply(test_rep, 1, quantile,0.975, na.rm=T)*n

rmse_sum=array(NA,c(J,2,3))
for(j in 1:J)
  for(k in 1:2)
    for( i in 1:3)
      rmse_sum[j,k,i]=sum(rmse_array[j,k,i,] )


std_sum=array(NA,c(J,2,3))
for(j in 1:J)
  for(k in 1:2)
    for( i in 1:3)
      std_sum[j,k,i]=sqrt(sum(   std_array[j,k,i,]^2 ))

bias_sum=array(NA,c(J,2,3))
for(j in 1:J)
  for(k in 1:2)
    for( i in 1:3)
      bias_sum[j,k,i]=sum( abs( bias_array[j,k,i,] ))


corr=cov_vector^2/(cov_vector^2+1)


##### Figure 3,   k hat increases as design matrix becomes more and more correlated.   ############

pdf("psis_logistic_cov_new_1.pdf",width=4,height=1.5)
par(mfrow=c(1,3),oma=c(0.8 ,0,1,0 ), pty='m',mar=c(1,1,0.1,0.3) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.9, cex.main=0.9)
cols_temp = col2rgb("red") / 255
cols2 <- rgb(cols_temp[1], cols_temp[2], cols_temp[3], alpha=0.15)
cols3 <- rgb(cols_temp[1], cols_temp[2], cols_temp[3], alpha=0.4)
plot(corr[order(corr) ] ,k_vec[order(corr)] , xlim=c(0,1),ylim=c(0,1), col=1, type='l',cex=0.7,lwd=1, axes=F, xlab="",ylab="", yaxs='i' )

points(corr[order(corr) ] ,k_vec[order(corr)],col=1, pch=19,cex=0.5)
abline(h=c(0.7),col='grey',lty=2)
axis(1,   padj=-1, at=c( 0,0.5,1),lwd=0.5,labels = c(0,0.5,1))
axis(2,   las=2, at=c(0,0.5,0.7,1),lwd=0.5,labels = c("0",".5",".7","1"))
box(bty='l',lwd=0.5)
mtext(1, text="correlations", cex=0.6, line = 0.8)
mtext(3, text="k hat", cex=0.6, line = -0.5, adj = 0.5)


plot(corr[order(corr) ] ,test_vec[order(corr)], xlim=c(0,1),ylim=c(-0.6, -0.25), col=2, type='l',cex=0.7,lwd=0.7, axes=F, xlab="",ylab="", yaxs='i')
lines(corr[order(corr) ] , (optimal_lpd[order(corr)]/n_test),col=4)
points(corr[order(corr) ] ,test_vec[order(corr)],col=2,pch=19,cex=0.4)
points(corr[order(corr) ] , (optimal_lpd[order(corr)]/n_test),col=4,pch=19,cex=0.4)

text(0.7,-0.55, "lpd of \n true posterior",cex=0.8,col=4)
text(0.84,-.32, "lpd of \n VI",cex=0.8,col=2)
#
# polygon(x=c(corr, rev(corr) ),y=c(  test_vec_75 , rev( test_vec_25 ) ),col=cols3 ,border=NA)
# polygon(x=c(corr , rev(corr ) ) ,y=c(  test_vec_025 , rev( test_vec_975) ),col=cols2,border=NA)
axis(2,   las=2, at=c(-0.3,-0.4,-0.5,-0.6),lwd=0.5,labels = c("-.3","-.4","-.5","-.6"))
axis(1,   padj=-1, at=c( 0,0.5,1),lwd=0.5,labels = c(0,0.5,1))
box(bty='l',lwd=0.5)
mtext(1, text="correlations", cex=0.6, line = 0.8)
mtext(3, text=" mean log\n predictive density", cex=0.6, line = -0.5, adj = 0.5)

diff=test_vec-(optimal_lpd/n_test)

plot( k_vec[order(k_vec)],  diff[order(k_vec)], col=1, type='l',cex=0.7,lwd=0.7, axes=F, xlab="",ylab="",  xlim=c(0.2,1),ylim=c(0,0.2) )
points( k_vec[order(k_vec)],  diff[order(k_vec)],pch=19,cex=0.5)
axis(1,  padj=-1, at=c(0.2,0.5,0.7,1),lwd=0.5,labels =c(0.2,0.5,0.7,1) )
axis(2,   las=2, at=c(0,0.1,0.2),lwd=0.5,labels = c("0",".1",".2"))
abline(v=c(0.7),lty=2,col='grey')
mtext(1, text="k hat", cex=0.6, line = 0.8)
mtext(3, text="VI lpd - true lpd", cex=0.6, line =-0.5  )
#mtext(3, text="k hat diagnoses the discrepancy\n in log predictive densities", cex=0.6, line =-0.5)

box(bty='l',lwd=0.5)
dev.off()





##### Figure 4, first and second moment root mean square error is diagnosed by k hat    ############
pdf("psis_logistic_cov_2.pdf",width=4,height=1.5)

par(mfcol=c(1,2),oma=c(0.3,0.7,0.1,0), pty='m',mar=c(1,0.7,0.5,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.5, cex.lab=0.9, cex.main=0.9)

plot(x=k_vec[order(k_vec)][-18], y=rmse_sum[order(k_vec),1,1][-18] ,type='l' , xlab="",ylab="" ,axes=F, yaxs='i',col='blue' , ylim=c(0,6),lwd=1)
box(bty='l',lwd=0.5)

abline(v=c(0.7),lty=2,lwd=0.6, col='grey')

lines(k_vec[order(k_vec)][-18] , rmse_sum[order(k_vec),1,3][-18], col='red' ,lwd=1)
lines(k_vec[order(k_vec)][-18] , rmse_sum[order(k_vec),1,2][-18], col='forest green ',lwd=1)

axis(1,   padj=-1, at=c(0.3,0.5,0.7,0.9),lwd=0.5)
axis(2,   las=2, at=c(0,2.5, 5),lwd=0.5, labels = c(0,2.5,5))
mtext(2, text="RMSE", cex=0.7, line = 0.7)
mtext(3, text="1st Moment", cex=0.7, line=-0.5  )
mtext(1, text="k hat", cex=0.7, line = 0.4)
legend(x="left",legend=c("Raw ADVI", "IS",  "PSIS"), col=c("blue","forest green", "red"),lty=1,lwd=0.6, cex=0.7,border = NA,box.lty=0,text.col=c("blue","forest green", "red"))

plot(x=k_vec[order(k_vec)][-18], y=rmse_sum[order(k_vec),2,1][-18] ,type='l' , xlab="",ylab="" ,axes=F, yaxs='i',col='blue',lwd=0.9, ylim=c(0,60))
box(bty='l',lwd=0.5)

abline(v=c(0.7),lty=2,lwd=0.6, col='grey')
lines(k_vec[order(k_vec)][-18] , rmse_sum[order(k_vec),2,3][-18], col='red' ,lwd=1)
lines(k_vec[order(k_vec)][-18] , rmse_sum[order(k_vec),2,2][-18], col='forest green',lwd=1)
#axis(3,at=c(0.5,0.6,0.7), tick = F,  labels=c("k=0.5","0.6", "0.7" ))
axis(1,   padj=-1, at=c(0.3,0.5,0.7,0.9),lwd=0.5)
axis(2,   las=2, at=c(0,25,50),lwd=0.5)
mtext(1, text="k hat", cex=0.7, line = 0.4)
mtext(2, text="RMSE", cex=0.7, line = 0.7)
mtext(3, text="2nd Moment", cex=0.7,line=-0.5  )
dev.off()




