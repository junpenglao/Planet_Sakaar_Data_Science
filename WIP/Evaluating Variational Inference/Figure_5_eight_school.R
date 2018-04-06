######################################
### R code for 8-school model#########
#### figure 5 and more ########
library(loo)
library(rstan)
log_proposal_lik_one_sample=function(beta_one_sample, mu,sd ){
  return( sum (dnorm(beta_one_sample[1:9],mean=mu[1:9],sd=sd[1:9],log = TRUE)[1:9])+ dnorm(log(beta_one_sample[10]),mean=mu[10],sd=sd[10],log = TRUE) +  log(beta_one_sample[10]) )
}
log_proposal_density=function(beta_sample_vi){
  mean_par= apply(beta_sample_vi[,1:9],2, mean)
  mean_par[10]=mean( log(beta_sample_vi[,10] ) )
  sd_par=apply(beta_sample_vi[,1:9],2, sd)
  sd_par[10]=sd( log(beta_sample_vi[,10] ) )
  return(apply(beta_sample_vi, 1, log_proposal_lik_one_sample, mu=mean_par  , sd= sd_par ) )
}
log_density_target=function(beta_one_sample,y,sigma){
  theta=beta_one_sample[1:8]
  mu=beta_one_sample[9]
  tau=beta_one_sample[10]
  return(sum(dnorm( theta, mean=mu, sd=tau, log=T ) ) +sum(  dnorm(y, mean=theta, sd=sigma,log=T)) )
}

est_k_hat=function(vi_marg, marginal_samples){
  log_proposal_coordinate=dnorm(vi_marg, mean= mean(vi_marg), sd=sd(vi_marg) ,log=T)
  true_density<-density(marginal_samples)
  approx_target <- approx(true_density$x, log(true_density$y), vi_marg,rule = 1,yleft=1e-6, yright=1e-6)
  log_target_coordinate<-approx_target$y
  ip_ratio_coordinate<-log_target_coordinate-log_proposal_coordinate
  marginal_k_hat_true<-psislw(lw=ip_ratio_coordinate)$pareto_k
  return(marginal_k_hat_true)
}
# check:  another way to calculate. It returns same result
#   log_proposal_density_trans=function(beta_sample_vi){
#   log_proposal_lik_one_sample=function(beta_one_sample, mu,sd ){
#     return(sum (dnorm(beta_one_sample,mean=mu,sd=sd,log = TRUE)))
#   }
#   return(apply(beta_sample_vi, 1, log_proposal_lik_one_sample, mu=apply(beta_sample_vi, 2, mean), sd=apply(beta_sample_vi, 2, sd))   )
# }

####1 non-centered parametrization
school_code_cp='
data {
int<lower=0> J;          // # schools
real y[J];               // estimated treatment
real<lower=0> sigma[J];  // std err of effect
}
parameters {
vector[J] theta_trans;
real mu;
real<lower=0> tau;
}
transformed parameters{
vector[J] theta;
theta=theta_trans*tau+mu;
}
model {
theta_trans ~normal (0,1);
for(j in 1:J){
y[j] ~ normal(theta[j], sigma[j]);
}
// y ~ normal(theta, sigma);
}'

J = 8
y = c(28,  8, -3,  7, -1,  1, 18, 12)
sigma = c(15, 10, 16, 11,  9, 11, 10, 18)

fit_school2=stan(model_code = school_code_cp, data=list(y=y, sigma=sigma, J=8),iter=5000,control = list(stepsize = 0.5, adapt_delta = 0.9 ))

log_lik_2 <- extract_log_lik(fit_school2, "lp__")
pw_2 <- psislw(-log_lik_2)
pareto_k_table(pw_2)
#loo_2 <- loo(log_lik_2)
#print(loo_2)

####1 centered parametrization

school_code='
data {
   int<lower=0> J;          // # schools
   real y[J];               // estimated treatment
   real<lower=0> sigma[J];  // std err of effect
}
parameters {
   real theta[J];
   real mu;
   real<lower=0> tau;
}
model {
   theta ~ normal(mu, tau);
   y ~ normal(theta, sigma);
}'

##########fit stan and vb
fit_school=stan(model_code = school_code, data=list(y=y, sigma=sigma, J=J),iter=5000,control = list( stepsize = 0.5, adapt_delta = 0.995 ))
log_lik_ <- extract_log_lik(fit_school, "lp__")
pw_ <- psislw(-log_lik_)
pareto_k_table(pw_)

stan_samples=extract(fit_school)
mu_sample=stan_samples[["mu"]]
tau_sample=stan_samples[["tau"]]

fit_vb_temp=fit_vb
m=stan_model(model_code = school_code)
fit_vb=vb( m,  data=list(y=y, sigma=sigma, J=8), iter=1e6,output_samples=1e5,tol_rel_obj=0.01)
vb_sample=extract(fit_vb)
mu_vb=vb_sample$mu
tau_vb=vb_sample$tau


m2=stan_model(model_code = school_code_cp)
fit_vb=vb( m2,  data=list(y=y, sigma=sigma, J=8), iter=3e6,output_samples=1e5,tol_rel_obj=0.01)
vb_sample2=extract(fit_vb)
mu_vb2=vb_sample2$mu
tau_vb2=vb_sample2$tau



######## run diagnoistics
vb_sample_mat=matrix(NA, 10 ,1e5)
vb_sample_mat=cbind(vb_sample$theta,vb_sample$mu, vb_sample$tau)
marginal_k_hat=c()
for(i in 1:8)
  marginal_k_hat[i]=est_k_hat(vb_sample$theta[,i], stan_samples$theta[,i]  )
marginal_k_hat[9]=est_k_hat(  vb_sample$mu, stan_samples$mu )
marginal_k_hat[10]=est_k_hat(  log(vb_sample$tau), log(stan_samples$tau) )


lp_vi=log_proposal_density(vb_sample_mat)
lp_target=apply(vb_sample_mat,1,log_density_target, y=y,sigma=sigma)
ip_ratio=lp_target-lp_vi
joint_diagnoistics=psislw(lw=ip_ratio)
joint_diagnoistics$pareto_k

################ run diagnoistics for cp

# vb_sample_mat=matrix(NA, 18 ,1e5)
# vb_sample_mat=cbind(vb_sample$theta,vb_sample$mu, vb_sample$tau,vb_sample$theta_trans)
#

# mu=apply(vb_sample_mat, 2,mean)
# sd=apply(vb_sample_mat, 2,sd)
# mu[10]=mean( log(vb_sample_mat[,10] ) )
# sd[10]=sd( log(vb_sample_mat[,10] ) )
#
# mu[11:18]=0
# sd[11:18]=1


vb_sample_mat=matrix(NA, 10 ,1e5)
vb_sample_mat=cbind(vb_sample$theta_trans,vb_sample$mu, log(vb_sample$tau))


log_proposal_lik_one_sample=function(beta_one_sample, mu,sd ){
return( sum (dnorm(beta_one_sample,mean=mu,sd=sd,log = TRUE)))
}

log_proposal_density=function(beta_sample_vi){
  mean_par= apply(beta_sample_vi,2, mean)
  sd_par=apply(beta_sample_vi,2, sd)
  # mean_par[1:8]=0
  # sd_par[1:8]=1
  return(apply(beta_sample_vi, 1, log_proposal_lik_one_sample, mu=mean_par  , sd= sd_par ) )
}
lp_vi=log_proposal_density(vb_sample_mat)

log_density_target=function(beta_one_sample,y,sigma){
  theta_trans=beta_one_sample[1:8]
  mu=beta_one_sample[9]
  log_tau=beta_one_sample[10]
  tau=exp(log_tau)
  theta=theta_trans*tau+mu
  return(sum(dnorm( theta_trans, mean=0, sd=1, log=T ) ) +sum(  dnorm(y, mean=theta, sd=sigma,log=T))+ log_tau)
}
lp_target=apply(vb_sample_mat,1,log_density_target, y=y,sigma=sigma)

ip_ratio=lp_target-lp_vi
joint_diagnoistics=psislw(lw=ip_ratio)
joint_diagnoistics$pareto_k

####### moment estimation
moment_vi=matrix(NA,10,2)
moment_est=function(samples)
  return(c(mean(samples),sd(samples)))
for(i in 1:8)
  moment_vi[i,]=moment_est(vb_sample$theta[,i])
moment_vi[9,]=moment_est(vb_sample$mu)
moment_vi[10,]=moment_est(vb_sample$tau)
moment_vi_true=matrix(NA,10,2 )
for(i in 1:8)
  moment_vi_true[i,]=moment_est(stan_samples$theta[,i])
moment_vi_true[9,]=moment_est(stan_samples$mu)
moment_vi_true[10,]=moment_est(stan_samples$tau)
bias=(moment_vi-moment_vi_true)
para_list=c(1:8,"mu", "tau")
para_list[8]="theta_8"


##########   graph   #############
#save(fit_school,fit_vb, vb_sample, stan_samples,marginal_k_hat, file="schools.RData" )
cols <- c(1 ,"blue","red")
cols2 <- sapply(cols, function(i) {
  c2r <- col2rgb(i) / 255
  c2r <- rgb(c2r[1], c2r[2], c2r[3], alpha=0.15)
})

#save(fit_school,fit_vb,vb_sample_mat, bias,bias_old, vb_sample, stan_samples,marginal_k_hat, file="schools_rp.RData" )


pdf("school_rp.pdf",width=6.7,height=3)
par(mfrow=c(2,4),oma=c(2 ,5,1.5,1.5 ), pty='m',mar=c(1,1,2,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.5, cex.lab=0.8, cex.main=0.9)
#########  p. 1  #############
 sample_plot_index=sample(1:length(stan_samples$tau),2000)
 plot(log(stan_samples$tau)[sample_plot_index], stan_samples$theta[sample_plot_index,1],xlim=c(-2,4),ylim=c(-20,50),type='p',pch=20,col=cols2[1],cex=0.4,axes=F,xlab="",ylab=" ")
 mtext(2, text="theta_1",line=0.8,cex=0.5)
 mtext(2, text="NUTS \n sampling",line=1.8,las=2,col=cols[1],cex=0.6)
 axis(1,   padj=-1, at=c(-2,0,2,4),lwd=0.5)
 mtext(1, text=expression(log~tau),line=0.6,cex=0.6)
 axis(2,   las=2, at=c(-20,0,20,40),lwd=0.5)
 box(bty='l',lwd=0.5)
 mtext(3, line=1 ,text ="posterior samples\n of tau and theta_1",cex=0.7)
 #########  p. 2,3  #############
 # for(i in c(3,8)){
 #   plot(0,xlim=c(-20,40),ylim=c(0,0.07),type='n',axes=F,xlab="",ylab=" ",col=cols[1])
 #   lines(density(stan_samples$theta[,i]),col=cols2[1])
 #   xx=density(stan_samples$theta[,i])
 #   polygon(c(xx$x ,rev(xx$x )), c( xx$y ,rep(0,length(xx$x))) , col=cols2[1], border=NA)
 #   if(i==3)
 #     mtext(2, text="density",line=0.5,cex=0.5 )
 #   axis(2, at=c(0,0.05) , las=2 ,lwd=0.5)
 #   axis(1,   padj=-1,  at=c(-20,0,20,40),lwd=0.5)
 #   box(bty='l',lwd=0.5)
 #   mtext(3, line=1 ,text = paste("true marginal density\n of school", i,""),cex=0.7)
 #
 # }

 plot(0,xlim=c(-25,40),ylim=c(0,0.1),type='n',axes=F,xlab="",ylab=" ",col=cols[1])
 lines(density(stan_samples$mu),col='grey')
 xx=density(stan_samples$mu)
 polygon(c(xx$x ,rev(xx$x )), c( xx$y ,rep(0,length(xx$x))) , col=cols2[1], border=NA)

 lines(density(vb_sample$mu),col=cols[2])
 xx=density(vb_sample$mu)
 polygon(c(xx$x ,rev(xx$x )), c( xx$y ,rep(0,length(xx$x))) , col=cols2[2], border=NA)
 mtext(2, text="density",line=0.8,cex=0.5 )
 axis(2, at=c(0,0.1) , las=2 ,lwd=0.5)
 axis(1,   padj=-1,  at=c(-20,0,20,40),lwd=0.5)
 box(bty='l',lwd=0.5)
 mtext(3, line=1 ,text = "marginal density\n of mu",cex=0.7)
 text(-10,0.01,labels=c("Truth"),col='grey',cex=0.6)
 text(18,0.09,labels=c("VI"),col=cols[2],cex=0.6)
 mtext(1, line=0.5 ,text = expression(mu) ,cex=0.6)



 # plot(0,xlim=c(0,40),ylim=c(0,0.2),type='n',axes=F,xlab="",ylab=" ",col=cols[1])
 # xx=density(  log(stan_samples$tau),adjust = 4)
 # x_trans=c(5e-6,exp( xx$x))
 # y_trans=c(5e-6,xx$y /exp( xx$x))
 plot(0,xlim=c(-4,5),ylim=c(0,0.6),type='n',axes=F,xlab="",ylab=" ",col=cols[1])
 xx=density(log(stan_samples$tau),adjust = 1.5)
 lines(xx, col='grey'  )
 x_trans= xx$x
 y_trans= xx$y
 polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[1], border=NA)


 xx=density(log(vb_sample$tau) )
 lines(xx, col=cols[2])
 x_trans= xx$x
 y_trans= xx$y
 polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[2], border=NA)
 axis(2, at=c(0,0.5) , las=2 ,lwd=0.5)
 axis(1,   padj=-1,  at=c(-4,0,4),lwd=0.5)
 box(bty='l',lwd=0.5)
 mtext(3, line=1 ,text = "marginal density\n of log(tau)",cex=0.7)
 text(0,0.5,labels=c("VI"),col=cols[2],cex=0.6)
 text(3,0.3,labels=c("Truth"),col='grey',cex=0.6, xpd=T)
 mtext(1, line=0.6 ,text = expression(log~tau) ,cex=0.6)

 #########  p.4  #############

 plot(1:10,marginal_k_hat,axes=F,xlab="",ylab=" ",type='n',pch=19,xlim=c(1,11),ylim=c(0,1.05))
 axis(2, at=c(0,1) , las=2 ,lwd=0.5)
 axis(1,   padj=-1,  at=c(1:8),labels = c(paste(expression(theta),1,sep="_"),2:8 ) , lwd=0.5, cex.axis=0.5)
 axis(1,   padj=-1,  at=c(9:10),labels = c(expression(mu), expression(tau)) , lwd=0.5, cex.axis=0.6,xpd=T)
 abline(h=c(0.5,0.7), lwd=0.5,col='grey',lty=2)
 axis(2,     at=c(0.5,0.7),labels = c("0.5","0.7") , lwd=0.5, cex.axis=0.7,col.axis='red',las=2)
 for(i in 1:10)
   lines(x=c(i,i),y=c(-3,marginal_k_hat[i]),lty=2,lwd=0.5 )
 points(1:10,marginal_k_hat, pch=20)
 box(bty='l',lwd=0.5)
 mtext(3, line=1 ,text ="marginal and joint \n k hat diagnoistics"  ,cex=0.7)
 mtext(2, line=1 ,text ="marginal k hat"  ,cex=0.5)
 mtext(1, line=0.6 ,text ="parameters"  ,cex=0.5)

 abline(h=joint_diagnoistics$pareto_k,col=2)
 axis(4, at=joint_diagnoistics$pareto_k, tick = F, labels = "joint k \n =0.65",las=2,  cex.axis=0.7,col.axis='red',hadj=0.1)

 #########  p.5  #############
 plot(log(vb_sample$tau)[sample_plot_index],  vb_sample$theta[sample_plot_index,1] ,xlim=c(-2,4),ylim=c(-20,50),type='p',pch=20,col=cols2[2],cex=0.4,axes=F,xlab="",ylab="")
mtext(2, text="theta_1",line=-0.8,cex=0.5)
mtext(1, text=expression(log~tau),line=1,cex=0.6)
mtext(2, text="reparameterized\n ADVI   ",line=0.6,las=2, col=cols[2],cex=0.6)
axis(1,   padj=-1, at=c(-2,0,4),lwd=0.5)
axis(2,   las=2, at=c(-20,0,20,40),lwd=0.5)
box(bty='l',lwd=0.5)
#########  p.6  #############
for(i in c(3,8)){
  plot(0,xlim=c(-20,40),ylim=c(0,0.07),type='n',axes=F,xlab="",ylab=" ",col=2)
  #lines(density(vb_sample$theta[,i]))
  mu=mean( vb_sample$theta[,i])
  sd=sd(vb_sample$theta[,i])
  xxx=seq(-20,40,0.1)
  lines(xxx, dnorm(xxx, mean=mu, sd=sd),col=cols[2]  )
  polygon(c(xxx ,rev(xxx)), c(dnorm(xxx, mean=mu, sd=sd) ,rep(0,length(xxx))) , col=cols2[2], border=NA)
  xx=density(stan_samples$theta[,i])
   lines(xx, col='grey')
   polygon(c(xx$x ,rev(xx$x )), c( xx$y ,rep(0,length(xx$x))) , col=cols2[1], border=NA)
   if(i==3)
    mtext(2, text="density",line=0.5,cex=0.5 )
   text(-7,0.03,labels=c("VI"),col=cols[2],cex=0.6)
   text(16,0.05,labels=c("Truth"),col='grey',cex=0.6)


  axis(2, at=c(0,0.05) , las=2 ,lwd=0.5)
  axis(1,   padj=-1, at=c(-20,0,20,40),lwd=0.5)
  box(bty='l',lwd=0.5)
  mtext(1, line=1 ,text = paste("theta_", i,""),cex=0.6)
  mtext(3, line=-1 ,text = paste("marginal density \n of school effect" ,i),cex=0.6)
}
#########  p.7  #############
plot(bias,col=4,axes=F,xlab="",ylab=" ",xlim=c(-3,5.5),ylim=c(-2.6,2),type='n')
abline(v=0,lwd=0.5,lty=2,col='grey')
abline(h=0,lwd=0.5,lty=2,col='grey')
points(bias,col=4, cex=0.1, pch=20 )
points(bias_old,col='grey60', cex=0.5, pch=20 )

for(i in 1:nrow(bias_old) )
  lines(c(bias_old[i,1], bias[i,1]),c(bias_old[i,2], bias[i,2]),col='grey60',lwd=0.1,lty=2  )
text(bias,col=4,para_list, cex=0.7 ,xpd=T)
text(-2.3,-1.8,labels = "(new\n estimation)",cex=0.5)
text(4,-2,labels = "(previous\n estimation)",cex=0.5)

axis(2, at=c(-2,0,2) , las=2 ,lwd=0.5)
axis(1,   padj=-1, at=c(-3,0,3),lwd=0.5)
box(bty='l',lwd=0.5)
mtext(1, line=1 ,text = "bias of posterior mean",cex=0.6)
mtext(2, line=0.7 ,text = "bias of posterior sd",cex=0.6)
mtext(3, text="VI estimation error", cex=0.7)
axis(4,    at=c(-2.6,2), labels = c("under-\n dispersed","over-\n dispersed"),lwd=0.5, cex.axis=0.6,tick=F,las=2,hadj=0.3)

dev.off()







bias_old=bias




################################################
###### conditonal distribution of theta######
school_code2='
data {
int<lower=0> J;          // # schools
real y[J];               // estimated treatment
real<lower=0> sigma[J];  // std err of effect
real<lower=0> tau;
}
parameters {
real theta[J];
real mu;
}
model {
theta ~ normal(mu, tau);
y ~ normal(theta, sigma);
}'

tau_sim=seq(0.1,30, 0.5)

theta_mean_mat=matrix(NA, 8, length(tau_sim))
theta_sd_mat=matrix(NA, 8, length(tau_sim))
for( i in 1:length(tau_sim)){
fit_school_sim=stan(model_code = school_code2, data=list(y=y, sigma=sigma, J=8 , tau=tau_sim[i]),iter=2000,control = list( stepsize = 0.5, adapt_delta = 0.99 ))
theta_mean_mat[,i]=apply(  extract(fit_school_sim)$theta, 2, mean)
theta_sd_mat[,i]=apply(  extract(fit_school_sim)$theta, 2, sd)
}

############### graph 2###############
save(theta_mean_mat,theta_sd_mat,tau_sim,file="school_conditional.RData")
load("school_conditional.RData")

pdf("school_conditional.pdf",width=6.5,height=1.8)
par(mfrow=c(1,2),oma=c(0.6,4,0,2 ), pty='m',mar=c(1,3,0.5,3) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.5, cex.lab=0.8, cex.main=0.9)

plot(0,xlim=c(0,30),ylim=c(-5,30), type='n',axes=F,xlab="",ylab=" ")
for(j in 1:8)
  lines( lowess(tau_sim, theta_mean_mat[j,],f=0.11))

axis(2, at=c(0,15,30) , las=2 ,lwd=0.5)
axis(1,   padj=-1, at=c(0,10,20,30),lwd=0.5)
box(bty='l',lwd=0.5)
mtext(1, line=0.5 ,text = "fixed tau",cex=0.6)
mtext(2, line=1 ,text = "conditional\n posterior mean\n of theta",cex=0.6,las=2)
axis(4,    at=30, labels = "school\n index",lwd=0.5, tick=F,las=2)
temp=theta_mean_mat[,length(tau_sim)]
axis(4,    at=temp[-c(3,4)], labels = c(1:8)[-c(3,4)],lwd=0.5, tick=F,las=2)
axis(4,    at=temp[c(3,4)], labels = c(1:8)[c(3,4)],lwd=0.5, tick=F,las=2,hadj=1)
lines(y=c(-30,23),  x= rep(mean(vb_sample$tau),2),col='blue',lty=2)
lines(y=c(-30,29),  x= rep(mean(stan_samples$tau),2),col='grey40',lty=2)
text(6, 30, labels = "true posterior \n mean of tau",cex=0.6, col='grey30',xpd=T)
text(15, 24, labels = "VI posterior \n mean of tau",cex=0.6, col='blue')



plot(0,xlim=c(0,30),ylim=c(0,20),  type='n',axes=F,xlab="",ylab=" ")
for(j in 1:8)
   lines( lowess(tau_sim, theta_sd_mat[j,],f=0.2))

axis(2, at=c(-10,0,10,20) , las=2 ,lwd=0.5)
axis(1,   padj=-1, at=c(0,10,20,30),lwd=0.5)
box(bty='l',lwd=0.5)
mtext(1, line=0.5 ,text = "fixed tau",cex=0.6)
mtext(2, line=1 ,text = "conditional\n posterior sd\n of theta",cex=0.6,las=2)
axis(4,    at=20, labels = "school\n index",lwd=0.5, tick=F,las=2)
axis(4,    at=theta_sd_mat[-c(1,2,4,7),length(tau_sim)], labels = c(1:8)[-c(1,2,4,7)],lwd=0.5, tick=F,las=2)
axis(4,    at=theta_sd_mat[c(1,2,4),length(tau_sim)], labels = c(1:8)[c(1,2,4)],lwd=0.5, tick=F,las=2,hadj=1)
axis(4,    at=theta_sd_mat[c(7),length(tau_sim)], labels = c(1:8)[c(7)],lwd=0.5, tick=F,las=2,hadj=-0.8)
lines(y=c(-30,14),  x= rep(mean(vb_sample$tau),2),col='blue',lty=2)
lines(y=c(-30,18),  x= rep(mean(stan_samples$tau),2),col='grey40',lty=2)
text(6, 19, labels = "true posterior \n mean of tau",cex=0.6, col='grey30',xpd=T)
text(13, 15, labels = "VI posterior \n mean of tau",cex=0.6, col='blue')

dev.off()
