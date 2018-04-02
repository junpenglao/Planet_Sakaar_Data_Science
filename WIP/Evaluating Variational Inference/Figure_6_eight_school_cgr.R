 ###  stan and R code for cgr diagnoistics in 8 school model##############
school_code='
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}
parameters {
  real mu;
  real<lower=0> tau;
  real theta[J];
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
'
y=c(28,8,-3,7,-1,1,18,12)  ## data
sigma=c(15, 10, 16, 11, 9,11,10,18)  ## data
J=8
m=stan_model(model_code = school_code)
fit_vb=vb( m,  data=list(y=y, sigma=sigma, J=8), iter=1e5,output_samples=1e5,tol_rel_obj=0.01)
vb_sample=extract(fit_vb)

prob=matrix(NA,M,D)
for(i in 1:M){
  y_sim=as.vector( y_mean[,i]+rnorm(N,0,sigma_0[i])  )
  fit_vb=vb( m, data=list(x=x,y=y_sim, D=D,N=N),iter=1e5,output_samples=5e4,tol_rel_obj=0.01,eta=1,adapt_engaged=F)
  vb_samples=extract(fit_vb)
  b_sim=vb_samples$b
  b_mean=apply(b_sim, 2, mean)
  b_sd=apply(b_sim, 2, sd)
  prob[i,]=pnorm(b_0[,i], b_mean,b_sd)
}

library(LaplacesDemon)
M=1000
prob=matrix(NA,M,10)
for(i in 50:M)
{
  mu_0 = rnorm(1,0, 5)
  tau_0 = rhalfcauchy(1, scale=5)
  theta_0= rnorm(8,mu_0, tau_0)
  y_sim = rnorm(8,theta_0, sigma)
  b_0=c(theta_0, mu_0, log(tau_0))
  fit_vb=vb( m,  data=list(y=y_sim, sigma=sigma, J=8), iter=1e5,output_samples=5e4,tol_rel_obj=0.01,eta=1,adapt_engaged=F)
  vb_samples=extract(fit_vb)
  vb_samples_mat= cbind(vb_samples$theta, vb_samples$mu, log(vb_samples$tau))
  b_mean=apply(vb_samples_mat, 2, mean)
  b_sd=apply(vb_samples_mat, 2, sd)
  prob[i,]=pnorm(b_0, b_mean,b_sd)
}


######## change to non-centered parametrizartion.#########
school_code_rp='
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
y ~ normal(theta, sigma);
mu ~ normal(0, 5);
tau ~ cauchy(0, 5);
}'
m_rp=stan_model(model_code = school_code_rp)

prob_rp=matrix(NA,M,10)
for(i in 1:M)
{
  mu_0 = rnorm(1,0, 5)
  tau_0 = rhalfcauchy(1, scale=5)
  theta_0= rnorm(8,mu_0, tau_0)
  y_sim = rnorm(8,theta_0, sigma)
  b_0=c(theta_0, mu_0, log(tau_0))
  fit_vb=vb( m,  data=list(y=y_sim, sigma=sigma, J=8), iter=1e5,output_samples=5e4,tol_rel_obj=0.01,eta=1,adapt_engaged=F)
  vb_samples=extract(fit_vb)
  vb_samples_mat= cbind(vb_samples$theta, vb_samples$mu, log(vb_samples$tau))

  b_mean=apply(vb_samples_mat, 2, mean)
  b_sd=apply(vb_samples_mat, 2, sd)
  prob_rp[i,]=pnorm(b_0, b_mean,b_sd)
}



par(mfrow=c(2,5))
for(i in 1:10){
  hist(prob[,i],breaks = 25,axes=F,xlim=c(0,1),xlab="", ylab="",main = "",probability = T)
  axis(2)
  xxx1= (prob[,i])
  xxx2= (1-prob[,i])
  ksTest=ks.test(xxx1,xxx2)
  mtext(3, text=paste("KS-test p=",  round(ksTest$p.value,digits=2 ) ), cex=0.7,line=-1  )

}

ks_test_pv=function(vec){
  xxx1= (vec)
  xxx2= (1-vec)
  ksTest=ks.test(xxx1,xxx2)
  return(ksTest$p.value)
}
#save(prob, prob_rp, file="school_cgr.RData")

round(ks_test_pv(prob[,10]),digits=2 )
round(ks_test_pv(prob_rp[,10]),digits=2 )
pdf("/Users/YulingYao/Documents/Research/psis_diagnostics/tex/fig/school_cgr.pdf",height=1.4,width=4)
par(mfrow=c(1,3), oma=c(1 ,0.5,0.7,0 ), pty='m',mar=c(0.4,0.4,0,0.5) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9)
 hist(  prob[,1]  ,breaks = 20,axes=F,xlim=c(0,1),xlab="", ylab="",main = "",probability = T,ylim=c(0,5))#,border = c("grey",rep(1,18),"grey" )
  mtext(3, text="centered", cex=0.7, col=2,line=-0.2  )
  mtext(3, text=expression(theta[1]), cex=0.8,line=-1.2  )
  abline(h=1,col='grey', lty=2)

  mtext(3, text=paste("KS-test p=",  round(ks_test_pv(prob[,1]),digits=2 ) ), cex=0.7,line=-2  )
  axis(1, padj=-1,lwd=0.5, at=c(0,0.5,1), labels = c(0,0.5,1))
  axis(2, at=c(0,2,4),lwd=0.5, las=2)

  lines(x=c(0.5,0.5),y=c(-1,3.8),col='red',lwd=1)
  mtext(1, text=expression(p[theta[1~":"]]), cex=0.6,line=0.6  )



  hist(prob[,10],breaks = 20,axes=F,xlim=c(0,1),xlab="", ylab="",main = "",probability = T,ylim=c(0,10))
  mtext(3, text="p=0.00, reject", cex=0.7,line=-2  )  ## you can check ks_test_pv(prob[,10])=0.00
  mtext(3, text=expression(tau), cex=0.7,line=-1  )
  mtext(3, text="centered", cex=0.7, col=2,line=-0.3  )
  axis(1, padj=-1,lwd=0.5, at=c(0,0.5,1), labels = c(0,0.5,1))
  axis(2, at=c(0,5,10),lwd=0.5, las=2)
  lines(x=c(0.5,0.5),y=c(-1,7.5),col='red',lwd=1)
  mtext(1, text=expression(p[tau~":"]), cex=0.6,line=0.5  )
  abline(h=1,col='grey', lty=2)
  hist(prob_rp[,10],breaks = 20,axes=F,xlim=c(0,1),xlab="", ylab="",main = "",probability = T,ylim=c(0,5))
  mtext(3, text="p=0.00, reject", cex=0.7,line=-2  )
  mtext(3, text=expression(tau), cex=0.7,line=-1  )

  mtext(3, text="non-centered", cex=0.7, col=4 ,line=-0.3 )
  axis(1, padj=-1,lwd=0.5, at=c(0,0.5,1), labels = c(0,0.5,1))
  axis(2, at=c(0,2,4),lwd=0.5, las=2)
  lines(x=c(0.5,0.5),y=c(-1,3.8),col='red',lwd=1)
  mtext(1, text=expression(p[tau~":"]), cex=0.6,line=0.5  )
  abline(h=1,col='grey', lty=2)
dev.off()
