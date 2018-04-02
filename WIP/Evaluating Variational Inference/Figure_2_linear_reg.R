######  Figure_2, a bayesian linear regression, ##############################
#############the result is sensitive to the tolorance, which scales with the problem complexity.  K hat gives a convergence diagnoistic
library(rstan)
library(loo)  ### loo is used for PSIS
setwd("")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
stan_code='
data {
int <lower=0> N;
int <lower=0> D;
matrix [N,D] x ;
vector [N] y;
}
parameters {
vector [D] b;
real <lower=0> sigma;
}
model {
y ~ normal(x * b, sigma);
b~normal(0,1);
sigma~gamma(0.5,0.5);
}
generated quantities{
real  log_density;
log_density=normal_lpdf(y |x * b, sigma)+normal_lpdf(b|0,1)+gamma_lpdf(sigma|0.5,0.5)+log(sigma);
}
'
# linear regression model
##log(sigma) in the last line:  the jacobian term in the joint density using log(sigma) as the transformed parameters.

m=stan_model(model_code = stan_code)   # the function for PSIS re-weighting.
ip_weighted_average=function(lw, x){
  ip_weights=exp(lw-min(lw))
  return(  t(ip_weights)%*%x /sum(ip_weights) )
}

set.seed(1000)
N=10000  # a linear regression with 10^5 data and 100 variables
D=100
beta=rnorm(D,0,1)
x=matrix(rnorm(N*D,0,1), N, D)
y=as.vector(x%*%beta+rnorm(N,0,2))

time_temp=proc.time()
fit_stan=stan(model_code=stan_code, data=list(x=x,y=y, D=D,N=N), iter=3000)
time_temp2=proc.time()
time_diff=c(time_temp2-time_temp)
running_time_stan=sum(get_elapsed_time(fit_stan))
stan_sample=extract(fit_stan)
trans_para=cbind(stan_sample$b, log(stan_sample$sigma))
stan_mean= apply(trans_para, 2, mean)
stan_square= apply(trans_para^2, 2, mean)

tol_vec=c(0.03,0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001) # choose relative tolerance. The default in the published ADVI paper is 0.01.
sim_N=50   # repeat simulations. 50 is enough as we can check the sd. The only uncertainty is stochastic optimization in ADVI
I=length(tol_vec)
K_hat=running_time=matrix(NA,sim_N,I )
bias_mean=bias_square=array(NA,c(3,sim_N,I,length(stan_mean)) )
set.seed(1000)
for(i in c(I) )
  for(sim_n in 1:sim_N)
  {
      tol=tol_vec[i]
      time_temp=proc.time()
      fit_vb=vb(m, data=list(x=x,y=y, D=D,N=N), iter=1e5,output_samples=5e4,tol_rel_obj=tol,eta=0.05,adapt_engaged=F)   ## it is also sensitive to eta.
      time_temp2=proc.time()
      time_diff=c(time_temp2-time_temp)
      running_time[sim_n,i]=  time_diff[3]

      vb_samples=extract(fit_vb)
      trans_parameter=cbind(vb_samples$b,log(vb_samples$sigma))
      vi_parameter_mean=apply(trans_parameter, 2, mean)
      vi_parameter_sd=apply(trans_parameter, 2, sd)
      normal_likelihood=function(trans_parameter){
        one_data_normal_likelihood=function(vec){
          return( sum( dnorm(vec,mean=vi_parameter_mean,sd=vi_parameter_sd,  log=T)))
        }
        return( apply(trans_parameter, 1, one_data_normal_likelihood))
      }
      lp_vi=normal_likelihood(trans_parameter)
      lp_target=vb_samples$log_density
      ip_ratio=lp_target-lp_vi

      ok=complete.cases(ip_ratio)
      joint_diagnoistics=psislw(lw=ip_ratio[ok])

      bias_mean[1,sim_n,i,]=vi_parameter_mean-stan_mean
      bias_square[1,sim_n,i,]=apply(trans_parameter^2, 2, mean)-stan_square

      K_hat[sim_n,i]=joint_diagnoistics$pareto_k
      trans_parameter=trans_parameter[ok,]

      psis_lw=joint_diagnoistics$lw_smooth

      bias_mean[2,sim_n,i,]=ip_weighted_average(lw=ip_ratio, x=trans_parameter)-stan_mean
      bias_square[2,sim_n,i,]=ip_weighted_average(lw=ip_ratio, x=trans_parameter^2)-stan_square

      bias_mean[3,sim_n,i,]=ip_weighted_average(lw=psis_lw, x=trans_parameter)-stan_mean
      bias_square[3,sim_n,i,]=ip_weighted_average(lw=psis_lw, x=trans_parameter^2)-stan_square
      print(paste("=======================    i=",i,"   ========================"))
      print(paste("=======================iter",sim_n,"========================"))

  }

running_time=matrix(NA,5,I)   ## calculate the running time again. most of the elapsed time calculated above is on sampling. now we skip the sampling procudure.
set.seed(1000)
for(i in 1:I )
  for(sim_n in 1:5)
  {   tol=tol_vec[i]
      time_temp=proc.time()
      fit_vb=vb(m, data=list(x=x,y=y, D=D,N=N), iter=1e5,output_samples=2,tol_rel_obj=tol,eta=0.09,adapt_engaged=F)
      time_temp2=proc.time()
      time_diff=c(time_temp2-time_temp)
      running_time[sim_n,i]=  time_diff[3]
  }


time_vec=apply(running_time, 2, mean, na.rm=T)

#save(K_hat,running_time_vec,bias_mat, bias_mean_new,bias_square_new,running_time,running_time_stan,file="linear_1e52.RData")
#save(K_hat, bias_mean,bias_square,running_time,running_time_stan,file="linear_1e52_copy.RData")
#load("linear_1e52.RData")
k_vec=apply(K_hat, 2, mean, na.rm=T)   ## average among all repeated simulations
time_vec=apply(running_time, 2, mean, na.rm=T)   ## average among all repeated simulations
bias_mat=matrix(NA,I,3)   ## average among all repeated simulations
for(i in 1:I)
 for(j in 1:3)
   bias_mat[i,j]= sqrt((D+1)*mean(bias_mean_new[j,,i,]^2,na.rm=T)) # L_2 norm of first order error (RMSE)

bias_sq_mat=matrix(NA,I,3)
for(i in 1:I)
  for(j in 1:3)
    bias_sq_mat[i,j]= sqrt((D+1)*mean(bias_square_new[j,,i,]^2,na.rm=T))  # L_2 norm of second order error (RMSE of x^2)

## Figure 2  PSIS in linear regression ########
library(plotrix)
pdf("~/Desktop/linear_large_n.pdf",width=4,height=1.5)   ## Figure 2 linear regression
par(mfcol=c(1,3),oma=c(0.5,0.8,0.1,0.2), pty='m',mar=c(1,0.6,0.5,0.7) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.9, cex.main=0.9)
plot(log10(tol_vec),k_vec,xlim=c(-4.3,-1.5),ylim=c(0,2.8), type='l' , xlab="",ylab="" ,axes=F,lwd=1,yaxs='i',xpd=T)
abline(h=c(0.7),lty=2,lwd=0.6, col='grey')
points(log10(tol_vec),k_vec,pch=19,cex=0.3)
axis(1,padj=-0.5, at=c(-5,-4,-3,-2),labels = c(NA,expression(10^-4),expression(10^-3), expression(10^-2)  )  ,lwd=0.5, cex.axis=0.7)
at.x <- outer(c(2,4,6,8) , 10^(-3:-5))
lab.x <- ifelse(log10(at.x) %% 1 == 0, at.x, NA)
axis(1,padj=-1, at=log10(at.x), labels=NA,lwd=0.2,lwd.ticks=0.4,tck=-0.007)
axis(2, at=c(0,0.5,0.7,1,  2) ,labels=c(0,".5",".7",1,2)  ,lwd=0.5,las=2)
box(bty='l',lwd=0.5)
mtext(2, text="k hat", cex=0.7, line = 0.5)
mtext(1, text="relative tolerance", cex=0.7, line = 0.5,las=1)
plot( time_vec,k_vec,xlim=c(0,72),ylim=c(0,2.8), type='l' , xlab="",ylab="" ,axes=F,lwd=1,yaxs='i',xpd=T,xaxs='i')
abline(h=c(0.7),lty=2,lwd=0.6, col='grey')
points(time_vec,k_vec,pch=19,cex=0.3)
axis(1,padj=-1, at=c(0,20,40,60),lwd=0.5)
axis(2, at=c(0,0.5,0.7,1,  2) ,labels=c(0,".5",".7",1,2)  ,lwd=0.5,las=2)
box(bty='l',lwd=0.5)
mtext(2, text="k hat", cex=0.7, line = 0.5)
mtext(1, text="running time (s)", cex=0.7, line = 0.5,las=1)
axis.break(1,65,style="zigzag")
axis(1,padj=-0.5, at=c(70),lwd=0.5,col=2,labels = NA, tick=0.5, cex.axis=0.4)
round(running_time_stan,-2)
text(67,0.44,labels = "NUTS\n sampling\n time=2300",xpd=T, cex=0.65,col=2)
lines(x=c(70,70),y=c(0,0.15),col=2,lwd=0.5,lty=2)
plot(bias_mat,xlim=c(0.4,1.57),ylim=c(0,0.05),type='n',    xlab="",ylab="" ,axes=F,lwd=1,yaxs='i')
for(i in 1:3)
  lines(k_vec,bias_mat[,i],col=c("blue","red","forest green")[i],lwd=1,xpd=T)
axis(1,padj=-1, at=c(0.5,1,1.5),labels = c(.5,1,1.5),lwd=0.5)
axis(2, at=c(0,0.02,0.04) ,labels=c(0,".02",".04")  ,lwd=0.5,las=2)
mtext(2, text="RMSE", cex=0.6, line = 1)
mtext(1, text="k hat", cex=0.7, line = 0.5,las=1)
text(1.46,0.04,labels = "Raw ADVI",col="blue",xpd=T,cex=0.8)
text(1.48,0.047,labels = "IS",col="forest green",xpd=T,cex=0.8)
text(1.4,0.015,labels = "PSIS",col="red",xpd=T,cex=0.8)
box(bty='l',lwd=0.5)
dev.off()

