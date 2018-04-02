#### Figure 7 ADVI has a bad fit in Horseshoe  logistic

library(rstan)
options(mc.cores = parallel::detectCores())
datafile <- 'leukemia.RData'
load(datafile,verbose=T)
x <- scale(x)
d <- NCOL(x)
n <- NROW(x)
# compile the model
stanmodel <- stan_model('glm_bernoulli_rhs.stan')
scale_icept=10
slab_scale=5
slab_df=4
# data and prior
tau0 <- 1/(d-1) * 2/sqrt(n) # should be a reasonable scale for tau (see Piironen&Vehtari 2017, EJS paper)
scale_global=tau0
data <- list(n=n, d=d, x=x, y=as.vector(y), scale_icept=10, scale_global=tau0,
             slab_scale=5, slab_df=4)
# NUTS solution (increase the number of iterations, here only 100 iterations to make this script run relatively fast)
fit_nuts <- sampling(stanmodel, data=data, iter=3000, control=list(adapt_delta=0.9))
# save(fit_nuts, file="stan_fit.RData")
# save(fit_advi, file="vi_fit.RData")
# load("vi_fit.RData")
# load("stan_fit.RData")
# ADVI
fit_advi <- vb(stanmodel, data=data,iter=1e6,output_samples=1e4,tol_rel_obj=0.001,eta = 0.1 )
# example of how to make predictions (here on training data)
e_vi <- extract(fit_advi)
f <- e$beta %*% t(x) + as.vector(e$beta0)
mu <- colMeans(binomial()$linkinv(f))
plot(mu,y)

# investigate the posterior of the coefficients of the most relevant variables
# these plots should reveal the multimodality
e <- extract(fit_nuts)
ind <- order(abs(colMeans(e$beta)), decreasing = T)[1:5] # indices of the coefficient with largest absolute mean
qplot(e$beta[,ind[1]], e$beta[,ind[2]])
qplot(e$beta[,ind[1]])
qplot(e$beta[,ind[2]])

# result from ADVI should probably be somewhat different as it is likely to catch only one mode
e_vi <- extract(fit_advi)
qplot(e_vi$beta[,ind[1]], e_vi$beta[,ind[2]])
qplot(e_vi$beta[,ind[1]])
qplot(e_vi$beta[,ind[2]])




density_target=c()
library(arm)
library(invgamma)
S=length(e_vi$beta0)
for( i in 1:S)
density_target[i]=sum( log(invlogit( e_vi$f[i,]))* y +log(1- invlogit(  e_vi$f[i,]))*(1-y) )+dnorm(e_vi$beta0[i], 0,  scale_icept, log=T)+   dinvgamma(e_vi$caux[i], 0.5*slab_df, 0.5*slab_df, log=T)+ dcauchy(e_vi$tau[i] ,0,scale_global,log=T)+  sum(dcauchy(e_vi$lambda[i,] ,0,1, log=T))+ sum(dnorm(e_vi$z[i,], 0,1, log=T))+log(e_vi$tau[i])+sum(log(e_vi$lambda[i,]))+log(e_vi$caux[i])

trans_parameter=cbind(e_vi$beta0, e_vi$z, log(e_vi$tau),log(e_vi$lambda),log(e_vi$caux) )
  vi_parameter_mean=apply(trans_parameter, 2, mean)
  vi_parameter_sd=apply(trans_parameter, 2, sd)
  one_data_normal_likelihood=function(vec){
    return( sum( dnorm(vec,mean=vi_parameter_mean,sd=vi_parameter_sd,  log=T)))
  }

  lp_vi= apply(trans_parameter, 1, one_data_normal_likelihood)
  ip_ratio=density_target-lp_vi

  library(loo)
  joint_diagnoistics=psislw(lw=ip_ratio[complete.cases(ip_ratio)])
  joint_diagnoistics$pareto_k





  cols <- c(1 ,"blue","red")
  cols2 <- sapply(cols, function(i) {
    c2r <- col2rgb(i) / 255
    c2r <- rgb(c2r[1], c2r[2], c2r[3], alpha=0.15)
  })
  pdf("horseshoe.pdf",width=4,height=4/3)

  par(mfrow=c(1,3),oma=c(1.6 ,1.5,0,0.5 ), pty='m',mar=c(0.5,0.4,0.5,0.3) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.6, cex.lab=0.8, cex.main=0.9,xpd=F)
  plot(0,xlim=c(-3,15),ylim=c(0,0.6) ,type='n',axes=F,xlab="",ylab=" ",yaxs='i' )
  xx=density( e$beta[,ind[1]])
  lines(xx, col=cols[2] ,lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[2], border=NA)

  xx=density(e_vi$beta[,ind[1]],adjust=200)
  lines(xx, col=cols[3],lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[3], border=NA,xpd=T)
  axis(1,   padj=-1, at=c(0,5,10,15),lwd=0.5)
  mtext(1, text=expression(beta[1834]),line=1,cex=0.7)
  axis(2,   las=2, at=c(0,0.3,0.6),label=c("0",".3",".6"),lwd=0.5)
  box(bty='l',lwd=0.5)
  mtext(2, line=1 ,text =" posterior desnity",cex=0.7)

  text(2,.4 ,  labels ="VI",cex=0.85,col=2)
  text(10,.1 ,  labels ="NUTS",cex=0.85,col=4)


  plot(0,xlim=c(-6,18),ylim=c(0,0.3) ,type='n',axes=F,xlab="",ylab=" ",yaxs='i' )
  xx=density( log(e$lambda[, ind[1]]))
  lines(xx, col=cols[2] ,lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[2], border=NA)

  xx=density(log(e_vi$lambda[, ind[1]]),adjust=2)
  lines(xx, col=cols[3],lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[3], border=NA,xpd=T)
  axis(1,   padj=-1, at=c(-5,5,15),lwd=0.5)
  mtext(1, text=expression(log~lambda[1834]),line=1,cex=0.7)
  axis(2,   las=2, at=c(0,0.15,0.3),label=c("0",".15",".3"), lwd=0.5)
  box(bty='l',lwd=0.5)



  plot(0,xlim=c(-13,-5),ylim=c(0,1) ,type='n',axes=F,xlab="",ylab=" ",yaxs='i' )
  xx=density( log(e$tau),adj=1.8)
  lines(xx, col=cols[2] ,lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[2], border=NA,xpd=T)

  xx=density(log(e_vi$tau),adjust=2)
  lines(xx, col=cols[3],lwd=0.5)
  x_trans= xx$x
  y_trans= xx$y
  polygon(c(x_trans ,rev(x_trans) ), c(y_trans ,rep(0,length(x_trans))) , col=cols2[3], border=NA)
  axis(1,   padj=-1, at=c(-11,-8,-5),lwd=0.5)
  mtext(1, text=expression(log~tau),line=1,cex=0.7)
  axis(2,   las=2, at=c(0,0.5,1),label=c("0",".5","1"),lwd=0.5)
  box(bty='l',lwd=0.5)

dev.off()





sigma_t=mean(y)*(1-mean(y))


eff_k_stan=matrix(NA,dim(e$lambda)[1],d)
eff_k_vi=matrix(NA,dim(e_vi$lambda)[1],d)
for(i in  1:d){
  eff_k_stan[,i]=1/(1+n*sigma_t^(-2)*e$tau^2*e$lambda[,i]^2)
  eff_k_vi[,i]=1/(1+n*sigma_t^(-2)*e_vi$tau^2*e_vi$lambda[,i]^2)
}

length(  apply(eff_k_stan, 1,sum))
length(  apply(eff_k_vi, 1,sum))

m_eff_stan= apply(eff_k_stan, 1,sum)
m_eff_vi= apply(eff_k_vi, 1,sum)[sample(1:8000,2400)]



pdf("linear_reg_cgr_m_eff.pdf",height=1.4, width=3)
par(mfrow=c(1,2),oma=c(1,1,1,0), pty='m',mar=c(0.5,0.4,0.5,0) ,mgp=c(1.5,0.25,0), lwd=0.5,tck=-0.01, cex.axis=0.5, cex.lab=0.9, cex.main=0.9)
hist(m_eff_vi, prob=T,breaks = seq(d,4000,-60) ,xlim=c(6000,7140),ylim=c(0,0.008),axes=F,xlab="", ylab="",main = "")
lines(x=rep(mean(m_eff_vi),2),y=c(-1,0.005), col=2)
text(6700,0.005,labels = "posterior\n mean\n = 6988",cex=0.7,col=2)
axis(1, padj=-1,lwd=0.5, at=c(6000,6500,7000))
axis(2, lwd=0.5, at=c(0,0.002,0.004),las=2)
mtext(1, text=expression(m[eff]), line = 0.6, cex=0.7)
mtext(3, text="in VI posteriors ", line = -0.3, cex=0.7)

hist(m_eff_stan, prob=T,breaks = seq(d,4000,-60) ,xlim=c(6000,7140),ylim=c(0,0.008),axes=F,xlab="", ylab="",main = "")
lines(x=rep(mean(m_eff_stan),2),y=c(-1,0.005), col=2)
text(6700,0.005,labels = "posterior\n mean\n = 6940",cex=0.7,col=2)
axis(1, padj=-1,lwd=0.5, at=c(6000,6500,7000))
mtext(1, text=expression(m[eff]), line = 0.6, cex=0.7)
mtext(3, text=" in NUTS posteriors ", line = -0.3, cex=0.7)
mtext(3, text=" effective number of parameters ", line = 0, cex=0.7,outer=T)
dev.off()
