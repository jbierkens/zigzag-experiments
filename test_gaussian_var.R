#set.seed(38)
n.observations <- 1e4
rho <- 1
sigma <- 1
true.mean <- 1
epoch.exponents <- 1:4
n.experiments <- 20

MSE.ZZ <- matrix(NA, nrow = length(epoch.exponents), ncol = 1)
MSE.CV <- matrix(NA, nrow = length(epoch.exponents), ncol = 1)
MSE.soCV <- matrix(NA, nrow = length(epoch.exponents), ncol = 1)
MSE.SGLD <- matrix(NA, nrow = length(epoch.exponents), ncol = 1)

require(RZigZag)

for (i in 1:length(epoch.exponents)) {
  n.epochs <- 10^epoch.exponents[i]
  cat("second moment ", n.epochs, "epochs\n")
  SE.ZZ <- matrix(NA, nrow = n.experiments, ncol = 1)
  SE.CV <- matrix(NA, nrow = n.experiments, ncol = 1)
  SE.soCV <- matrix(NA, nrow = n.experiments, ncol = 1)
  SE.SGLD <- matrix(NA, nrow = n.experiments, ncol = 1)
  
  for (j in 1:n.experiments) {
    x <- rnorm(n.observations, mean = true.mean, sd = sigma)
    V <- matrix(1/rho^2 + n.observations/sigma^2, nrow = 1, ncol = 1)
    posterior.mean <- matrix((mean(x))/(1 + sigma^2/(n.observations * rho^2)))
    second.moment <- 1/V[1,1] + (posterior.mean[1,1])^2
    mu.data <- matrix(x / (1 + sigma^2/(n.observations * rho^2)), ncol = n.observations)
    skeleton.ZZ <- ZigZagGaussian(V, posterior.mean, n.epochs, x0=posterior.mean, controlvariates=FALSE, n_samples = n.epochs * n.observations, computeCovariance=TRUE)
    SE.ZZ[j] <- (mean(skeleton.ZZ$samples^2) - second.moment)^2
    cat("c=1: ")
    skeleton.CV <- ZigZagGaussian(V, mu.data, n.epochs, x0=posterior.mean, controlvariates=TRUE, n_samples = n.epochs * n.observations, computeCovariance=TRUE, c = 1)
    SE.CV[j] <- (mean(skeleton.CV$samples^2) - second.moment)^2
    cat("c=0.1: ")
    skeleton.soCV <- ZigZagGaussian(V, mu.data, n.epochs, x0=posterior.mean, controlvariates=TRUE, n_samples = n.epochs * n.observations, computeCovariance=TRUE, c = 0.1)
    SE.soCV[j] <- (mean(skeleton.soCV$samples^2) - second.moment)^2
    result <- SGLDGaussian(V, mu.data, n.epochs, x0=posterior.mean, batchsize = n.observations/100, stepsize = 1/n.observations, thinning = 1)
    SE.SGLD[j] <- (mean(result$x^2) - second.moment)^2
  }
  MSE.ZZ[i] <- mean(SE.ZZ)
  MSE.CV[i] <- mean(SE.CV)
  MSE.soCV[i] <- mean(SE.soCV)
  MSE.SGLD[i] <- mean(SE.SGLD)
}

ymin <- min(MSE.ZZ, MSE.CV, MSE.soCV, MSE.SGLD)
ymax <- max(MSE.ZZ, MSE.CV, MSE.soCV, MSE.SGLD)

ZZ.lm <- lm(log(MSE.ZZ,10) ~ epoch.exponents)
CV.lm <- lm(log(MSE.CV,10) ~ epoch.exponents)
soCV.lm <- lm(log(MSE.soCV,10) ~ epoch.exponents)
SGLD.lm <- lm(log(MSE.SGLD,10) ~ epoch.exponents)

SAVEPATH <- "./"

filename <- paste(SAVEPATH, Sys.Date(), "-mse-secondmoment-", n.observations, "obs.pdf", sep="")
pdf(filename)
plot(10^epoch.exponents, MSE.SGLD, log="xy", col="green", ylim=c(ymin,ymax), xlab="epochs", ylab="MSE", type="p")
abline(SGLD.lm$coefficients[[1]], SGLD.lm$coefficients[[2]],col = 'green')
points(10^epoch.exponents, MSE.ZZ, col="black")
abline(ZZ.lm$coefficients[[1]], ZZ.lm$coefficients[[2]], col='black')
points(10^epoch.exponents, MSE.CV, col="magenta")
abline(CV.lm$coefficients[[1]], CV.lm$coefficients[[2]],col = 'magenta')
points(10^epoch.exponents, MSE.soCV, col="magenta4")
abline(soCV.lm$coefficients[[1]], soCV.lm$coefficients[[2]],col = 'magenta4')
dev.off()


rm(skeleton.CV, skeleton.soCV, skeleton.ZZ, result)
filename <- paste(SAVEPATH, Sys.Date(), "-mse-secondmoment-", n.observations, "obs.RData", sep="")
save.image(filename)
