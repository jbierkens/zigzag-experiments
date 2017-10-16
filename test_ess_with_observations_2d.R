require("RZigZag")

generate.logistic.data <- function(beta, nobs) {
  ncomp <- length(beta)
  dataX <- matrix(rnorm((ncomp -1) * nobs), nrow = ncomp -1);
  vals <- beta[1] + colSums(dataX * as.vector(beta[2:ncomp]))
  generateY <- function(p) { rbinom(1, 1, p)}
  dataY <- sapply(1/(1 + exp(-vals)), generateY)
  return(list(dataX, dataY))
}

n.dimension <- 2
beta <- c(1,2)

#set.seed(38)

n.experiments <- 10
exponents <- 6:11
n.epochs <- 1e4
malastepsizes <- 18/(2^(exponents))

ess_zigzag_means <- matrix(nrow = length(exponents), ncol = 1)
ess_upperbound_means <- matrix(nrow = length(exponents), ncol = 1)
ess_subsampling_means <- matrix(nrow = length(exponents), ncol = 1)
ess_cv_means <- matrix(nrow = length(exponents), ncol = 1)
ess_zigzag_sds <- matrix(nrow = length(exponents), ncol = 1)
ess_upperbound_sds <- matrix(nrow = length(exponents), ncol = 1)
ess_subsampling_sds <- matrix(nrow = length(exponents), ncol = 1)
ess_cv_sds <- matrix(nrow = length(exponents), ncol = 1)
ess_MALA_means <- matrix(nrow = length(exponents), ncol = 1)
ess_MALA_sds <- matrix(nrow = length(exponents), ncol = 1)
essps_zigzag_means <- matrix(nrow = length(exponents), ncol = 1)
essps_upperbound_means <- matrix(nrow = length(exponents), ncol = 1)
essps_subsampling_means <- matrix(nrow = length(exponents), ncol = 1)
essps_cv_means <- matrix(nrow = length(exponents), ncol = 1)
essps_zigzag_sds <- matrix(nrow = length(exponents), ncol = 1)
essps_upperbound_sds <- matrix(nrow = length(exponents), ncol = 1)
essps_subsampling_sds <- matrix(nrow = length(exponents), ncol = 1)
essps_cv_sds <- matrix(nrow = length(exponents), ncol = 1)
essps_MALA_means <- matrix(nrow = length(exponents), ncol = 1)
essps_MALA_sds <- matrix(nrow = length(exponents), ncol = 1)

require("batchmeans")
source("asvar.R")

for (j in 1:length(exponents)) {
  n.observations <- 2^exponents[j]
  cat(n.observations,'observations.\n')
  ess_zigzag <- matrix(nrow = n.experiments, ncol = 1)
  ess_upperbound <- matrix(nrow = n.experiments, ncol = 1)
  ess_subsampling <- matrix(nrow = n.experiments, ncol = 1)
  ess_cv <- matrix(nrow = n.experiments, ncol = 1)
  ess_MALA <- matrix(nrow = n.experiments, ncol = 1)
  essps_zigzag <- matrix(nrow = n.experiments, ncol = 1)
  essps_upperbound <- matrix(nrow = n.experiments, ncol = 1)
  essps_subsampling <- matrix(nrow = n.experiments, ncol = 1)
  essps_cv <- matrix(nrow = n.experiments, ncol = 1)
  essps_MALA <- matrix(nrow = n.experiments, ncol = 1)
  
  
  n.batches <- 200

  for (i in 1:n.experiments) {
    # new data set
    logisticData <- generate.logistic.data(beta, n.observations)
    
    # canonical zig zag
    ptm.start <- proc.time()[3]
    result.ZZ <- ZigZagLogistic(logisticData[[1]], logisticData[[2]], n.epochs, subsampling=FALSE, controlvariates=FALSE, beta0 = beta, n_samples=0, n_batches = n.batches, computeCovariance=FALSE)
    ptm.stop <- proc.time()[3]
    ess_zigzag[i] <- result.ZZ$ESS[1]
    essps_zigzag[i] <- ess_zigzag[i]/(ptm.stop - ptm.start)
    
    ptm.start <- proc.time()[3]
    result.upperbound <- ZigZagLogistic(logisticData[[1]], logisticData[[2]], n.epochs, subsampling=FALSE, controlvariates=FALSE, beta0 = beta, n_samples=0, n_batches = n.batches, computeCovariance=FALSE, upperbound=TRUE)
    ptm.stop <- proc.time()[3]
    ess_upperbound[i] <- result.upperbound$ESS[1]
    essps_upperbound[i] <- ess_upperbound[i]/(ptm.stop - ptm.start)
    
    ptm.start <- proc.time()[3]
    result.SS <- ZigZagLogistic(logisticData[[1]], logisticData[[2]], n.epochs, subsampling=TRUE, controlvariates=FALSE, beta0 = beta, n_samples=0, n_batches = n.batches, computeCovariance=FALSE)
    ptm.stop <- proc.time()[3]
    ess_subsampling[i] <- result.SS$ESS[1]
    essps_subsampling[i] <- ess_subsampling[i]/(ptm.stop - ptm.start)
    
    ptm.start <- proc.time()[3]
    result.CV <- ZigZagLogistic(logisticData[[1]], logisticData[[2]], n.epochs, subsampling=TRUE, controlvariates=TRUE, beta0 = beta, n_samples=0, n_batches = n.batches, computeCovariance=FALSE)
    ptm.stop <- proc.time()[3]
    ess_cv[i] <- result.CV$ESS[1]
    essps_cv[i] <- ess_cv[i]/(ptm.stop - ptm.start)
    
    ptm.start <- proc.time()[3]
    result.MALA <- MALALogistic(logisticData[[1]], logisticData[[2]], n.epochs, beta, malastepsizes[j])
    ess_MALA[i] <- bm_ess(result.MALA$beta[1,])
    ptm.stop <- proc.time()[3]
    essps_MALA[i] <- ess_MALA[i]/(ptm.stop - ptm.start)
    
  }
  ess_zigzag_means[j] <- mean(ess_zigzag)/n.epochs
  ess_zigzag_sds[j] <- sd(ess_zigzag)/n.epochs
  
  ess_upperbound_means[j] <- mean(ess_upperbound)/n.epochs
  ess_upperbound_sds[j] <- sd(ess_upperbound)/n.epochs
  
  ess_subsampling_means[j] <- mean(ess_subsampling)/n.epochs
  ess_subsampling_sds[j] <- sd(ess_subsampling)/n.epochs
  
  ess_cv_means[j] <- mean(ess_cv)/n.epochs
  ess_cv_sds[j] <- sd(ess_cv)/n.epochs
  
  ess_MALA_means[j] <- mean(ess_MALA)/n.epochs
  ess_MALA_sds[j] <- sd(ess_MALA)/n.epochs

  essps_zigzag_means[j] <- mean(essps_zigzag)
  essps_zigzag_sds[j] <- sd(essps_zigzag)
  
  essps_upperbound_means[j] <- mean(essps_upperbound)
  essps_upperbound_sds[j] <- sd(essps_upperbound)
  
  essps_subsampling_means[j] <- mean(essps_subsampling)
  essps_subsampling_sds[j] <- sd(essps_subsampling)
  
  essps_cv_means[j] <- mean(essps_cv)
  essps_cv_sds[j] <- sd(essps_cv)
  
  essps_MALA_means[j] <- mean(essps_MALA)
  essps_MALA_sds[j] <- sd(essps_MALA)
}

plot(result.MALA$beta[1,],result.MALA$beta[2,],'p',asp = 1)

SAVEPATH = "./"
## SAVE USEFUL DATA
rm(logisticData)
rm(result.CV, result.SS, result.upperbound, result.ZZ, result.MALA)
filename <- paste(SAVEPATH, Sys.Date(), "-ESSpE-", n.dimension, "d.Rdata", sep="")
save.image(filename)

## FIRST: ESSpE

y_max = log(max(ess_zigzag_means+ess_zigzag_sds, ess_subsampling_means + ess_subsampling_sds, ess_cv_means + ess_cv_sds, ess_upperbound_means+ess_upperbound_sds, ess_MALA_means+ess_MALA_sds),2)
y_min = log(min(ess_zigzag_means-ess_zigzag_sds, ess_subsampling_means - ess_subsampling_sds, ess_cv_means - ess_cv_sds, ess_upperbound_means - ess_upperbound_sds, ess_MALA_means-ess_MALA_sds),2)

filename <- paste(SAVEPATH, Sys.Date(), "-ESSpE-", n.dimension, "d.pdf", sep="")
pdf(filename)

require("Hmisc")
errbar(exponents, log(ess_zigzag_means,2), log(ess_zigzag_means+ess_zigzag_sds,2), log(ess_zigzag_means-ess_zigzag_sds,2), add=FALSE, pch=1, ylim=c(-6.0, +5.0), cap=.015, ann=FALSE)
errbar(exponents, log(ess_upperbound_means,2), log(ess_upperbound_means+ess_upperbound_sds,2), log(ess_upperbound_means-ess_upperbound_sds,2), add=TRUE, pch=1, cap=.015, xlog = TRUE, ylog=TRUE,
       col = 'red', errbar.col = 'red', ann=FALSE)

errbar(exponents, log(ess_subsampling_means,2), log(ess_subsampling_means+ess_subsampling_sds,2), log(ess_subsampling_means-ess_subsampling_sds,2), add=TRUE, pch=1, cap=.015, xlog = TRUE, ylog=TRUE,
       col = 'blue', errbar.col = 'blue', ann=FALSE)

errbar(exponents, log(ess_cv_means,2), log(ess_cv_means+ess_cv_sds,2), log(ess_cv_means-ess_cv_sds,2),add=TRUE, pch=1, cap=.015,
       xlog = TRUE, ylog=TRUE, col = 'magenta', errbar.col = 'magenta', ann=FALSE)

errbar(exponents, log(ess_MALA_means,2), log(ess_MALA_means+ess_MALA_sds,2), log(ess_MALA_means-ess_MALA_sds,2),add=TRUE, pch=1, cap=.015,
       xlog = TRUE, ylog=TRUE, col = 'green', errbar.col = 'green', ann=FALSE)

basic.lm <- lm(log(ess_zigzag_means,2) ~ exponents)
upperbound.lm <- lm(log(ess_upperbound_means,2) ~ exponents)
subsampling.lm <- lm(log(ess_subsampling_means,2) ~ exponents)
cv.lm <- lm(log(ess_cv_means,2) ~ exponents)
MALA.lm <- lm(log(ess_MALA_means,2) ~ exponents)

abline(basic.lm$coefficients[[1]], basic.lm$coefficients[[2]],col = 'black')
abline(upperbound.lm$coefficients[[1]], upperbound.lm$coefficients[[2]],col = 'red')
abline(subsampling.lm$coefficients[[1]], subsampling.lm$coefficients[[2]], col = 'blue')
abline(cv.lm$coefficients[[1]], cv.lm$coefficients[[2]], col='magenta')
abline(MALA.lm$coefficients[[1]], MALA.lm$coefficients[[2]], col='green')

title(xlab="log(number of observations) base 2", ylab="log(ESS / epoch) base 2")
dev.off()


## BELOW: ESSps plot
y_max = log(max(essps_zigzag_means+essps_zigzag_sds, essps_subsampling_means + essps_subsampling_sds, essps_cv_means + essps_cv_sds, essps_upperbound_means+essps_upperbound_sds, essps_MALA_means+essps_MALA_sds),2)
y_min = log(min(essps_zigzag_means-essps_zigzag_sds, essps_subsampling_means - essps_subsampling_sds, essps_cv_means - essps_cv_sds, essps_upperbound_means - essps_upperbound_sds, essps_MALA_means-essps_MALA_sds),2)

filename <- paste(SAVEPATH, Sys.Date(), "-ESSps-", n.dimension, "d.pdf", sep="")
pdf(filename)

errbar(exponents, log(essps_zigzag_means,2), log(essps_zigzag_means+essps_zigzag_sds,2), log(essps_zigzag_means-essps_zigzag_sds,2), add=FALSE, pch=1, ylim=c(5,16),cap=.015, ann=FALSE)

errbar(exponents, log(essps_upperbound_means,2), log(essps_upperbound_means+essps_upperbound_sds,2), log(essps_upperbound_means-essps_upperbound_sds,2), add=TRUE, pch=1, cap=.015, xlog = TRUE, ylog=TRUE,
       col = 'red', errbar.col = 'red', ann=FALSE)
errbar(exponents, log(essps_subsampling_means,2), log(essps_subsampling_means+essps_subsampling_sds,2), log(essps_subsampling_means-essps_subsampling_sds,2), add=TRUE, pch=1, cap=.015, xlog = TRUE, ylog=TRUE,
       col = 'blue', errbar.col = 'blue', ann=FALSE)
errbar(exponents, log(essps_cv_means,2), log(essps_cv_means+essps_cv_sds,2), log(essps_cv_means-essps_cv_sds,2),add=TRUE, pch=1, cap=.015,
       xlog = TRUE, ylog=TRUE, col = 'magenta', errbar.col = 'magenta', ann=FALSE)

errbar(exponents, log(essps_MALA_means,2), log(essps_MALA_means+essps_MALA_sds,2), log(essps_MALA_means-essps_MALA_sds,2),add=TRUE, pch=1, cap=.015,xlog = TRUE, ylog=TRUE, col = 'green', errbar.col = 'green', ann=FALSE)

basic.lm <- lm(log(essps_zigzag_means,2) ~ exponents)
upperbound.lm <- lm(log(essps_upperbound_means,2) ~ exponents)
subsampling.lm <- lm(log(essps_subsampling_means,2) ~ exponents)
cv.lm <- lm(log(essps_cv_means,2) ~ exponents)
MALA.lm <- lm(log(essps_MALA_means,2) ~ exponents)

abline(basic.lm$coefficients[[1]], basic.lm$coefficients[[2]],col = 'black')
abline(upperbound.lm$coefficients[[1]], upperbound.lm$coefficients[[2]],col = 'red')
abline(subsampling.lm$coefficients[[1]], subsampling.lm$coefficients[[2]], col = 'blue')
abline(cv.lm$coefficients[[1]], cv.lm$coefficients[[2]], col='magenta')
abline(MALA.lm$coefficients[[1]], MALA.lm$coefficients[[2]], col='green')
title(xlab="log(number of observations) base 2", ylab="log(ESS per second) base 2")
dev.off()
