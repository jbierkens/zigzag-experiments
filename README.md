# Zig-Zag experiments
Code for reproducing the experiments in J. Bierkens, P. Fearnhead, G. Roberts, https://arxiv.org/abs/1607.03188. 

Note: For users interested in applying the Zig-Zag algorithm to logistic regression, a user friendly `R` logistic regression package is available on CRAN, which can be installed using `install.packages("RZigZag")`. The R experiments below use a different version of this package which has slightly more extensive features but has little documentation.

## Experiments in R: Gaussian mean and logistic regression

This concerns reproducing the experiments of Section 6.3 (Gaussian mean regression) and Section 6.4 (Logistic regression).

1. Download all files and change to the folder containing the files
2. Install the R packages `batchmeans`, `Hmisc`, `Rcpp` and `RcppEigen`, e.g. using the R command
```
install.packages(c("batchmeans","Hmisc", "Rcpp", "RcppEigen"))
```
3. Exit R and install the RZigZig package: from the command line type
```
R CMD INSTALL RZigZag_0.1.tar.gz 
```
4. Now, the experiments can be run using the R-scripts `test_gaussian.R`, `test_gaussian_var.R`, `test_ess_with_observations_2d.R`, `test_ess_with_observations_nd.R`

## Experiment in Julia: non-identifiable logistic regression

This concerns reproducing the experiment of Section 6.5 (Non-identifiable logistic regression), which is implemented in Julia. We assume the user has a working version of Julia.

1. Download all files and change to the folder containing the files
2. Install the Julia packages `PyPlot`, `Polynomials` using the Julia commands
```Julia
Pkg.add("PyPlot")
Pkg.add("Polynomials")
```
3. The experiment can be run using the Julia script `run_nonidentifiable_logistic.jl`
