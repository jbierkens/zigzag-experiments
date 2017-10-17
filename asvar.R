asvar <- function (x, size = "sqroot", warn = FALSE) 
{
  n = length(x)
  if (n < 1000) {
    if (warn) 
      warning("too few samples (less than 1,000)")
    if (n < 10) 
      return(NA)
  }
  if (size == "sqroot") {
    b = floor(sqrt(n))
    a = floor(n/b)
  }
  else if (size == "cuberoot") {
    b = floor(n^(1/3))
    a = floor(n/b)
  }
  else {
    if (!is.numeric(size) || size <= 1 || size == Inf) 
      stop("'size' must be a finite numeric quantity larger than 1.")
    b = floor(size)
    a = floor(n/b)
  }
  y = sapply(1:a, function(k) return(mean(x[((k - 1) * b + 
                                               1):(k * b)])))
  mu.hat = mean(y)
  asvar.hat = b * sum((y - mu.hat)^2)/(a - 1)
  var.hat = var(x)
  
  list(as.var = asvar.hat, ess = n * var.hat/asvar.hat)
}


bm_ess <- function(x, size = "sqroot", warn = FALSE) {
  asvar(x, size = size, warn = warn)$ess
}