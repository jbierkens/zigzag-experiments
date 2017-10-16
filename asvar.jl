function EstimateAsymptoticVarianceSampleMean(X::AbstractArray{Float64,1}, alpha::Float64 = 1/3)
  # estimate the asymptotic variance for a stationary time series using block means
  # see v.d. Vaart, Time Series, Section 5.1
  # alpha determines block size through l = [n^alpha]
  n = length(X)
  l = Int(floor(n^alpha))
  Z = [mean(X[i:(i+l-1)]) for i in 1:(n-l+1)]
  mu = mean(X)
  v = l/(n-l+1) * sumabs2(Z-mu)
  F(x) = 1/(n-l+1) * sum(sqrt(l)*(Z-mu).<=x)
  return (v,F)
end

function BatchMeans(X::AbstractArray{Float64,1}; alpha::Float64 = 1/3, n_batches = -1)

  n = length(X)
  if (n_batches < 0)
    batch_size = Int(floor(n^alpha))
    n_batches = Int(floor(n / n^alpha))
  else
    batch_size = Int(floor(n/n_batches))
  end
  if n_batches > n/2
    return Inf
  end
  unused = n - batch_size * n_batches
  batch_means = [mean(X[unused + (i-1)*batch_size + 1:unused + i*batch_size]) for i in 1:n_batches]
  v = var(batch_means)*batch_size
end

function ess(X::AbstractArray{Float64,1};alpha::Float64=1/3, variance = -1, n_batches=-1)

  asvar = BatchMeans(X,alpha=alpha,n_batches=n_batches)
  if variance < 0
    variance = var(X)
  end
  return length(X) * variance/asvar
end

function SquaredJumpingDistance(X::AbstractArray{Float64})

  if ndims(X) == 1
    n_samples = length(X)
    differences = X[2:end] - X[1:end-1]
  else
    n_samples = size(X)[2]
    differences = X[:,2:end] - X[:,1:end-1]
  end
  return sumabs2(differences)/(n_samples-1)

end

function PooledVariance(AsVar::Vector{Float64}, n_samples::Vector{Int}, SampleVariance::Vector{Float64})

  return sum(n_samples .* SampleVariance ./ AsVar)/sum(n_samples ./AsVar)

end
