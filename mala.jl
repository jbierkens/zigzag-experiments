function MALA(U_fun::Function, grad_U::Function, h::Real, dim::Integer, n_iter::Integer, Xi_init = zeros(dim))

  Xi = zeros(dim, n_iter + 1)
  Xi[:,1] = Xi_init
  Z = randn(dim, n_iter)
  U = rand(n_iter)
  q(x, y) = exp(-1/(2*h) * sumabs2(y - x + (h/2)*grad_U(x)))
  acceptance_count = 0
  for i=1:n_iter
    Xi_current = Xi[:,i]
    Xi_proposed = Xi_current - (h/2) * grad_U(Xi_current) + sqrt(h) * Z[:,i];
    accept_prob = min(1, exp(U_fun(Xi_current)-U_fun(Xi_proposed)) * q(Xi_proposed, Xi_current) /q(Xi_current,Xi_proposed))
    if U[i] <= accept_prob
      Xi[:,i+1] = Xi_proposed
      acceptance_count += 1
    else
      Xi[:,i+1] = Xi_current
    end
  end
  println("MALA: Fraction of accepted proposals: ", acceptance_count / n_iter)
  return Xi
end

function consensusMALA(U_fun::Vector{Function}, grad_U::Vector{Function}, h::Real, dim::Integer, n_samples::Integer, Xi_init::Vector{Float64} = zeros(dim))
  n_batches = length(U_fun)
  samples = zeros(dim, n_samples, n_batches)
  W = Vector{Array{Float64,2}}(n_batches)  # weights
  cumulative_samples = zeros(dim, n_samples)
  cumulative_weights = zeros(dim,dim)
  for i=1:n_batches
    samples[:,:,i] = MALA(U_fun[i], grad_U[i], h, dim, n_samples, Xi_init)[:,2:end]
    # println(size(samples[:,:,i]))
    # print(cov(samples[:,:,i],2)); println()
    W[i] = inv(cov(samples[:,:,i],2))
    cumulative_samples += W[i] * samples[:,:,i]
    cumulative_weights += W[i]
  end
  println(cumulative_weights)
  consensus_samples = cumulative_weights \ cumulative_samples
end
