using Polynomials
# using PyPlot

include("mala.jl")
include("zigzagtools.jl")

function maximalTravelTime(xi, theta, xi_stars, current_region)

  (dimension, n_regions)  = size(xi_stars)

  # crossingtimes = Vector{Float64}(n_regions)
  # for i=1:n_regions
  #   crossingtimes[i]
  #
  crossingtimes = sum((0.5*(xi_stars[:,current_region] .+ xi_stars) .- xi).* (xi_stars[:,current_region] .- xi_stars), 1)./(sum(theta .* (xi_stars[:,current_region] .- xi_stars), 1))
  crossingtimes[(crossingtimes .< 0) | (crossingtimes .==-Inf)] = Inf
  crossingtimes[current_region] = Inf
  return findmin(crossingtimes)
end

function currentRegion(xi, xi_stars)
  (dimension, n_regions)  = size(xi_stars)
  distances = sum((xi .- xi_stars).^2,1)
  return findmin(distances)[2]

end

function determineExtrema(dataX::Vector{Float64}, dataY::Vector{Int64}, rho::Real)

  val = rho + 2 * sum((1./(1 + exp(-dataX/2)) - dataY).*dataX)
  if (val <=0)
    bimodal = true
    n_regions = 3
    xi_stars = Array{Float64,2}(2, n_regions)
    # and so xi_1^* >= 1/2
    f = z -> rho*(z+1/2) + sum((1./(1+exp(-(z+1/2)*dataX))-dataY).*dataX)
    xi1star = findRootMonotoneFunction(f)+1/2
    f = z-> rho/2 + sum((1./(1+exp(-(z^2+1/2)*dataX))-dataY).*dataX)
    xi2hat = findRootMonotoneFunction(f)
    println("Bimodal, xi_1^* = ", xi1star, ", \\hat xi_2 = ", xi2hat)
    xi_stars[:,1] = [0.5, xi2hat]
    xi_stars[:,2] = [0.5,-xi2hat]
    xi_stars[:,3] = [xi1star, 0]
  else
    bimodal = false
    n_regions = 1
    xi_stars = Array{Float64,2}(2,n_regions)
    # and so xi_1^* < 1/2, have to look in reverse direction:
    f = z -> -rho*(1/2-z) - sum((1./(1+exp(-(1/2-z)*dataX))-dataY).*dataX)
    xi1star = 1/2-findRootMonotoneFunction(f)
    xi_stars[:,1] = [xi1star,0]
    println("Unimodal, xi_1^* = ", xi1star)
  end
  return xi_stars
end

function manyReferencePoints(xi_hat, n_referencepoints)
  println("Number of referencepoints: ", n_referencepoints)
  Xi_reference = Array{Float64,2}(2,n_referencepoints)
  xi2_max = 2 # hardcoded for the moment
  delta_xi2 = 2*xi2_max/(n_referencepoints-1)
  println("Delta xi_2 = ", delta_xi2)
  Xi_reference[2,:] = -xi2_max + delta_xi2 * Vector(0:n_referencepoints-1)
  Xi_reference[1,:] = xi_hat[1] .- Xi_reference[2,:].^2
  return Xi_reference
end

function cvNonidentifiableLogistic(dataX::Vector{Float64}, dataY::Vector{Int64}, n_proposedswitches::Int64, xi_init::Vector{Float64} = [0.0;0.0], rho::Real = 1)

  dim = 2;
  array_size = 1000000
  T = zeros(array_size);
  Xi = zeros(dim, array_size);
  Xi[:,1] = xi_init;
  Theta_current = ones(dim);
  Xi_current = Xi[:,1];
  n = length(dataX);
  a = zeros(2,3);
  extrema = determineExtrema(dataX, dataY, rho)
  xi_hat = extrema[:,end]; # MLE
  xi_stars = extrema

  # Gareth's idea
  # alpha = 0.5
  # xi_stars = manyReferencePoints(xi_hat, Integer(ceil(n^alpha)))

  count_proposed_switches = 0
  count_accepted_switches = 0
  current_region = currentRegion(Xi_current+.00000000001*Theta_current,xi_stars) # slight bit of symmetry breaking helps
  (T_max, new_region) = maximalTravelTime(Xi_current, Theta_current, xi_stars, current_region)
  Xi_star = xi_stars[:,current_region]
  x_max = maximum(abs(dataX))
  a[1,3]= n*x_max^2/2;
  a[2,3]= n*x_max^2*abs(Xi_star[2]);

  while count_proposed_switches < n_proposedswitches
    switched = false;
    T_travelled = 0;
    while(~switched)
      relpos = Xi_current - Xi_star
      bigfactor1 = n * x_max^2 * (abs(relpos[1])+2*max(abs(Xi_current[2]),abs(Xi_star[2]))*abs(relpos[2]))
      bigfactor2 = n * x_max^2 * (1 + 2 * abs(relpos[2]) + 2 * max(abs(Xi_current[2]),abs(Xi_star[2])))
      a[1,1]=Theta_current[1]*(relpos[1]) + bigfactor1/4;
      a[1,2]=rho+1/4*bigfactor2;
      a[2,1]=Theta_current[2]*rho*relpos[2] +2*n*x_max*abs(relpos[2]) + abs(Xi_star[2])*bigfactor1/2;
      a[2,2]=rho+2*n*x_max + abs(Xi_star[2])/2*bigfactor2;

      m = Vector{Poly}(dim); # vector of polynomials
      U = -log(rand(dim));
      T_proposed = zeros(dim);
      for j=1:dim
        if all(a[j,:] .<= 0)
          T_proposed[j] = Inf
        else
          m[j] = Poly(a[j,:]);
          # println("j = ", j)
          T_proposed[j] = findRootPolyMaxZero(m[j], U[j]);
        end
        # println("T[", j, "] = ", T_proposed[j])
      end
      (T_min,index_to_switch) = findmin(T_proposed);
      if T_min > T_max # we are entering a new region
        println("Entering region around ", xi_stars[:,new_region])
        T_travelled += T_max
        Xi_current +=  T_max * Theta_current
        current_region = new_region
        (T_max, new_region) = maximalTravelTime(Xi_current +0.00000000001*Theta_current, Theta_current, xi_stars, current_region)
        Xi_star = xi_stars[:,current_region]
      else # we can check if we are actually switching
        count_proposed_switches += 1
        T_travelled += T_min;
        T_max -= T_min # we have a smaller distance to travel to new region
        Xi_current += T_min * Theta_current;
        m_current = m[index_to_switch](T_min);
        J = rand(1:n) # random observation to use
        h_current = 1/(1 + exp(-(Xi_current[1]+Xi_current[2]^2)*dataX[J]));
        h_star = 1/(1 + exp(-(Xi_star[1]+Xi_star[2]^2)*dataX[J]))
        rel_pos = Xi_current - Xi_star
        if index_to_switch == 1
          lambda_current = pp(Theta_current[1]*(rho*rel_pos[1] + n*dataX[J]*(h_current -h_star)))
        else
          lambda_current = pp(Theta_current[2]*(rho*rel_pos[2] + 2*n*dataX[J]*((h_current - dataY[J])*Xi_current[2] - (h_star - dataY[J])*Xi_star[2])))
        end
        V = rand();
        if (lambda_current > m_current + 0.00001)
          println("CV: bound: ", m_current, ", actual value: ", lambda_current)
          println("CV: intermediate bound: ", Theta_current[2]*rho*rel_pos[2] + 2*n*abs(dataX[J])*(abs(rel_pos[2])+abs(Xi_star[2])*abs(dataX[J])*(abs(rel_pos[1])/4 + max(abs(Xi_current[2]),abs(Xi_star[2]))*abs(rel_pos[2]))))
          error("CV: Bound exceeded in iteration ", i, " while switching component ", index_to_switch)
        end
#      println("CV: Xi_current = ", Xi_current)
        if (V <= lambda_current/m_current)
          Theta_current[index_to_switch] = -Theta_current[index_to_switch];
          switched = true;
#        println("switched component ", index_to_switch)
          if count_accepted_switches+2 > array_size
            T_extra = Vector{Float64}(array_size)
            Xi_extra = Array{Float64}(2,array_size)
            T_new = [T; T_extra]
            Xi_new = [Xi Xi_extra]
            T = T_new
            Xi = Xi_new
            array_size *= 2
          end
          T[count_accepted_switches+2] = T[count_accepted_switches + 1] + T_travelled;
          Xi[:,count_accepted_switches+2] = Xi_current;
          count_accepted_switches += 1;
          (T_max,new_region) = maximalTravelTime(Xi_current, Theta_current,xi_stars,current_region)
        end
      end # of if to check whether we switched regions
    end # of while loop until a switch
  end # of for loop over iterations
  println("CV: Fraction of accepted switches: ", count_accepted_switches/count_proposed_switches)
  return (T[1:count_accepted_switches+1],Xi[:,1:count_accepted_switches+1])
end

function zzNonidentifiableLogistic(dataX::Vector{Float64}, dataY::Vector{Int64}, n_proposedswitches::Int64, xi_init::Vector{Float64} = [0.0;0.0], rho::Real = 1)

  dim = 2;
  T = zeros(n_proposedswitches+2);
  Xi = zeros(dim, n_proposedswitches+2);
  Xi[:,1] = xi_init;
  Theta_current = ones(dim);
  Xi_current = Xi[:,1];
  n = length(dataX);
  a = zeros(dim, 2); # coefficients of upper bound polynomial for 0 to 1st order
  a[1,2] = rho
  a[2,2] = rho + 2 * sum(abs(dataX))
  count_proposed_switches = 0
  count_accepted_switches = 0
  m = Vector{Poly}(dim); # vector of polynomials

  while count_proposed_switches < n_proposedswitches
    switched = false;
    T_travelled = 0;
    while(~switched)
      a[1,1] = Theta_current[1]*rho*Xi_current[1] + sum(abs(dataX));
      a[2,1] = Theta_current[2]*rho*Xi_current[2] + 2*abs(Xi_current[2])*sum(abs(dataX));
      # a[2,1] = n *(Theta_current[2] *(rho/n - y_bar) * Xi_current[2] + 2 * Xi_current[1]^2 + 2 * Xi_current[2]^2 + 8 * abs(Xi_current[2])^3);
      U = -log(rand(dim));
      T_proposed = zeros(dim);
      count_proposed_switches += 1
      for j=1:dim
        if all(a[j,:] .<= 0)
          T_proposed[j] = Inf
        else
          m[j] = Poly(a[j,:]);
          # println("j = ", j)
          T_proposed[j] = findRootPolyMaxZero(m[j], U[j]);
        end
        # println("T[", j, "] = ", T_proposed[j])
      end
      (T_min,index_to_switch) = findmin(T_proposed);
      T_travelled += T_min;
      Xi_current += T_min * Theta_current;
      m_current = m[index_to_switch](T_min);
      h_vec = 1 + exp(-(Xi_current[1] + Xi_current[2]^2)*dataX)
      big_sum = sum((1./h_vec  - dataY).*dataX)
      if (index_to_switch == 1)
        lambda_current = pp(Theta_current[1]*(rho *Xi_current[1] + big_sum))
      else
        lambda_current = pp(Theta_current[2]*(rho * Xi_current[2] + 2 * Xi_current[2] * big_sum));
      end
      V = rand();
      if (lambda_current > m_current + 0.001)
        println("bound: ", m_current, ", actual value: ", lambda_current)
        error("Bound exceeded in proposed switch ", count_proposed_switches, " while switching component ", index_to_switch)

      end
      if (V <= lambda_current/m_current)
        Theta_current[index_to_switch] = -Theta_current[index_to_switch];
        switched = true;
#        println("switched component ", index_to_switch)
        T[count_accepted_switches+2] = T[count_accepted_switches+1] + T_travelled;
        Xi[:,count_accepted_switches+2] = Xi_current;
        count_accepted_switches +=1;
      end
    end # of while loop until a switch
  end # of for loop over iterations
  println("ZZ: Fraction of accepted switches: ",count_accepted_switches/count_proposed_switches)
  return (T[1:count_accepted_switches+1],Xi[:,1:count_accepted_switches+1])
end # of function zzNonidentifiableGaussian

function consensusNonidentifiableLogistic(dataX::Vector{Float64}, dataY::Vector{Int64}, n_batches::Int64, stepsize::Float64, n_samples::Int64, xi_init::Vector{Float64} = [0.0;0.0], rho::Real = 1)

  n = length(dataX)
  batch_size = div(n,n_batches)
  Xbatches = reshape(dataX, n_batches, batch_size)
  Ybatches = reshape(dataY, n_batches, batch_size)
  # batch_means = mean(batches, 2)
  rho_batch = rho / n_batches
  # samples = zeros(2, n_samples, n_batches)
  U_fun = Vector{Function}(n_batches)
  grad_U = Vector{Function}(n_batches)
  for i=1:n_batches
    U_fun[i] = xi -> rho_batch/2*sumabs2(xi) + sum(log(1 + exp(g(xi).*Xbatches[i,:]))- g(xi) * Xbatches[i,:] .* Ybatches[i,:])
    grad_U[i] = xi -> [rho_batch * xi[1] + sum((1./(1+exp(-g(xi).*Xbatches[i,:]))-Ybatches[i,:]) .* Xbatches[i,:]);
                       rho_batch * xi[2] + 2 * xi[2] * sum((1./(1+exp(-g(xi).*Xbatches[i,:]))-Ybatches[i,:]).* Xbatches[i,:])]
  end

  return consensusMALA(U_fun, grad_U, stepsize, 2, n_samples, xi_init)
end

function SGLDNonidentifiableLogistic(dataX::Vector{Float64}, dataY::Vector{Int64},n_batches::Int64,stepsize::Float64, n_samples::Int64, xi_init::Vector{Float64} = [0.0;0.0], rho::Real = 1)

  n = length(dataX)
  batch_size = div(n,n_batches)
  Xbatches = reshape(dataX, n_batches, batch_size)
  Ybatches = reshape(dataY, n_batches, batch_size)
  rho_batch = rho / n_batches
  U_fun = Vector{Function}(n_batches)
  grad_U = Vector{Function}(n_batches)
  for i=1:n_batches
    # U_fun[i] = xi -> rho_batch/2*sumabs2(xi) + sum(log(1 + exp(g(xi).*Xbatches[i,:]))- g(xi) * Xbatches[i,:] .* Ybatches[i,:])
    grad_U[i] = xi -> [rho_batch * xi[1] + sum((1./(1+exp(-g(xi).*Xbatches[i,:]))-Ybatches[i,:]) .* Xbatches[i,:]);
                       rho_batch * xi[2] + 2 * xi[2] * sum((1./(1+exp(-g(xi).*Xbatches[i,:]))-Ybatches[i,:]).* Xbatches[i,:])]
  end
  I = rand(1:n_batches, n_samples)
  Xi = zeros(2, n_samples + 1)
  Xi_current = xi_init
  Xi[:,1] = Xi_current
  Z = randn(n_samples)

  for i=1:n_samples
    Xi_current += -stepsize * n/batch_size * grad_U[I[i]](Xi_current) + sqrt(2 * stepsize) * Z[i]
    Xi[:,i+1] = Xi_current
  end

  return Xi

end
