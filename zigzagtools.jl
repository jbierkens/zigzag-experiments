using Polynomials

hasPositiveRealPart(x::Number) = (real(x) > 0)

pp(x::Number) = max(x,0) # positive part

function findRootMonotoneFunction(f)
  # find root t of f such that t >= 0
  # assume f is a monotonically increasing function such that f(0) < 0

  if f(0.0) >= 0
    error("monotonically increasing function with f(0) >= 0 has no positive root.")
  end
  t0 = 0.0

  prec = 1.0e-10

  # find t2 such that p(t2) > 0
  # may cause infinite loop if p not increasing
  t2 = 1.0
  while (f(t2) <= 0)
    t2 *= 2
  end
  t1 = t2/2

  while abs(f(t1)) > prec
    if f(t1) < 0
      t0 = t1
    else
      t2 = t1
    end
    t1 = t0 + (t2-t0)/2
  end

  return t1

end

function findRootMonotonePolynomial(p)
  # find root t of p such that t >= 0
  # assume f is a monotonically increasing polynomial such that p(0) < 0

  theroots = filter(isreal,roots(p))
  if isempty(r)
    error("No root found")
  else
    maximum(real(theroots))
  end

end


function findRootPolyMaxZero(p::Polynomials.Poly, val::Real)
  # for m(t) = (p(t))^+, find t such that \int_0^t m(s) \ d s = val
  # assume p(t) is monotonically increasing polynomial for t >= 0
  if p(0) >= 0
    # println("using direct method, p(0) >= 0, p'(0) = ", polyder(p)(0), ", value = ", val)
    # return findRootMonotonePolynomial(polyint(p) - val)
    return findRootMonotoneFunction(polyint(p) - val)
  else
    println("shifting")
    t0 = findRootMonotoneFunction(p) # p(t) >= 0 for t >= t0 by assumption
    P_prelim = polyint(p)
    f = t -> P_prelim(t+t0) - P_prelim(t0) - val
    return findRootMonotoneFunction(f) + t0
    # P = P_prelim - P_prelim(t0) - val
    # println("P(x) = ", P)
    # return findRootMonotonePolynomial(P)
  end
end

function equidistantSamples(T::Vector{Float64},X::Array{Float64,2}, n_samples::Int = length(T))

  n_skeletonpoints = length(T)
  dim = size(X)[1]
  t_max = T[end]
  dt = t_max / (n_samples-1)
  samples = Array{Float64,2}(dim,n_samples)
  t_current = dt
  t0 = T[1]
  x0 = X[:,1]
  n_sampled = 0
  for i = 1:n_skeletonpoints-1
    x1 = X[:,i+1]
    t1 = T[i+1]
    while (t_current < t1 && n_sampled < n_samples)
      samples[:,n_sampled + 1] = x0 + (x1-x0) *(t_current - t0)/(t1-t0)
      n_sampled += 1
      t_current += dt
    end
    x0 = x1
    t0 = t1
  end
  # t_travelled = 0
  # current_skeleton_index = 1
  # delta_skeleton_time = T[current_skeleton_index + 1] - T[current_skeleton_index]
  # for i = 1:n_samples
  #   while ((current_skeleton_index + 1) * delta_skeleton_time < stepsize * i)
  #     t_travelled = (current_skeleton_index + 1) * delta_skeleton_time
  #     current_skeleton_index += 1
  #     if current_skeleton_index < n_skeletonpoints
  #       delta_skeleton_time = T[current_skeleton_index + 1] - T[current_skeleton_index]
  #     else
  #       # reached the end, now what?
  #     end
  #   end # of while
  #
  #   samples[i] = X[:,current_skeleton_index] + (stepsize * i - t_travelled)/(delta_skeleton_time) * (X[:,current_skeleton_index+1] - X[:,current_skeleton_index])
  #   t_travelled += stepsize * i
  # end
  #
  #
  #
  return samples[:,1:n_sampled]


end

switchingtime = function (a,b,u)
# generate switching time for rate of the form max(0, a + b s)
  if (b > 0)
    if (a < 0)
      return -a/b + switchingtime(0, b, u);
    else # a >= 0
      return -a/b + sqrt(a^2/b^2 - 2 * log(u)/b);
    end
  elseif (b == 0) # degenerate case
    if (a < 0)
      return Inf;
    else # a >= 0
      return -log(u)/a;
    end
  else # b <= 0
    if (a <= 0)
      return Inf;
    else # a > 0
      y = -log(u); t1=-a/b;
      if (y >= a * t1 + b *t1^2/2)
        return Inf;
      else
        return -a/b - sqrt(a^2/b^2 + 2 * y /b);
      end
    end
  end
end
