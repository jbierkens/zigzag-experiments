# using PyPlot
include("nonidentifiable_logistic.jl")
include("asvar.jl")
include("zigzagtools.jl")

# data initialization
logistic_parameter = -1
dimension = 2
n_observations = 1000
stepsize_MALA = 0.012
dataX = randn(n_observations)
p_vec = 1./(1 + exp(-logistic_parameter * dataX))
dataY = convert(Vector{Int64},rand(n_observations) .<= p_vec)
rho = 1

# for asymptotic variance /ess
n_batches = 500

# MALA initialization
Xi_init = [-1.0;0.0]
n_iter_MALA = 1000
g(xi) = xi[1] + xi[2].^2
U_fun(xi) = rho/2*sumabs2(xi) + sum(log(1 + exp(g(xi).*dataX))- g(xi) * dataX .* dataY)
grad_U(xi) = [rho * xi[1] + sum((1./(1+exp(-g(xi).*dataX))-dataY) .* dataX);
              rho * xi[2] + 2 * xi[2] * sum((1./(1+exp(-g(xi).*dataX))-dataY).* dataX)]

# n_iter_MALA =  * n_iter_ZZ
tic()
Xi_MALA = MALA(U_fun, grad_U, stepsize_MALA, 2, n_iter_MALA, Xi_init)
t_mala = toc()
asvar_mala = (BatchMeans(Xi_MALA[1,:],n_batches=n_batches),BatchMeans(Xi_MALA[2,:],n_batches=n_batches))

# Zig-Zag initialization
tic()
(T,Xi_ZZ) = zzNonidentifiableLogistic(dataX, dataY, 100*n_iter_MALA, Xi_init, rho)
t_zz = toc();
# samples_zz = equidistantSamples(T,Xi_ZZ,n_iter_MALA)
# samples_zz_ext = equidistantSamples(T,Xi_ZZ, Int(1e7))
# asvar_zz = (BatchMeans(samples_zz[1,:],n_batches=n_batches), BatchMeans(samples_zz[2,:],n_batches=n_batches))
# ess_zz_extended = (ess(samples_zz_ext[1,:]),ess(samples_zz_ext[2,:]))

n_batches_consensus = 10
stepsize_consensus = stepsize_MALA * n_batches_consensus
Xi_consensus = consensusNonidentifiableLogistic(dataX, dataY, n_batches_consensus, stepsize_consensus, n_iter_MALA, Xi_init)

n_batches_SGLD = 10
stepsize_SGLD = 0.001
Xi_SGLD = SGLDNonidentifiableLogistic(dataX, dataY, n_batches_SGLD, stepsize_SGLD, n_iter_MALA)

# tic()
# (T_cv,Xi_cv) = @profile cvNonidentifiableLogistic(dataX, dataY, n_proposedswitches_CV, Xi_init, rho)
n_proposedswitches_CV = n_iter_MALA * 100
(T_cv,Xi_cv) = cvNonidentifiableLogistic(dataX, dataY, n_proposedswitches_CV, Xi_init, rho)
# t_cv = toc()
# samples_cv = equidistantSamples(T_cv,Xi_cv, n_iter_MALA)
# asvar_cv = (BatchMeans(samples_cv[1,:],n_batches=n_batches), BatchMeans(samples_cv[2,:],n_batches=n_batches))

# variance_estimate = (PooledVariance([asvar_mala[1], asvar_zz[1], asvar_cv[1]], [length(Xi_MALA[1,:]), length(samples_zz[1,:]), length(samples_cv[1,:])], [var(Xi_MALA[1,:]), var(samples_zz[1,:]), var(samples_cv[1,:])]),
  # PooledVariance([asvar_mala[2], asvar_zz[2], asvar_cv[2]], [length(Xi_MALA[2,:]), length(samples_zz[2,:]), length(samples_cv[2,:])], [var(Xi_MALA[2,:]), var(samples_zz[2,:]), var(samples_cv[2,:])]))
# ess_mala = (ess(Xi_MALA[1,:], variance = variance_estimate[1], n_batches=n_batches), ess(Xi_MALA[2,:], variance = variance_estimate[2], n_batches=n_batches))
# ess_zz = (ess(samples_zz[1,:],variance = variance_estimate[1], n_batches=n_batches),ess(samples_zz[2,:], variance = variance_estimate[2], n_batches=n_batches))
# ess_cv = (ess(samples_cv[1,:], variance = variance_estimate[1], n_batches=n_batches),ess(samples_cv[2,:], variance = variance_estimate[2], n_batches=n_batches))

# samples_cv_ext = equidistantSamples(T,Xi_cv, Int(1e7))
# ess_cv_extended = (ess(samples_cv_ext[1,:]),ess(samples_cv_ext[2,:]))
# println("--------------------")
# println("ESS MALA : ", ess_mala, ", runtime MALA : ", t_mala, "s")
# println("ESS ZZ   : ", ess_zz, ", runtime ZZ   : ", t_zz, "s")
# println("ESS CV   : ", ess_cv, ", runtime CV   : ", t_cv, "s")

# using Gadfly
# l1 = layer(x=Xi_ZZ[1,:], y=Xi_ZZ[2,:],Theme(default_color=color("red")),Geom.path,order=1)
# l2 = layer(x=Xi_MALA[1,:], y=Xi_MALA[2,:],Theme(default_color=color("blue")),Geom.point,order=2)
# p1 = plot(l1,l2,Coord.cartesian(aspect_ratio=1))
# draw(PDF("export.pdf", 10cm,10cm), p1)
n_gridpoints = 100
x = linspace(-3,0.5, n_gridpoints)
y = linspace(-1.5,1.5,n_gridpoints)

xgrid = repmat(x,1,n_gridpoints)
ygrid = repmat(y',n_gridpoints,1)
z = zeros(n_gridpoints, n_gridpoints)
for i in 1:n_gridpoints
  for j in 1:n_gridpoints
    z[i,j] = exp(-U_fun([x[i];y[j]]))
  end
end

using PyPlot
figure(1);
clf();
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
PyPlot.plot(Xi_ZZ[1,:], Xi_ZZ[2,:],color="blue",zorder=6)
# PyPlot.scatter(samples_zz[1,:], samples_zz[2,:],color="blue",zorder=6)
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)
# PyPlot.plot(Xi_cv[1,:], Xi_cv[2,:],color="black",zorder=3)
# PyPlot.scatter(samples_cv[1,:], samples_cv[2,:],color="black",zorder=3)

figure(2);clf();
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
PyPlot.scatter(Xi_MALA[1,:], Xi_MALA[2,:], color="blue",zorder=5)
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)

figure(3);clf();
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
PyPlot.scatter(Xi_consensus[1,:], Xi_consensus[2,:], color="blue")
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)

figure(4);clf();
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
PyPlot.scatter(Xi_SGLD[1,:], Xi_SGLD[2,:], color="blue")
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)

figure(5);clf();
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
PyPlot.plot(Xi_cv[1,:], Xi_cv[2,:], color="blue")
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)

figure(6);clf()
cp = contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
axis([-3.0,0.5,-1.5,1.5])
axis(:equal)
