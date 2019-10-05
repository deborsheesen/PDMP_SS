using StatsBase
using Distributions
using KernelDensity
include("logistic_functions.jl")


function simulate_α(c, Nobs)
    α_true = zeros(Nobs)
    α_true[1] = rand(Normal())
    for t in 2:Nobs 
        α_true[t] = c*α_true[t-1] + rand(Normal())
    end
    α_true
end

function μ_α(α, c, σ_α, i)
    σ_ξ = sqrt((1+c^2))/σ_α
    if i == 1
        μ = α[2] 
    elseif i == Nobs 
        μ = α[Nobs-1]
    else
        μ = α[i-1] + α[i+1]
    end
    return σ_ξ^2*c*μ
end


function gradient_estimate_α(X_i, y_i, i, Nobs, ξ, c, σ_α)
    σ_ξ = sqrt((1+c^2))/σ_α
    α, β = ξ[1:Nobs], ξ[Nobs+1:end]
    μ_ξ = μ_α(α, c, σ_α, i)
    return exp(α[i] + dot(X_i, β))/(1 + exp(α[i] + dot(X_i, β))) - y_i + (α[i] - μ_ξ)/σ_ξ^2
end

dot(a,b) = sum(a.*b)

function event_time_linear_bound(ai, bi)     # for linear bounds
    # this assumed that bi is positive 
    u = rand(1)[1]
    if ai >= 0 
        return (-ai + sqrt(ai^2 - 2*bi*log(u))) / bi
    else
        return -ai/bi + sqrt(-2*log(u)/bi)
    end
end

function acvf(x, maxlag)
    n = size(x)[1]
    acvf_vec = zeros(maxlag+1)
    xmean = mean(x)
    for lag in 0:maxlag
        index, index_shifted = 1:(n-lag), (lag+1):n
        acvf_vec[lag+1] = mean((x[index]-xmean).*(x[index_shifted]-xmean))
    end
    acvf_vec
end


function plot_traj(X, Nobs)
    plot(size=(900,300), layout=(2,5))
    ticks = [0, Int(floor(size(X)[2]/2)), Int(size(X)[2])]
    plot!(X[1,:], subplot=1, label="", title="alpha dim 1", xticks=ticks, xlabel="iteration")
    plot!(X[2,:], subplot=2, label="", title="alpha dim 2", xticks=ticks, xlabel="iteration")
    plot!(X[3,:], subplot=3, label="", title="alpha dim 3", xticks=ticks, xlabel="iteration")
    plot!(X[4,:], subplot=4, label="", title="alpha dim 4", xticks=ticks, xlabel="iteration")
    plot!(X[5,:], subplot=5, label="", title="alpha dim 5", xticks=ticks, xlabel="iteration")
    plot!(X[Nobs+1,:], subplot=6, label="", title="beta dim 1", xticks=ticks, xlabel="iteration")
    plot!(X[Nobs+2,:], subplot=7, label="", title="beta dim 2", xticks=ticks, xlabel="iteration")
    plot!(X[Nobs+3,:], subplot=8, label="", title="beta dim 3", xticks=ticks, xlabel="iteration")
    plot!(X[Nobs+4,:], subplot=9, label="", title="beta dim 4", xticks=ticks, xlabel="iteration")
    plot!(X[Nobs+5,:], subplot=10, label="", title="beta dim 5", xticks=ticks, xlabel="iteration")
end

function plot_kde(samples, ξ_traj, Nobs)
    plot(size=(900,300), layout=(2,5))

    k1, k2 = kde(samples[1,:]), kde(ξ_traj[1,:])
    plot!(k1.x, k1.density, subplot=1, label="", title="alpha dim 1", xticks=[])
    plot!(k2.x, k2.density, subplot=1, label="")

    k1, k2 = kde(samples[2,:]), kde(ξ_traj[2,:])
    plot!(k1.x, k1.density, subplot=2, label="", title="alpha dim 2", xticks=[])
    plot!(k2.x, k2.density, subplot=2, label="")

    k1, k2 = kde(samples[3,:]), kde(ξ_traj[3,:])
    plot!(k1.x, k1.density, subplot=3, label="", title="alpha dim 3", xticks=[])
    plot!(k2.x, k2.density, subplot=3, label="")

    k1, k2 = kde(samples[4,:]), kde(ξ_traj[4,:])
    plot!(k1.x, k1.density, subplot=4, label="", title="alpha dim 4", xticks=[])
    plot!(k2.x, k2.density, subplot=4, label="")

    k1, k2 = kde(samples[5,:]), kde(ξ_traj[5,:])
    plot!(k1.x, k1.density, subplot=5, label="", title="alpha dim 5", xticks=[])
    plot!(k2.x, k2.density, subplot=5, label="")

    k1, k2 = kde(samples[Nobs+1,:]), kde(ξ_traj[Nobs+1,:])
    plot!(k1.x, k1.density, subplot=6, label="", title="beta dim 1")
    plot!(k2.x, k2.density, subplot=6, label="")

    k1, k2 = kde(samples[Nobs+2,:]), kde(ξ_traj[Nobs+2,:])
    plot!(k1.x, k1.density, subplot=7, label="", title="beta dim 2")
    plot!(k2.x, k2.density, subplot=7, label="")

    k1, k2 = kde(samples[Nobs+3,:]), kde(ξ_traj[Nobs+3,:])
    plot!(k1.x, k1.density, subplot=8, label="", title="beta dim 3")
    plot!(k2.x, k2.density, subplot=8, label="")

    k1, k2 = kde(samples[Nobs+4,:]), kde(ξ_traj[Nobs+4,:])
    plot!(k1.x, k1.density, subplot=9, label="", title="beta dim 4")
    plot!(k2.x, k2.density, subplot=9, label="")

    k1, k2 = kde(samples[Nobs+5,:]), kde(ξ_traj[Nobs+5,:])
    plot!(k1.x, k1.density, subplot=10, label="", title="beta dim 5")
    plot!(k2.x, k2.density, subplot=10, label="")
end

function plot_projection(ξ_traj, s_BPS, b_BPS, Nobs, direction)
    ξ_proj = ξ_traj'*direction
    bps_proj = s_BPS'*direction
    min = minimum([minimum(ξ_proj), minimum(bps_proj)])
    max = maximum([maximum(ξ_proj), maximum(bps_proj)])
    
    plot(layout=(1,2), size=(800,200))
    plot!(b_BPS, bps_proj, subplot=1, label="", title="BPS", ylim=[min, max])
    plot!(ξ_proj, subplot=2, label="", title="NRG", ylim=[min, max])
end