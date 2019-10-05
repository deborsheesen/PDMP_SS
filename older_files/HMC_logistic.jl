using StatsBase
using Distributions
include("logistic_functions.jl")
include("common_functions.jl")

function gradient_U(ξ, X, y, σ_α, σ_β)
    d, Nobs = size(X)
    α, β = ξ[1:Nobs], ξ[Nobs+1:end]

    grad_β = [derivative_full([reshape(α, 1, Nobs); X], y, [1; β], k+1, σ_β) for k in 1:d] 
    grad_α = [gradient_estimate_α(X[:,i], y[i], i, Nobs, ξ, c, σ_α) for i in 1:Nobs]

    return [grad_α; grad_β]
end

function negative_log_prior(ξ, c, σ_α, σ_β, Nobs)
    α, β = ξ[1:Nobs], ξ[Nobs+1:end]
    σ_ξ = sqrt((1+c^2))/σ_α
    
    log_prior_β = 1/(2σ_β^2)*sum(β.^2)    
    
    μ = [μ_α(α, c, σ_α, i) for i in 1:Nobs]
    log_prior_α = sum((α - μ).^2) / (2σ_ξ^2)  
    
    return log_prior_α + log_prior_β
end 

function U_logistic(ξ, X, y, c, σ_α, σ_β) 
    d, Nobs = size(X)
    α, β = ξ[1:Nobs], ξ[Nobs+1:end]
    return sum(log.(1+exp.(α + X'*β)) - y.*(α + X'*β)) + negative_log_prior(ξ, c, σ_α, σ_β, Nobs)
end

struct Model{T}
    observations::T
end

# return negative log-likelihood and its gradient
function (m::Model)(ξ)
    
    σ_α = 2
    d = size(m.observations)[1] - 1
    Nobs = size(m.observations)[2]
    
    X = m.observations[1:d,:]
    y = m.observations[d+1,:]
    
    α, β = ξ[1:Nobs], ξ[Nobs+1:end]
    
    log_likelihood = sum(log.(1+exp.(α + X'*β)) - y.*(α + X'*β)) + negative_log_prior(ξ, c, σ_α, σ_β, Nobs)
    gradient = gradient_U(ξ, X, y, σ_α, σ_β)
    
    return log_likelihood, gradient    
end


# Source: https://www.cs.toronto.edu/~radford/ham-mcmc-simple

function leapfrog(model, L, δ, current_q, M_inv) 
    
    dim = length(current_q)
    p = rand(Normal(), dim)            # Simulate momentum variables
    current_p = copy(p)
    q = copy(current_q)
    
    p .-= δ/2*model(q)[2]              # Make a half step for momentum at the beginning
    for l in 1:L
        q .+= δ*M_inv*p                # Make a full step for the position
        if l != L                      # Make a full step for the momentum, except at end of trajectory
            p .-= δ*model(q)[2] 
        end    
    end
    p .-= δ/2*model(q)[2]              # Make a half step for momentum at the end
    p *= -1                            # Negate momentum at end of trajectory to make the proposal symmetric
    return p, q, current_p
end

function HMC_basic(model, L, δ, T, dim, mass_matrix=nothing) 
    
    if mass_matrix == nothing 
        mass_matrix = eye(dim)
    end
    M_inv = inv(mass_matrix)
    chain = zeros(dim, T+1)
    chain[:,1] = rand(Normal(), dim)
    accept = 0
    for t in 1:T 
        current_q = copy(chain[:,t])
        p, proposed_q, current_p = leapfrog(model, L, δ, current_q, M_inv) 
        
        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U, proposed_U = model(current_q)[1], model(proposed_q)[1]
#         current_K, proposed_K = current_p'*M_inv*current_p/2, p'*M_inv*p/2
        current_K, proposed_K = current_p'*current_p/2, p'*p/2
        
        # Accept/reject
        if rand(1)[1] < exp(current_U - proposed_U + current_K - proposed_K)
            chain[:,t+1] = copy(proposed_q)
            accept += 1
        else
            chain[:,t+1] = copy(current_q)
        end
    end
    return chain, accept
end

