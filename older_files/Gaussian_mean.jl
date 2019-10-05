function zigzag_Gaussian_mean(X, Σ_prior, Σ_likelihood, Niter, λ_ref=0)

    d = size(Σ_prior)[2]
    Xbar = reshape(mean(X, 2), d, )
    Λ = n*inv(Σ_likelihood) + inv(Σ_prior)
    Λ_tilde = inv(Σ_likelihood)
    θ = ones(d)
    
    skeleton = zeros(d, Niter)
    bouncing_times = zeros(Niter)
    t = 0
    refreshments = 0

    for iter in 2:Niter
        ξ = copy(skeleton[:,iter-1])
        a = [θ[i]*Λ[i,:]'*ξ - n*Λ_tilde[i,:]'*Xbar for i in 1:d]
        b = [θ[i]*Λ[i,:]'*θ for i in 1:d]

        event_times = [get_event_time(a[i], b[i]) for i in 1:d]
        τ, i0 = findmin(event_times)  
        if λ_ref > 0
            refreshment_time = rand(Exponential(1/λ_ref),1)[1]
        end
        if λ_ref == 0 || τ < refreshment_time 
            t += τ
            ξ += θ*τ
            θ[i0] *= -1
        else
            t += refreshment_time
            ξ += θ*refreshment_time
            θ = 2rand(Binomial(1,0.5), d) - 1
            refreshments += 1
        end
        bouncing_times[iter] = t
        skeleton[:,iter] = copy(ξ)
    end
    return skeleton, bouncing_times, refreshments
end

function NRG_Gaussian_mean(X, Σ_prior, Σ_likelihood, Nbounces, max_times)
    d = size(Σ_prior)[2]
    Xbar = reshape(mean(X, 2), d, )
    Λ = n*inv(Σ_likelihood) + inv(Σ_prior)
    Λ_tilde = inv(Σ_likelihood)
    θ = ones(d)
    
    bounce_counter = zeros(d)
    step = 1
    
    chain = zeros(d, Nbounces)
    while sum(bounce_counter) < Nbounces 
        
        ξ = chain[:,step]
        i = sample(1:d, 1)[1]
        s = 0 
        bounces = 0
        
        while s < max_times[i]
            a, b = θ[i]*Λ[i,:]'*ξ - n*Λ_tilde[i,:]'*Xbar, θ[i]*Λ[i,:]'*θ
            τ = get_event_time(a, b)
            if s + τ < max_times[i]
                ξ[i] += θ[i]*τ
                s += τ
                θ[i] *= -1
                bounces += 1
            else 
                ξ[i] += θ[i]*(max_times[i]-s)
                s = max_times[i]
            end
        end
        bounce_counter[i] += bounces
        step +=1
        chain[:,step] = copy(ξ)
        if step > size(chain)[2] - 2
            chain_new = zeros(d, 2size(chain)[2])
            chain_new[:,1:size(chain)[2]] = copy(chain)
            chain = copy(chain_new)
            chain_new = nothing
        end
    end     
    return chain[:,1:step], bounce_counter 
end