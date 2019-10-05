using Distributions
include("ZZ_subsampling.jl")

function ZZ_dominated_Hessian(X, y, max_attempts, β_0, σ=10, A=nothing, Print=true) 

    d, Nobs = size(X) 
    if A == nothing 
        A = eye(d)
    end
    m = size(A,1)
    
    bouncing_times = []
    push!(bouncing_times, 0.)
    skeleton_points = zeros(m, 1000)
    skeleton_points[:,1] = A*copy(β_0)
    ξ = copy(β_0)
    θ = ones(d)
    t, switches = 0, 0
    
    Q = X*X'/4 + eye(d)/σ^2
    a = [θ[i]*derivative_full(X, y, ξ, i, Nobs, σ) for i in 1:d] 
    b = [√d*norm(Q[:,i]) for i in 1:d] 
    
    # run sampler:
    for attempt in 1:max_attempts
        event_times = [get_event_time(a[i], b[i]) for i in 1:d]        
        τ, i0 = findmin(event_times)                
        t += τ 
        ξ_new = ξ + τ*θ
        θ_old = copy(θ)
        a += b*τ
        
        rate = pos(θ_old[i0]*derivative_full(X, y, ξ_new, i0, Nobs, σ)[1]) 
        bound = a[i0] 
        alpha = rate/bound
        if alpha > 1 
            print("Error, rate > bound. Rate = ", rate, ", bound = ", bound, "\n")
            break
        elseif rand(1)[1] < alpha
            θ[i0] *= -1
            switches += 1
            skeleton_points[:,switches+1] = A*ξ_new
            push!(bouncing_times, t)
        end   
        
        a[i0] = θ_old[i0]*derivative_full(X, y, ξ_new, i0, Nobs, σ)
        if switches == size(skeleton_points,2) - 1 
            skeleton_points = extend_skeleton_points(skeleton_points)
        end
        ξ = copy(ξ_new)
    end
    if Print == true
        print(signif(100*switches/max_attempts,2),"% of switches accepted \n")
    end
    return hcat(skeleton_points[:,1:switches+1], A*ξ), push!(bouncing_times, t)
end