using Distributions, Optim, ProgressMeter
include("mbsampler.jl")

abstract type outputtimer end
abstract type outputformater end

abstract type ll_model end
abstract type prior_model end
abstract type msampler end
    
# Define abstract type for gaussian prior, sub-types of this abstract types must have attributes mu and sigma2  
abstract type gaussian_prior <:prior_model end
abstract type laplace_prior <:prior_model end
abstract type bound end

struct ll_logistic_sp<:ll_model
    X::SparseMatrixCSC
    y::Array{Int64}
    Nobs::Int64
end
ll_logistic_sp(X, y) = ll_logistic_sp(X, y, length(y)) 


mutable struct const_bound<:bound
    a::Array{Float64}
end

mutable struct linear_bound<:bound
    const_::Array{Float64} # storing whatever constants we have 
    a::Array{Float64}
    b::Array{Float64}
end

mutable struct outputscheduler
    opf::outputformater
    opt::outputtimer
end

mutable struct zz_state 
    ξ::Array{Float64}
    θ::Array{Float64}
    α::Array{Float64}
    n_bounces::Array{Int64}
    est_rate::Array{Float64}
    T::Float64
    mu::Array{Float64}
    m2::Array{Float64} #second moment
    ξ_lastbounce::Array{Float64}
    T_lastbounce::Float64
end

zz_state(d) = zz_state(zeros(d), ones(d), ones(d), zeros(d), ones(d), 0., zeros(d), zeros(d), zeros(d), 0.)


mutable struct zz_sampler <:msampler
    i0::Int64
    gs::sampler_list
    bb::bound
    L::Int64
    adapt_speed::String
end

mutable struct zz_sampler_decoupled <:msampler
    i0::Int64
    gs::sampler_list
    bb_pr::bound
    bb_ll::bound
    L::Int64
    adapt_speed::String
end

function feed(outp::outputscheduler, state::zz_state, time::Float64, bounce::Bool)
    
    if add_output(outp.opf, state, time, bounce)
        outp.opf.tcounter +=1 
        if outp.opf.tcounter > size(outp.opf.bt_skeleton,2)
            outp.opf.xi_skeleton = extend_skeleton_points(outp.opf.xi_skeleton, outp.opf.size_increment)
            outp.opf.bt_skeleton = extend_skeleton_points(outp.opf.bt_skeleton, outp.opf.size_increment)
            outp.opf.theta_skeleton = extend_skeleton_points(outp.opf.theta_skeleton, outp.opf.size_increment)
            outp.opf.alpha_skeleton = extend_skeleton_points(outp.opf.alpha_skeleton, outp.opf.size_increment)
        end
        outp.opf.xi_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, state.ξ)
        outp.opf.theta_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, state.θ)
        outp.opf.bt_skeleton[:,outp.opf.tcounter] = time
        outp.opf.alpha_skeleton[:,outp.opf.tcounter] = compress_xi(outp.opf, state.α)
        
        outp.opf.theta = state.θ
        outp.opf.n_bounces = state.n_bounces
        
        outp.opf.xi_mu = state.mu
        outp.opf.xi_m2 = state.m2
    end
    outp.opt = eval_stopping(outp.opt, state.ξ, time, bounce)
    return outp
end

#--------------------------------------------------------------------------------------------------------

function is_running(opt::outputtimer)
    return opt.running
end

#--------------------------------------------------------------------------------------------------------

function add_output(opf::outputformater, state::zz_state, time::Float64, bounce::Bool)
   return bounce 
end

function compress_xi(opf::outputformater, xi)
   return xi 
end

function extend_skeleton_points(skeleton_points, extension=1000)
    m, n = size(skeleton_points)
    skeleton_new = zeros(m, n+extension)
    skeleton_new[:,1:n] = skeleton_points
    return skeleton_new
end 

#--------------------------------------------------------------------------------------------------------

function finalize(opf::outputformater)
    opf.xi_skeleton = opf.xi_skeleton[:,1:opf.tcounter]
    opf.bt_skeleton = opf.bt_skeleton[:,1:opf.tcounter]
    opf.alpha_skeleton = opf.alpha_skeleton[:,1:opf.tcounter]
end

mutable struct projopf <:outputformater
    d::Int64
    xi_skeleton::Array{Float64}
    bt_skeleton::Array{Float64}
    theta::Array{Float64} 
    theta_skeleton::Array{Float64}
    alpha_skeleton::Array{Float64}
    n_bounces::Array{Int64}
    tcounter::Int64
    size_increment::Int64
    A
    d_out::Int64
    xi_mu::Array{Float64}
    xi_m2::Array{Float64}
end

projopf(A, size_increment::Int64) = projopf(built_projopf(A, size_increment)...)

zz_state(opf::projopf) = zz_state(opf.xi_skeleton[:,opf.tcounter], opf.theta, opf.alpha_skeleton[:,opf.tcounter], opf.n_bounces, ones(length(opf.theta)))


function built_projopf(A, size_increment)
    d_out, d = size(A)
    xi_skeleton = zeros(d_out, 10*size_increment)
    bt_skeleton = zeros(1, 10*size_increment)
    theta_skeleton = ones(d_out, 10*size_increment)
    tcounter = 1
    theta = ones(d)
    alpha_skeleton = ones(d_out, 10*size_increment)
    n_bounces = zeros(d)
    xi_mu = zeros(d)
    xi_m2 = zeros(d)
    return d, xi_skeleton, bt_skeleton, theta, theta_skeleton, alpha_skeleton, n_bounces, tcounter, size_increment, A, d_out, xi_mu, xi_m2
end

function compress_xi(outp::projopf, xi)
   return outp.A * xi  
end


mutable struct maxa_opt <:outputtimer
    running::Bool
    max_attempts::Int64
    acounter::Int64
end
maxa_opt(max_attempts) = maxa_opt(true, max_attempts, 1)

function eval_stopping(opf::maxa_opt, xi, time, bounce)
    opf.acounter+=1
    if opf.acounter >= opf.max_attempts
        opf.running = false
    end
    return opf
end


#--------------------------------------------------------------------------------------------------------
# ----------------------------------------------- MODELS ------------------------------------------------
#--------------------------------------------------------------------------------------------------------
mutable struct model
    ll::ll_model
    pr::prior_model
end

function log_likelihood_vec(ll::ll_logistic_sp, ξ, mb)
   return - ( log.(1 + vec(exp.(ξ'll.X[:,mb]))) - ll.y[mb] .* vec(ξ'll.X[:,mb]) )
end

function partial_derivative_vec(ll::ll_logistic_sp, ξ, k, mb) 
    mb_size = length(mb)
    nz_ind = ll.X[k,mb].nzind
    pd_vec = spzeros(mb_size)
    mb_nz_ind = mb[nz_ind]
    pd_vec[nz_ind] = ll.X[k,mb_nz_ind].* ( vec(exp.(ξ'll.X[:,mb_nz_ind]) ./ 
                                         (1+exp.(ξ'll.X[:,mb_nz_ind]))) - ll.y[mb_nz_ind] )
    return pd_vec
end


# --------------------------------------------------------------------------------------------------------
# Derivative for model = prior + likelihood 
# --------------------------------------------------------------------------------------------------------

function log_posterior(m::model, ξ) 
    return log_likelihood(m.ll, ξ) + log_prior(m.pr, ξ) 
end

function gradient(m::model, ξ) 
    return gradient(m.ll, ξ) + gradient(m.pr, ξ) 
end

function partial_derivative_vec(m::model, ξ, k, mb) 
    return partial_derivative_vec(m.ll, ξ, k, mb) + partial_derivative(m.pr, ξ, k)/m.ll.Nobs
end

function partial_derivative(m::model, ξ, k) 
    Nobs = length(m.ll.y)
    return sum(partial_derivative_vec(m, ξ, k, 1:Nobs))
end


log_likelihood(ll::ll_model, ξ) = sum(log_likelihood_vec(ll, ξ, 1:ll.Nobs))

function gradient(ll::ll_model, ξ) 
    d = length(ξ)
    return [sum(partial_derivative_vec(ll::ll_model, ξ, k, 1:ll.Nobs)) for k in 1:d]
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::mbsampler)
    return ll.Nobs*sum(partial_derivative_vec(ll, ξ, k, mb).*get_ubf(gs,mb))
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::sampler_list)
    return estimate_ll_partial(ll, ξ, k, mb, gs.mbs[k])
end

function estimate_ll_partial(ll::ll_model, ξ, k, mb, gs::cvmbsampler_list)
    return gs.gradient_root_sum[k] +  ll.Nobs*sum((partial_derivative_vec(ll, ξ, k, mb) 
                                             - gs.gradient_root_vec[k,mb]).*get_ubf(gs.mbs[k],mb))
end

# for likelihood without control variates:
function estimate_rate(ll::ll_logistic_sp, mstate::zz_state, i0, mb, gs::mbsampler_list)
    rate_ll = pos(mstate.θ[i0]*mstate.α[i0]*estimate_ll_partial(ll, mstate.ξ, i0, mb, gs.mbs[i0]))
    return rate_ll
end

# for likelihood with control variates
function estimate_rate(ll::ll_logistic_sp, mstate::zz_state, i0, mb, gs::cvmbsampler_list)
    rate_1 = pos(mstate.θ[i0]*mstate.α[i0]*gs.gradient_log_ll_root_sum[i0])
    rate_2 = pos(mstate.θ[i0]*mstate.α[i0]*ll.Nobs*sum((partial_derivative_vec(ll, mstate.ξ, i0, mb) 
                                      - gs.gradient_log_ll_root_vec[i0,mb]).*get_ubf(gs.mbs[i0],mb)) )
    return rate_1 + rate_2
end

# for prior
function estimate_rate(pr::gaussian_prior, mstate::zz_state, i0)
    rate_pr = pos(mstate.θ[i0]*mstate.α[i0]*partial_derivative(pr, mstate.ξ, i0))
    return rate_pr
end

#--------------------------------------------------------------------------------------------------------
# ------------------------------------------------ BOUNDS -----------------------------------------------
#--------------------------------------------------------------------------------------------------------

linear_bound(ll::ll_model, pr::gaussian_prior, gs_list::sampler_list) = 
linear_bound(build_linear_bound(ll, pr, gs_list), zeros(size(ll.X,1)), zeros(size(ll.X,1))) 

evaluate_bound(bb::linear_bound, t, k) = bb.a[k] + t*bb.b[k]
evaluate_bound(bb::const_bound, t, k)  = bb.a[k]

pos(x::Float64) = return max.(x, 0.)
pos(x::Int64) = max.(x, 0)
pos(x::Array{Float64}) = [pos(x[i]) for i in 1:length(x)]
pos(x::Array{Int64}) = [pos(x[i]) for i in 1:length(x)]

# ------------------------------------------ Bounds functions ------------------------------------------

# Build linear bound for prior
function build_bound(pr::gaussian_prior)
    d = length(get_σ2(pr))
    return linear_bound(zeros(d), zeros(d), zeros(d))
end

# update linear bound for prior
function update_bound(bb_prior::linear_bound, pr::gaussian_prior, mstate::zz_state)
    d = size(mstate.ξ)
    bb_prior.a = abs.(mstate.ξ-get_μ(pr))./get_σ2(pr)
    bb_prior.b = 1./get_σ2(pr)
end

# build constant bound for likelihood not using control variates
function build_bound(ll::ll_logistic_sp, gs::mbsampler_list)
    d, Nobs = size(ll.X)
    a = zeros(d)
    for i in 1:d
        nz_ind = ll.X[i,:].nzind
        a[i] = maximum(abs.(ll.X[i,nz_ind]./get_weights(gs.mbs[i],nz_ind)))
    end
    return const_bound(a)
end

# update constant bound for likelihood not using control variates
function update_bound(bb_ll::const_bound, ll::ll_logistic_sp, mstate::zz_state, gs::mbsampler_list)
    #do nothing
end

# build linear bound for likelihood while using control variates
function build_bound(ll::ll_logistic_sp, gs::cvmbsampler_list)
    d, Nobs = size(ll.X)
    C_lipschitz = spzeros(Nobs)
    const_ = zeros(d)
    normXj = [norm(ll.X[:,j]) for j in 1:Nobs]
    for i in 1:d 
        nz_ind = ll.X[i,:].nzind
        C_lipschitz[nz_ind] = abs.(ll.X[i,nz_ind]).*normXj[nz_ind]/4
        const_[i] = maximum(C_lipschitz[nz_ind]./get_weights(gs.mbs[i], nz_ind))
    end
    return linear_bound(const_, zeros(d), zeros(d))
end

# update linear bound for likelihood while using control variates
function update_bound(bb_ll::linear_bound, ll::ll_logistic_sp, mstate::zz_state, gs::cvmbsampler_list)
    norm_ = norm(gs.root-mstate.ξ)
    bb_ll.a = pos(mstate.θ.*mstate.α.*gs.gradient_log_ll_root_sum) + mstate.α.*bb_ll.const_.*norm_
    bb_ll.b = mstate.α*norm(mstate.α).*bb_ll.const_
end


#-------------------------------------- Derivative for prior -----------------------------------------
        
function gradient(pr::prior_model, ξ) 
    d, = length(ξ)
    return [partial_derivative(pr, ξ, k) for k in 1:d]
end

function log_prior(pr::gaussian_prior, ξ) 
    return -0.5*sum((ξ - get_μ(pr)).^2 ./ get_σ2(pr))
end

function partial_derivative(pr::gaussian_prior, ξ, k) 
    return (ξ[k] - get_μ(pr)[k])./(get_σ2(pr)[k])
end

# ------------------------- Structure implementing Gaussian non-hierarchical prior ------------------------

struct gaussian_prior_nh <:gaussian_prior
    μ::Array{Float64}
    σ2::Array{Float64}
end

gaussian_prior_nh(d, σ2) = gaussian_prior_nh(zeros(d), σ2*ones(d))

get_σ2(prior::gaussian_prior) = prior.σ2
get_μ(prior::gaussian_prior) = prior.μ


# ---------------------------------- EVENT TIMES FOR POISSON PROCESS ----------------------------------

# For linear bounds
function get_event_time(ai::Float64, bi::Float64)     
    # this assumed that bi is non-negative
    if bi > 0 
        u = rand()
        if ai >= 0 
            return (-ai + sqrt(ai^2 - 2*bi*log(u))) / bi
        else
            return -ai/bi + sqrt(-2*log(u)/bi)
        end
    elseif bi == 0
        return rand(Exponential(1/ai))
    else 
        print("Error, slope is negative \n")
    end
end

get_event_times(bb::linear_bound) = [get_event_time(bb.a[i],bb.b[i]) for i in 1:length(bb.a)]
get_event_times(bb::const_bound)  = [get_event_time(bb.a[i],0.)      for i in 1:length(bb.a)]


function mbs_estimate(gw::mbsampler, f, x)
    mb = gsample(gw)
    return  sum(gw.ubf[mb].*map(f,(x[mb])))
end

## --------------------------------------------------------------------------------------------------
## ----------------------------------- UPDATE STEPS FOR PARAMETERS ----------------------------------
## --------------------------------------------------------------------------------------------------

function get_event_times(mysampler::zz_sampler_decoupled, mstate::zz_state, model::model)
    d = size(model.ll.X,1)
    update_bound(mysampler.bb_pr, model.pr, mstate)
    update_bound(mysampler.bb_ll, model.ll, mstate, mysampler.gs)
    τ_pr, i0_pr = findmin(get_event_times(mysampler.bb_pr)) 
    τ_ll, i0_ll = findmin(get_event_times(mysampler.bb_ll)) 
    return τ_pr, τ_ll, i0_pr, i0_ll
end

function evolve_path(mysampler::zz_sampler_decoupled, mstate::zz_state, τ)
    mstate.ξ += τ*mstate.θ.*mstate.α
    mstate.T += τ
end

function update_state(mysampler::zz_sampler_decoupled, mstate::zz_state, model::model, τ_pr, τ_ll, i0_pr, i0_ll)
    bounce = false
    alpha = 0.
    
    if τ_ll <= τ_pr
        mysampler.i0 = i0_ll
        # estimate rate using a mini-batch
        mb = gsample(mysampler.gs.mbs[mysampler.i0])
        rate_estimated = estimate_rate(model.ll, mstate, mysampler.i0, mb, mysampler.gs)
        alpha = rate_estimated/evaluate_bound(mysampler.bb_ll, τ_ll, mysampler.i0)
    elseif τ_pr < τ_ll
        mysampler.i0 = i0_pr
        # calculate rate from prior
        rate = estimate_rate(model.pr, mstate, mysampler.i0)
        alpha = rate/evaluate_bound(mysampler.bb_pr, τ_pr, mysampler.i0)
    end
    if alpha > 1+1e-10
        print("alpha: ", alpha, "\n")
    elseif rand() < alpha        
        mstate.θ[mysampler.i0] *= -1
        bounce = true
        mstate.n_bounces[mysampler.i0] += 1
    end 
    return bounce
end

## --------------------------------------------------------------------------------------------------
## ----------------------------------------- ZIG-ZAG SAMPLER ----------------------------------------
## --------------------------------------------------------------------------------------------------

function ZZ_sample_decoupled(model::model, outp::outputscheduler, mysampler::zz_sampler_decoupled, mstate::zz_state, Print=true)
    
    d, Nobs = size(model.ll.X) 
    t = copy(outp.opf.bt_skeleton[outp.opf.tcounter])
    counter = 1
    
    # run sampler:
    bounce = false
    start = time()
    while(is_running(outp.opt))
        τ_pr, τ_ll, i0_pr, i0_ll = get_event_times(mysampler, mstate, model)
        τ = min(τ_pr, τ_ll)
        t += τ
        evolve_path(mysampler, mstate, τ)
        bounce = update_state(mysampler, mstate, model, τ_pr, τ_ll, i0_pr, i0_ll)
        outp = feed(outp, mstate, t, bounce)
        counter += 1
        if counter%10_000 == 0 gc() end
        if Print && counter%(outp.opt.max_attempts/10) == 0 
            print(Int64(100*counter/(outp.opt.max_attempts)), "% attempts in ", round((time()-start)/60, 2), " mins \n")
        end
    end
    finalize(outp.opf)
    return outp
end

#--------------------------------------------------------------------------------------------------------
# Other stuff: 
#--------------------------------------------------------------------------------------------------------

function extract_samples(skeleton_points, bouncing_times, h, interpolation="linear") 
    d, n = size(skeleton_points)
    path_length = bouncing_times[end] - bouncing_times[1]
    n_samples = Int64(floor(path_length/h)) + 1
    samples = zeros(d, n_samples+1)
    samples[:,1] = skeleton_points[:,1] 
    sample_index = 2
    time_location = bouncing_times[1] + h
    
    for i in 1:(n-1)
        start, stop = skeleton_points[:,i], skeleton_points[:,i+1] 
        Δ_pos = stop - start   
        Δ_T = bouncing_times[i+1] - bouncing_times[i]
        while time_location <= bouncing_times[i+1]
            if interpolation == "linear"
                samples[:,sample_index] = start + Δ_pos/Δ_T*(time_location - bouncing_times[i])
            elseif interpolation == "constant"
                samples[:,sample_index] = start
            end
            time_location += h
            sample_index += 1
        end
    end
    return samples
end

function compute_configT(m::model, samples::Array{Float64}, k)
    d, Nobs = size(X) 
    n_samples = size(samples,2)
    configT = 0.0
    for i in 1:n_samples
        configT += samples[k,i]*partial_derivative(m::model, samples[:,i], k)
    end
    return configT/n_samples
end


function find_root(my_model::model, ξ_0)
    d, Nobs = size(my_model.ll.X)
    function gradient!(F, ξ)
        F[:] = gradient(my_model, ξ) 
    end
    neg_log_posterior(ξ) = - log_posterior(my_model, ξ)  
    result = optimize(neg_log_posterior, gradient!, ξ_0, LBFGS())
    root = result.minimizer
    return root
end


"
Stochastic gradient descent for finding root.
"
function stochastic_gradient(m::model, ξ, batch_size) 
    d = length(ξ)
    # pick random minibatch 
    mb = Int.(floor.(my_model.ll.Nobs*rand(batch_size)))+1
    return [(m.ll.Nobs*mean(partial_derivative_vec(m.ll, ξ_0, k, mb)) 
             + partial_derivative(m.pr, ξ_0, k)) for k in 1:d]
end

function SGD(m::model, ξ_0, batch_size, γ, tol) 
    d = length(ξ_0) 
    ξ_current = zeros(d)
    ξ_updated = copy(ξ_0)
    @showprogress for iter in 1:10^4  
        ξ_updated = ξ_current - γ*stochastic_gradient(m, ξ_current, batch_size)
        if norm(ξ_updated-ξ_current) < tol 
            @printf("converged in %f iterations", iter)
            break;
        else 
            ξ_current = copy(ξ_updated)
        end
    end
    return ξ_current
end


function acf(x, maxlag)
    n = size(x)[1]
    acf_vec = zeros(maxlag+1)
    xmean = mean(x)
    for lag in 0:maxlag
        index, index_shifted = 1:(n-lag), (lag+1):n
        acf_vec[lag+1] = mean((x[index]-xmean).*(x[index_shifted]-xmean))
    end
    acf_vec/var(x)
end


getBytes(x::DataType) = sizeof(x);

function getBytes(x)
   total = 0;
   fieldNames = fieldnames(typeof(x));
   if fieldNames == []
      return sizeof(x);
   else
     for fieldName in fieldNames
        total += getBytes(getfield(x,fieldName));
     end
     return total;
   end
end



function compute_ESS(opf::outputformater, B::Int64) 
    dim = size(opf.xi_skeleton,1)
    T = opf.bt_skeleton[1,opf.tcounter]
    
    batch_length = T/B
    Y = zeros(B, dim)
    t = opf.bt_skeleton[1,1]
    xi = opf.xi_skeleton[:,1]
    vel = opf.theta_skeleton[:,1]
    
    k = 1 #counter for skeleton point
    
    for i in 1:(B-1)
        while t < i*T/B 
            next_bounce_time = min(opf.bt_skeleton[1,k+1], i*T/B)
            Δt = next_bounce_time - t
            Y[i,:] += xi*Δt + vel.*Δt.^2/2
            t += Δt 
            if next_bounce_time == opf.bt_skeleton[1,k+1] 
                xi = opf.xi_skeleton[1,k+1] 
                vel = opf.theta_skeleton[1,k+1]
                k += 1
            else 
                xi += vel.*Δt
            end
        end
    end
    Y *= sqrt(B/T)
    
    var1 = opf.xi_m2 - opf.xi_mu.^2
    var2 = zeros(dim)
    for i in 1:dim 
        var2[i] = var(Y[:,dim])
    end
    ESS = T*var1./var2
end



