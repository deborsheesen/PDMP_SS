using StatsBase, Distributions

# Define abstract type for gaussian prior, sub-types of this abstract types must have attributes mu and sigma2  
abstract type prior end
abstract type gaussian_prior <: prior end
abstract type laplace_prior <: prior end

# returns the rate used in the simulation of the zig-zag proccess
function get_rate(gprior<:gaussian_prior, x, mb)
    gprior.mu 
    gprior.sigma2
end

#-------------------------------------------------
# Structure implementing horseshoe prior
#-------------------------------------------------
struct horse_prior <: gaussian_prior
    d #::Int64
    mu#::Array{Float64}(d)
    sigma2
    #hyper parameters
end

struct davids_prior <: gaussian_prior
    d #::Int64
    mu#::Array{Float64}(d)
    sigma2
    #hyper parameters
end

function block_gibbs_update(prior::horse_prior, beta)
    #gibbs steps here 
    prior.mu = ....
end


