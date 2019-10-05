using Distributions
using StatsBase

function double_derivative_full(X, ξ, k, l, σ_β=10) # double derivative for full dataset
    d, Nobs = size(X)
    ξ = reshape(ξ,1,d)
    sum(X[l,:].*X[k,:].*reshape(exp.(ξ*X)./square.(1+exp.(ξ*X)), Nobs,)) + (k==l)/σ_β^2
end

# CLUSTERING:

function radius(cluster, center) 
    cluster_size = size(cluster)[2]
    radius = norm(cluster[:,1]-center)
    for i in 2:cluster_size
        if norm(cluster[:,i]-center) > radius 
            radius = norm(cluster[:,i]-center)
        end
    end
    return radius
end

function computational_bounds_clustering(clusters, centers) 
    d = size(clusters[1])[1] - 1
    n_clusters = length(clusters) 
    bounds = zeros(d)
    for h in 1:n_clusters
        cluster_size = size(clusters[h])[2]
        cluster, center = clusters[h][1:d,:], centers[h][1:d]
        for i in 1:d 
            bounds[i] += 2*cluster_size*radius(cluster, center) + 4*cluster_size*abs(center[i])
        end
    end
    return bounds
end   


# AUTOCORRELATION FUNCTION 

# Geyer's method 
# section 3.3 of [Geyer, Charles J. "Practical Markov chain Monte Carlo." Statistical science (1992)]
function acf(x, maxlag)
    n = size(x)[1]
    acf_vec = zeros(maxlag)
    xmean = mean(x)
    for lag in 1:maxlag
        index, index_shifted = 1:(n-lag), (lag+1):n
        acf_vec[lag] = mean((x[index]-xmean).*(x[index_shifted]-xmean))
    end
    acf_vec/var(x)
end
                
function Geyer_IACT(x) 
    acf_vec = acf(x, size(x)[1]-1) 
    n = size(acf_vec)[1]
    n -= (1-n%2)
    m = findfirst(vcat(1,acf_vec[2(1:Int(floor(n/2)))]) + acf_vec[2(0:Int(floor(n/2))) + 1] .< 0, 1) - 1
    1+sum(acf_vec[1:2m-1])
end

