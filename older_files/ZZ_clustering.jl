using Distributions, Optim, Clustering 
include("ZZ_subsampling.jl")

function find_clusters_sorted(root, X, y, n_clusters)
    
    d, Nobs = size(X)
    gradient_root = zeros(d, Nobs)
    for n in 1:Nobs 
        gradient_root[:,n] = [derivative(X[:,n], y[n], k, root, Nobs, σ)[1] for k in 1:d]
    end
    
    clusters = zeros(d, Nobs)
    cluster_sizes = zeros(d, 2n_clusters)
    for dim in 1:d
        # split clusters:
        result = kmeans(reshape(gradient_root[dim,:], 1, Nobs), n_clusters)
        cluster_labels = copy(result.assignments)
        sizes = [sum(cluster_labels .== cluster) for cluster in 1:n_clusters]

        mean_y = [mean(y[cluster_labels.==i]) for i in 1:n_clusters]
        tosplit = (1:n_clusters)[(mean_y .> 0) + (mean_y .< 1) .== 2]
        n_tosplit = length(tosplit) 
        
        for i in 1:n_tosplit 
            c_label = tosplit[i] 
            c_indices = (1:Nobs)[cluster_labels .== c_label] 
            labels_0 = c_indices[y[c_indices] .== 0]
            cluster_labels[labels_0] = n_clusters+i 
        end
        
        current_sum = 0
        for cluster in 1:(n_clusters+n_tosplit)
            cluster_indices = (1:Nobs)[cluster_labels .== cluster]
            cluster_size = length(cluster_indices)
            cluster_sizes[dim,cluster] = cluster_size
            
            C = current_sum + (1:cluster_size) 
            clusters[dim, C] = copy(cluster_indices)
            current_sum += cluster_size
        end
    end
    return Int.(clusters), Int.(cluster_sizes)
end

function rate_clustering_sorted(θ, clusters, cluster_sizes, ξ, X, y, σ, k, weighting, ϵ)
    d, Nobs = size(X)
    rate = 0
    current_sum = 0
    n_clusters = sum(cluster_sizes .> 0)
    
    for cluster in 1:n_clusters
        cluster_size = cluster_sizes[cluster]
        C = current_sum + (1:cluster_size)  
        cluster_indices = clusters[C] 
        if weighting 
            weights = abs.(X[k,cluster_indices]) + ϵ
        else 
            weights = ones(cluster_size)
        end
        weights /= sum(weights)
        idx = sample(1:cluster_size, Weights(weights), 1)[1]
        mb = cluster_indices[idx]
        rate += ( (X[k,mb]*(exp.(ξ'X[:,mb])./(1+exp.(ξ'X[:,mb])) - y[mb]))/weights[idx] )[1]
        current_sum += cluster_size
    end
    
    return pos((θ[k]*rate)[1]) + pos(θ[k]*ξ[k]/σ^2) 
end


function form_clusters_kmeans(data, n_clusters) 
    result = kmeans(data, n_clusters)
    clusters = [data[:, result.assignments .== 1]]
    centers = [result.centers[:,1]];
    for h in 2:n_clusters 
        push!(clusters, data[:, result.assignments .== h])
        push!(centers, result.centers[:,h])  
    end
    return clusters, centers
end

function rate_clustering_kmeans(θ, clusters, centers, ξ, k, σ) 
    d = size(clusters[1])[1] - 1
    rate = 0
    for h in 1:length(clusters)
        cluster_size = size(clusters[h])[2]
        idx = sample(1:cluster_size, 1)[1]
        x, y = clusters[h][1:d,idx], clusters[h][d+1,idx]
        rate += derivative(x, y, k, ξ, Nobs, σ)cluster_size
    end    
    return pos((θ[k]*rate)[1]) 
end

function ZZ_clustering(X, y, max_attempts, β_0, n_clusters, root, σ=10, A=nothing, clustering="none", weighting=false, ϵ=1e-10) 

    if A == nothing 
        A = eye(d)
    end
    m = size(A,1)
    
    d, Nobs = size(X) 
    bouncing_times = []
    push!(bouncing_times, 0.)
    skeleton_points = zeros(m, 1000)
    skeleton_points[:,1] = A*copy(β_0)
    ξ = copy(β_0)
    θ = ones(d)
    t, switches, refreshments = 0, 0, 0  
    
    # Form clusters:
    if clustering == "sorted"
        clusters, cluster_sizes = find_clusters_sorted(root, X, y, n_clusters)
    elseif clustering == "kmeans"
        data = [X; reshape(y, 1, size(X)[2])] 
        clusters, centers = form_clusters_kmeans(data, n_clusters)
    elseif clustering == "separated"
        data = [X; reshape(y, 1, size(X)[2])] 
        cluster_1 = (1:Nobs)[y .== 1]
        cluster_2 = (1:Nobs)[y .== 0];
        clusters = [data[:, cluster_1]]
        push!(clusters, data[:, cluster_2])
        centers = 1
    end
    
    # Calculate bounds 
    if weighting == true 
        bounds = sum(abs.(X), 2)
    else
        bounds = Nobs*maximum(abs.(X), 2)
    end

    # run sampler:
    for attempt in 1:max_attempts
        a = bounds + abs.(ξ)/σ^2 
        b = ones(d)/σ^2
        event_times = [get_event_time(a[i], b[i]) for i in 1:d] 
        τ, i0 = findmin(event_times)                
        t += τ 
        ξ_new = ξ + τ*θ
            
        # Find rate:
        if clustering == "sorted"
            rate = rate_clustering_sorted(θ, clusters[i0,:], cluster_sizes[i0,:], 
                                          ξ_new, X, y, σ, i0, weighting, ϵ)
        elseif clustering == "kmeans" || clustering == "separated"
            rate = rate_clustering_kmeans(θ, clusters, centers, ξ_new, i0, σ) 
        else
            rate = rate_iid(ξ_new, X, y, i0, θ, n_clusters, σ, true)
        end    

        bound = a[i0] + b[i0]*τ
        alpha = rate/bound
        if alpha > 1 
            print("Error, rate > bound. \n ")
            break
        elseif rand(1)[1] < alpha
            θ[i0] *= -1
            switches += 1
            skeleton_points[:,switches+1] = A*ξ_new
            push!(bouncing_times, t)
        end   
        if switches == size(skeleton_points,2) - 1 
            skeleton_points = extend_skeleton_points(skeleton_points)
        end
        ξ = copy(ξ_new)
    end
    print(signif(100*switches/max_attempts,2),"% of switches accepted \n")
    return hcat(skeleton_points[:,1:switches+1], A*ξ), push!(bouncing_times, t)
end