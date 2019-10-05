using Distributions, TimeIt, ProgressMeter, PyPlot
include("zz_structures.jl")
include("mbsampler.jl")


#prob_het = 0.95

# Stratified sub-sampling with CV and with weights

function sampler_list(my_model, root, mb_size, stratified, CV, weighted, prob_het=0.95) 
    X, y = my_model.ll.X, my_model.ll.y
    d, Nobs = size(X)
    
    norm_Xj = [norm(X[:,j]) for j in 1:Nobs]
    
    if !stratified 
        mbs = Array{mbsampler}(d)
        #mbs[1] = umbsampler(Nobs, mb_size)
        for i in 1:d 
            nz_ind = X[i,:].nzind
            weights_het = spzeros(Nobs)
            if weighted 
                weights_het[nz_ind] = abs.(X[i,nz_ind]).*(!CV + CV*norm_Xj[nz_ind])
                if length(nz_ind) < length(X[i,:]) 
                    mbs[i] = spwumbsampler(Nobs, mb_size, weights_het, prob_het)
                else 
                    mbs[i] = wumbsampler(Nobs, mb_size, full(weights_het))
                end
            else
                mbs[i] = umbsampler(Nobs, mb_size)
            end
        end
        if CV 
            gs_list = cvmbsampler_list(my_model, mbs, root, true)
        else
            gs_list = mbsampler_list(d, mbs)
        end 
    else
        cmbsamplers = Array{mbsampler}(d)
        cmbsamplers[1] = umbsampler(Nobs, mb_size)
        N_cluster = mb_size
        
        if CV 
            weights = spzeros(d, Nobs)
            for i in 1:d 
                nz_ind = X[i,:].nzind
                weights[i,nz_ind] = abs.(X[i,nz_ind]).*norm_Xj[nz_ind] 
                weights[i,nz_ind] /= sum(weights[i,nz_ind])
            end
        else
            weights = sparse(abs.(X))
            for i in 1:d 
                weights[i,:] /= sum(weights[i,:])
            end
        end
        
        ll_pd_root_list = [partial_derivative_vec(my_model.ll, root, i, 1:Nobs) for i in 1:d]

        for dim in 1:d
            N_cluster_d = min(N_cluster, length(ll_pd_root_list[dim].nzval))
            csamplers = Array{mbsampler}(N_cluster_d)
            clusters = get_clustering(N_cluster_d, ll_pd_root_list[dim].nzval)
            for i in 1:N_cluster_d
                clusters[i] =  ll_pd_root_list[dim].nzind[clusters[i]]
            end
            scp = ones(N_cluster_d)
            for (ci, c) in enumerate(clusters)
                if weighted 
                    csamplers[ci] = wumbsampler(size(c)[1], scp[ci], weights[dim,c])
                    elseif !weighted
                    csamplers[ci] = wumbsampler(size(c)[1], scp[ci], ones(length(c))/length(c))
                end
            end
            if weighted 
                cmbsamplers[dim] = spcmbsampler(csamplers, clusters, weights[dim,:])
            elseif !weighted
                cmbsamplers[dim] = spcmbsampler(csamplers, clusters, ones(Nobs)/Nobs)
            end
        end
        if CV 
            gs_list = cvmbsampler_list(my_model, cmbsamplers, root, true)
        else
            gs_list = mbsampler_list(d, cmbsamplers)
        end 
    end
    return gs_list 
end
            
            

function run_sampler(gs_list, my_model, max_attempts, adapt_speed="none")

    d, Nobs = size(my_model.ll.X)
    A = eye(d)
    opf = projopf(A, 1000, hyperparam_size(my_model.pr))
    opt = maxa_opt(max_attempts)
    outp = outputscheduler(opf,opt)
    mstate = zz_state(d)
    
    bb_ll = build_bound(my_model.ll, gs_list)
    bb_pr = build_bound(my_model.pr)
    update_bound(bb_pr, my_model.pr, mstate)
    update_bound(bb_ll, my_model.ll, mstate, gs_list)
    
    L = 2
    my_zz_sampler = zz_sampler_decoupled(0, gs_list, bb_pr, bb_ll, L, adapt_speed)
    ZZ_sample_decoupled(my_model, outp, my_zz_sampler, mstate);
    
    bb_pr, bb_ll, mstate = nothing, nothing, nothing 
    gc()
    
    return outp.opf
end 

function plot_ACFS(xi_samples, maxlag) 
    d_out = size(xi_samples,1);
    maxlag = 200
    acfs = zeros(d_out, maxlag)
    for dim in 1:d_out 
        acfs[dim,:] = acf(xi_samples[dim,:], maxlag)
    end

    acfs_toplot = []
    xt = []
    for i in 1:Int(maxlag/20) 
        push!(acfs_toplot, acfs[:,i*20])
        if i%20 == 0 
            push!(xt, 20*i)
        else 
            push!(xt, "")
        end
    end
    boxplot(acfs_toplot, showfliers=false)
    grid(alpha=0.35)
    xlabel("Lag", fontsize=10)
    ylabel("ACF", fontsize=10)
    xticks(1:Int(maxlag/20) , xt);
end



function get_acfs(samples, maxlag) 
    ACFs = []
    xt = nothing
    for k in 1:length(samples) 
        xi_samples = samples[k]
        d_out = size(xi_samples,1)
        acfs = zeros(d_out, maxlag)
        for dim in 1:d_out 
            acfs[dim,:] = acf(xi_samples[dim,1:end-2], maxlag)
        end
        acfs_toplot = []
        for i in 1:Int(maxlag/50) 
            push!(acfs_toplot, acfs[:,i*50])
        end
        push!(ACFs, acfs_toplot) 
    end
    xt = []
    for i in 1:Int(maxlag/50) 
        if i%10 == 0 
            push!(xt, 50*i)
        else 
            push!(xt, "")
        end
    end
    return ACFs, xt
end

function generate_model(d, Nobs, pX, cov_dist="Gaussian", prior_sigma=1., intercept=true) 
    X = sprand(d, Nobs, pX)
    
    for i in 1:d 
        while length(X[i,:].nzind) == 0 
            X[i,:] = sprandn(Nobs,pX) 
        end
    end
    
    for i in 1:d 
        nzind = X[i,:].nzind
        if cov_dist == "uniform" 
            X[i,nzind] = rand(length(nzind))
        elseif cov_dist == "Laplace" 
            X[i,nzind] = rand(Laplace(0,1),length(nzind)) 
        end
    end
    
    if intercept 
        X[1,:] = ones(Nobs)
    end
    xi = randn(d) 
    y = float.([rand(Binomial(1,1/(1+exp(-X[:,i]'xi)))) for i in 1:Nobs])
    while (sum(y)==0) || (sum(y)==Nobs) 
        y = float.([rand(Binomial(1,1/(1+exp(-X[:,i]'xi)))) for i in 1:Nobs])
    end
    
    prior = gaussian_prior_nh(zeros(d), prior_sigma*ones(d))

    my_ll = ll_logistic_sp(X,y);
    my_model = model(my_ll, prior)
    root = find_root(my_model, rand(d))
    return my_model, root
end

function run_mbsamplers(my_model, root, mb_size, max_attempts, pX, Nobs, varying, rep, cov_dist, include_CV=false, include_stratified=false, adapt_speed="none")
    opfs = []
    wt = [true]
    sr, cv = [false], [false]
    if include_CV 
        cv = [true,false]
    end
    if include_stratified 
        sr = [true, false]
    end
    
    subfolder = "scaling_"*varying
    d = size(my_model.ll.X)[1]
    
    for stratified in sr
        for CV in cv
            for weighted in wt
                start = time()
                gs_list = sampler_list(my_model, root, mb_size, stratified, CV, weighted)
                opf = run_sampler(gs_list, my_model, max_attempts, adapt_speed)
                if varying == "pX" 
                    filename  = "/xtmp/PDMP_data_revision/"*subfolder*"/cov_dist:"*cov_dist*"-pX:"*string(pX)*"-rep:"*string(rep)*"-stratified:"*string(stratified)*"-CV:"*string(CV)*"-weighted:"*string(weighted)*"-d:"*string(d)*".jld"
                elseif varying == "Nobs" 
                    filename  = "/xtmp/PDMP_data_revision/"*subfolder*"/cov_dist:"*cov_dist*"-Nobs:"*string(Nobs)*"-rep:"*string(rep)*"-stratified:"*string(stratified)*"-CV:"*string(CV)*"-weighted:"*string(weighted)*"-d:"*string(d)*".jld"
                end
                save(filename, "xt_skeleton", opf.xi_skeleton, "bt_skeleton", opf.bt_skeleton)
                gs_list, opf = nothing, nothing
                gc()
                if varying == "pX"
                    print("Rep = ", rep, "; pX = ", pX, "; stratified, CV, weighted = ", stratified, " ", CV, " ", weighted, "; time = ", round((time()-start)/60,2), " min \n")
                elseif varying == "Nobs" 
                    print("Rep = ", rep, "; Nobs = ", Nobs, "; stratified, CV, weighted = ", stratified, " ", CV, " ", weighted, "; time = ", round((time()-start)/60,2), " min \n")
                end
            end
        end
    end
    
end

function plot_acfs_all(samples, maxlag, y_lim, include_stratified=false) 
    acfs_toplot, xt = get_acfs(samples, maxlag)
    if include_stratified 
        fig = figure("pyplot_subplot_mixed",figsize=(14,4))
        for i in 1:4 
            for j in 1:2 
                idx = 4*(j-1)+i
                subplot(2,4,i)
                boxplot(acfs_toplot[idx], showfliers=false)
                grid(alpha=0.35)
                if j == 2 
                    xlabel("Lag", fontsize=10)
                end
                if i == 1  
                    if j == 1 
                        ylabel("stratified true", fontsize=10)
                    else 
                        ylabel("stratified false")
                    end
                end

                if j == 1 
                    if i == 1 
                        title("CV true, weighted true")
                    elseif i == 2
                        title("CV true, weighted false")
                    elseif i == 3
                        title("CV false, weighted true")
                    elseif i == 4
                        title("CV false, weighted false")
                    end
                end

                xticks(1:Int(maxlag/10), xt)
                ylim(y_lim)
            end
        end
    else 
        fig = figure("pyplot_subplot_mixed",figsize=(14,2.5))
        for i in 1:4 
            subplot(1,4,i)
                boxplot(acfs_toplot[i], showfliers=false)
                grid(alpha=0.35)
                xlabel("Lag", fontsize=10)
                if i == 1 
                    title("CV true, weighted true")
                elseif i == 2
                    title("CV true, weighted false")
                elseif i == 3
                    title("CV false, weighted true")
                elseif i == 4
                    title("CV false, weighted false")
                end
                xticks(1:Int(maxlag/50), xt)
                ylim(y_lim)
            end
        end
            
            
    suptitle("ACFs", fontsize=14)
end

