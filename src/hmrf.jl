@with_kw mutable struct MRFBatch{T<:Real}
    K::Int
    X::AbstractArray{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    μ::AbstractArray{T}
    Σ::AbstractArray
    ω::T = 1f0 # penalty rate
end

@with_kw mutable struct MRFBatchSeg{T<:Real}
    K::Int
    X::AbstractArray{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    seg::AbstractArray{Int} = Flux.onecold(R', 1:K)
    μ::AbstractArray{T}
    Σ::AbstractArray
    ω::T = 1f0 # penalty rate
end

"""
Interface - to be changed
"""
function MrfMixGauss(X::AbstractArray{T}, adj::AbstractArray, K::Int, ω::T=convert(T, 1f0); 
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    n == length(adj) || throw(DimensionMismatch("Dimensions of X and adj mismatch."))
    # init R
    R = fill(1f0/K, n, K)
    # create model struct
    index = findall(x -> x>1f-3, std(X, dims=1)[:])
    model = MRFBatch(X=X, K=K, index=index, adj=adj, R=R, n=n, d=d, ω=ω)
    #MrfMixGauss!(model; tol=tol, maxiter=maxiter)
    model
end

"""
hard parcellation in clique energy
"""
function MrfMixGauss!(model::MRFBatchSeg{T}; tol::T=convert(T, 1f-6), maxiter::Int=10000) where T<: Real
    # to store centered data
    Xo = deepcopy(model.X)
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Markov Random Field Gaussian mixture...", dt=0.1, spinner=true)
    iter = 1
    while iter <= maxiter
        iter += 1
        # M step
        maximise!(model, Xo)
        # E step
        segment!(model)
        expect!(model, Xo, L, iter)
        incr = (L[iter] - L[iter-1]) / L[iter-1]
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        if abs(incr) < tol
            ProgressMeter.finish!(prog)
            return model
        end
    end
    ProgressMeter.finish!(prog)
    iter == maxiter+1 || @warn "Not converged after $(maxiter) steps."
    return model
end

function maximise!(model::Union{MRFBatchSeg{T}, MRFBatch{T}}, Xo::AbstractArray{T}) where T <: Real
    # posterior parameters
    sum!(model.nk, model.R')
    @debug "model.R" model.R
    # update μ
    mul!(model.μ, model.X', model.R)
    model.μ ./= model.nk'
    # update Σ
    @inbounds for k ∈ 1:model.K
        copyto!(Xo, model.X)
        Xo .-= transpose(view(model.μ, :, k))
        Xo .*= sqrt.(view(model.R, :, k))
        model.Σ[k] = cholesky!(Hermitian(Xo' * Xo ./ model.nk[k]) + I * 1f-5)
    end
end

function expect!(model::MRFBatchSeg{T}, Xo::AbstractArray{T}, L::AbstractArray{T}, iter::Int) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        copyto!(Xo, model.X)
        Xo .-= μk'
        copyto!(Rk, diag((Xo / model.Σ[k]) * Xo'))
        Rk .+= logdet(model.Σ[k])
        Rk .*= -0.5f0
        @debug "Rk" Rk
        # log prior
        for v ∈ model.n
            @debug sum([model.seg[idx] != k for idx in model.adj[v]])
            Rk[v] -= model.ω * sum([model.seg[idx] != k for idx in model.adj[v]])
        end
        @debug "Rk" Rk
    end
    L[iter] = logsumexp(model.R) / model.n
    Flux.softmax!(model.R, dims=2)
end

function segment!(model::MRFBatchSeg{T}) where T<:Real
    copyto!(model.seg, Flux.onecold(model.R', 1:model.K))
    model
end

"""soft parcellation in clique energy"""
function MrfMixGauss!(model::MRFBatch{T}; 
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T<: Real
    # to store centered data
    Xo = deepcopy(model.X)
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Markov Random Field Gaussian mixture...", dt=0.1, spinner=true)
    iter = 1
    while iter <= maxiter
        iter += 1
        # M step
        maximise!(model, Xo)
        # E step
        expect!(model, Xo, L, iter)
        incr = (L[iter] - L[iter-1]) / L[iter-1]
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        if abs(incr) < tol
            return model
        end
    end
    iter == maxiter+1 || @warn "Not converged after $(maxiter) steps."
    return model
end

function expect!(model::MRFBatch{T}, Xo::AbstractArray{T}, L::AbstractArray{T}, iter::Int) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        copyto!(Xo, model.X)
        Xo .-= μk'
        copyto!(Rk, diag((Xo / model.Σ[k]) * Xo'))
        Rk .+= logdet(model.Σ[k])
        Rk .*= -0.5f0
        @info "Rk" Rk
        # log prior
        for v ∈ model.n
            @debug sum([model.R[idx, k] for idx ∈ model.adj[v]])
            Rk[v] += model.ω * sum([model.R[idx, k] for idx ∈ model.adj[v]])
        end
        @debug "Rk" Rk
    end
    L[iter] = sum(logsumexp(model.R, dims=2)) / model.n
    Flux.softmax!(model.R, dims=2)
end
