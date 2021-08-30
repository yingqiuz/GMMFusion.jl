@with_kw mutable struct MRFBatch{T<:Real}
    K::Int
    X::AbstractArray{T}
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    seg::AbstractArray{Int} = onecold(R, 1:K)
    μ::AbstractArray{T}
    Σ::AbstractArray
    ω::T = 1f0 # penalty rate
end

@with_kw mutable struct MRFBatchSeg{T<:Real}
    K::Int
    X::AbstractArray{T}
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    seg::AbstractArray{Int} = onecold(R, 1:K)
    μ::AbstractArray{T}
    Σ::AbstractArray
    ω::T = 1f0 # penalty rate
    seg::AbstractArray{T} = onecold(R', 1:K)
end

"""
Interface - to be changed
"""
function MrfMixGauss(X::AbstractArray{T}, adj::AbstractArray, K::Int; 
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    n == length(adj) || throw(DimensionMismatch("Dimensions of X and adj mismatch."))
    # init R
    R = fill(1f0/K, n, K)
    # create model struct
    model = MRFBatch(X=X, K=K, adj=adj, R=R, n=n, d=d)
    MrfMixGauss!(model; tol=tol, maxiter=maxiter)
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
    iter = 0; incr = NaN32
    while iter < maxiter
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter), (:incr, incr)]
        )
        # M step
        maximise!(model, Xo)
        # E step
        segment!(model)
        expect!(model, Xo)
        if abs(incr) < tol
            return model
        end
        iter += 1
    end
    iter == maxiter || @warn "Not converged after $(iter) steps."
    return model
end

function maximise!(model::Union{MRFBatchSeg{T}, MRFBatch{T}}, Xo::AbstractArray{T}) where T <: Real
    # posterior parameters
    sum!(model.nk, model.R')
    @debug "model.R" model.R
    # update μ
    mul!(model.μ, model.X', R)
    model.μ ./= model.nk'
    # update Σ
    @inbounds for k ∈ 1:model.K
        copy!(Xo, model.X)
        Xo .-= transpose(view(model.μ, :, k))
        Xo .*= sqrt.(view(R, :, k))
        model.Σ[k] = cholesky!(Xo' * Xo ./ model.nk[k] + I * 1f-6)
    end
end

function expect!(model::MRFBatchSeg{T}, Xo::AbstractArray{T}) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        copyto!(Xo, model.X)
        Xo .-= μk'
        copyto!(Rk, diag((Xo / model.Σ[k]) * Xo'))
        Rk .+= logdet(model.Σ[K])
        Rk .*= -0.5f0
        # log prior
        for v ∈ model.n
            Rk[v] -= model.ω * sum([model.seg[idx] != k for idx in model.adj[v]])
        end
    end
    softmax!(model.R, dims=2)
end

function segment!(model::MRFBatchSeg{T}) where T<:Real
    copyto!(model.seg, onecold(model.R, 1:model.K))
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
    iter = 0; incr = NaN32
    while iter < maxiter
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter), (:incr, incr)]
        )
        # M step
        maximise!(model, Xo)
        # E step
        expect!(model, Xo)
        if abs(incr) < tol
            return model
        end
        iter += 1
    end
    iter == maxiter || @warn "Not converged after $(iter) steps."
    return model
end

function expect!(model::MRFBatch{T}, Xo::AbstractArray{T}) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        copyto!(Xo, model.X)
        Xo .-= μk'
        copyto!(Rk, diag((Xo / model.Σ[k]) * Xo'))
        Rk .+= logdet(model.Σ[K])
        Rk .*= -0.5f0
        # log prior
        for v ∈ model.n
            Rk[v] += model.ω * sum([model.R[idx, k] for idx ∈ model.adj[v]])
        end
    end
    softmax!(model.R, dims=2)
end

