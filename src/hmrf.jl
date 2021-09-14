@with_kw mutable struct MRFBatch{T<:Real}
    K::Int
    X::AbstractArray{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    μ::AbstractArray{T} = X' * R
    Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6))]
    ω::T = 10f0 # penalty rate
end

@with_kw mutable struct PairedMRFBatch{T<:Real}
    K::Int
    XH::AbstractArray{T}
    XL::AbstractArray{T}
    adj::AbstractArray
    U::AbstractArray{T} = zeros(T, dh, dl) + I
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(XL, dims=1)[:])
    nh::Int = size(XH, 1)
    dh::Int = size(XH, 2)
    nl::Int = size(XL, 1)
    dl::Int = size(XL, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    μ::AbstractArray{T} = XL' * R
    ΣH::AbstractArray = [cholesky!(Hermitian(cov(XH) + I * 1f-6))]
    ΣL::AbstractArray = [cholesky!(Hermitian(cov(XL) + I * 1f-6))]
    ω::T = 10f0 # penalty rate
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
    μ::AbstractArray{T} = X' * R
    Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6))]
    ω::T = 10f0 # penalty rate
end

@with_kw mutable struct PairedMRFBatchSeg{T<:Real}
    K::Int
    XH::AbstractArray{T}
    XL::AbstractArray{T}
    adj::AbstractArray
    U::AbstractArray{T} = zeros(T, dh, dl) + I
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(XL, dims=1)[:])
    nh::Int = size(XH, 1)
    dh::Int = size(XH, 2)
    nl::Int = size(XL, 1)
    dl::Int = size(XL, 2) 
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    seg::AbstractArray{Int} = Flux.onecold(R', 1:K)
    μ::AbstractArray{T} = XL' * R
    ΣH::AbstractArray = [cholesky!(Hermitian(cov(XH) + I * 1f-6))]
    ΣL::AbstractArray = [cholesky!(Hermitian(cov(XL) + I * 1f-6))]
    ω::T = 10f0 # penalty rate
end

"""
Interface - to be changed
"""
function MrfMixGauss(X::AbstractArray{T}, adj::AbstractArray, K::Int, ω::T=convert(T, 10f0); 
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    n == length(adj) || throw(DimensionMismatch("Dimensions of X and adj mismatch."))
    # init R
    R = fill(1f0/K, n, K)
    # create model struct
    index = findall(x -> x>1f-3, std(X, dims=1)[:])
    model = MRFBatch(X=X, K=K, index=index, adj=adj, R=R, n=n, d=d, ω=ω)
    MrfMixGauss!(model; tol=tol, maxiter=maxiter)
    model
end

"""
Fusion of HMRF-GMM
"""
function MrfMixGauss(
    XH::AbstractArray{T}, XL::AbstractArray{T}, adj::AbstractArray, K::Int, 
    R::AbstractArray{T}=fill(1f0/K, size(XH, 1), K), ω::T=convert(T, 10f0); 
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    nh, dh = size(XH)
    nl, dl = size(XL)
    nh == nl == length(adj) || throw(DimensionMismatch("Dimensions of X and adj mismatch."))
    # create model struct
    # initial U
    # low qaulity mean
    μ = Matrix{T}(undef, d, K)
    for k ∈ 1:K
        μk = view(μ, :, k)
        mean!(μk, XL[findall(x -> x == k, a), :]')
    end
    @debug "μ" μ
    U = Matrix{T}(undef, dh, dl)
    updateU!(U, XH, R, μ, K)
    model = PairedMRFBatch(XH=XH, XL=XL, U=U, K=K, adj=adj, R=R, nh=nh, dh=dh, nl=nl, dl=dl, ω=ω, μ=μ)
    MrfMixGauss!(model; tol=tol, maxiter=maxiter)
    model
end

"""
hard parcellation in clique energy
"""
function MrfMixGauss!(
    modelAll::AbstractVector{Union{MRFBatchSeg{T}, MRFBatch{T}, PairedMRFBatchSeg{T}, PairedMRFBatch{T}}};
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Markov Random Field Gaussian mixture...", dt=0.1, spinner=true)
    iter = 1
    while iter < maxiter
        iter += 1
        L[iter] = modelAll |> eachrow |> Map(x->batch(x)) |> Broadcasting() |> Folds.sum
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
    iter == maxiter || @warn "Not converged after $(maxiter) steps."
    return model
end

function batch(model::MRFBatchSeg{T}) where T <: Real
    Xo = deepcopy(model.X)
    # M step
    maximise!(model, Xo)
    # E step
    segment!(model)
    expect!(model, Xo)
end

function batch(model::MRFBatch{T}) where T <: Real
    Xo = deepcopy(model.X)
    # M step
    maximise!(model, Xo)
    # E step
    expect!(model, Xo)
end

function batch(model::PairedMRFBatch{T}) where T <: Real
    XHo = deepcopy(model.XH)
    XLo = deepcopy(model.XL)
    # M step
    maximise!(model, XHo, XLo)
    # E step
    expect!(model, Xo)
end

function batch(model::PairedMRFBatchSeg{T}) where T <: Real
    XHo = deepcopy(model.XH)
    XLo = deepcopy(model.XL)
    # M step
    maximise!(model, XHo, XLo)
    # E step
    segment!(model)
    expect!(model, Xo)
end

function maximise!(
    model::Union{MRFBatchSeg{T}, MRFBatch{T}}, Xo::AbstractArray{T}
) where T <: Real
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

function maximise!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, 
    XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    # posterior parameters
    sum!(model.nk, model.R')
    @debug "model.R" model.R
    # update μ
    updateμ!(model, XHo, XLo)
    updateU!(model, XHo)
    updateΣ!(model, XHo, XLo)
end

function updateμ!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    @inbounds for k ∈ model.K
        μk = view(model.μ, :, k)
        Rk = view(model.R, :, k)
        mul!(XHo, model.XH, U')
        copyto!(XLo, model.XL)
        rdiv!(XHo, model.ΣH[k])
        rdiv!(XLo, model.ΣL[k])
        mul!(μk, transpose(XHo + XLo), Rk)
        ldiv!(cholesky!(LinearAlgebra.inv!(ΣH[k]) + LinearAlgebra.inv!(ΣL[k])), μk)
    end
    model.μ ./= model.nk'
end

function updateU!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, Xo::AbstractArray{T}
) where T <: Real
    fill!(model.U, 0)
    @inbounds for k ∈ model.K
        Rk = view(Rk, :, k)
        copyto!(Xo, model.XH)
        Xo .*= sqrt.(Rk)
        U .+= transpose(Xo) * Xo
    end
    u, _, v = svd!(μ * transpose(R) * model.XH * U)
    mul!(U, u, v')
end

function updateΣ!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    @inbounds for k ∈ model.K
        μk = view(model.μ, :, k)
        Rk = view(Rk, :, k)
        mul!(XHo, model.XH, U')
        XHo .-= μk'
        copyto!(XLo, model.XL)
        XLo .-= μk'
        map([XHo, XLo]) do x
            x .*= sqrt.(Rk)
        end
        model.ΣH[k] = cholesky!(Hermitian(I * 1f-6 + (XHo' * XHo) ./ model.nk[k]))
        model.ΣL[k] = cholesky!(Hermitian(I * 1f-6 + (XLo' * XLo) ./ model.nk[k]))
    end
end

function expect!(model::Union{MRFBatchSeg{T}, MRFBatch{T}}, Xo::AbstractArray{T}) where T<:Real
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
        logPrior!(Rk, model, k)
        @debug "Rk" Rk
    end
    Flux.softmax!(model.R, dims=2)
    return logsumexp(model.R) / model.n
end

function expect!(
    model::Union{PairedMRFBatch{T}, PairedMRFBatchSeg{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        mul!(XHo, model.XH, U')
        copyto!(XLo, model.XL)
        # demean
        map([XHo, XLo]) do x
            x .-= μk'
        end
        copyto!(Rk, diag((XHo / model.ΣH[k]) * XHo' .+ (XLo / model.ΣL[k]) * XLo'))
        Rk .+= logdet(model.ΣH[k]) .+ logdet(model.ΣL[k])
        Rk .*= -0.5f0
        @debug "Rk" Rk
        logPrior!(Rk, model, k)
        @debug "Rk" Rk
    end
    Flux.softmax!(model.R, dims=2)
    return logsumexp(model.R) / model.n
end

function logPrior!(Rk::AbstractArray{T}, model::Union{MRFBatchSeg{T}, PairedMRFBatchSeg{T}}, k::Int) where T <: Real
    for v ∈ 1:model.n
        Rk[v] -= model.ω * sum([model.seg[idx] != k for idx in model.adj[v]])
    end
end

function logPrior!(Rk::AbstractArray{T}, model::Union{MRFBatch{T}, PairedMRFBatch{T}}, k::Int) where T <: Real
    for v ∈ 1:model.n
        Rk[v] += model.ω * sum([R[idx, k] for idx ∈ model.adj[v]])
    end
end

function segment!(model::Union{MRFBatchSeg{T}, PairedMRFBatchSeg{T}}) where T<:Real
    copyto!(model.seg, Flux.onecold(model.R', 1:model.K))
    model
end
