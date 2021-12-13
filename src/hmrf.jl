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
    Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    ω::T = convert(eltype(X), 1f0) # penalty rate
    σ::T = convert(eltype(X), 1f0) # length scale
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    kernel::Function = (foo(x, y) = exp( -sum((x .- y) .^ 2)/(2f0 * σ^2) ))
    f::AbstractArray = [[kernel(X[idx, :], X[kk, :]) for idx ∈ el] for (kk, el) ∈ enumerate(adj)]
    E1::AbstractArray{T} = zeros(eltype(X), n, K)
    E2::AbstractArray{T} = ones(eltype(X), n) .* (nk' ./ n)
    llhmap::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, n, K))
end

@with_kw mutable struct PairedMRFBatch{T<:Real}
    K::Int
    XH::AbstractArray{T}
    XL::AbstractArray{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(XL, dims=1)[:])
    n::Int = size(XH, 1)
    dh::Int = size(XH, 2)
    nl::Int = size(XL, 1)
    dl::Int = size(XL, 2)
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    μ::AbstractArray{T} = XL' * R
    ΣH::AbstractArray = [cholesky!(Hermitian(cov(XH) + I * 1f-6)) for k in 1:K]
    ΣL::AbstractArray = [cholesky!(Hermitian(cov(XL) + I * 1f-6)) for k in 1:K]
    U::AbstractArray{T} = qr!(randn(eltype(XH), dh, dh)).Q[:, 1:dl]
    ω::T = convert(eltype(XH), 10f0) # penalty rate
    llh::AbstractArray{T} = convert(Array{eltype(XH)}, fill(-Inf32, 10))
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
    Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    ω::T = convert(eltype(X), 10f0) # penalty rate
    σ::T = convert(eltype(X), 1f0) # length scale
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    kernel::Function = (foo(x, y) = exp( -sum((x .- y) .^ 2)/(2f0 * σ^2) ))
    f::AbstractArray = [[kernel(X[idx, :], X[kk, :]) for idx ∈ el] for (kk, el) ∈ enumerate(adj)]
    E1::AbstractArray{T} = zeros(eltype(X), n, K)
    E2::AbstractArray{T} = ones(eltype(X), n) .* (nk' ./ n)
    llhmap::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, n, K))
end

@with_kw mutable struct PairedMRFBatchSeg{T<:Real}
    K::Int
    XH::AbstractArray{T}
    XL::AbstractArray{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(XL, dims=1)[:])
    n::Int = size(XH, 1)
    dh::Int = size(XH, 2)
    nl::Int = size(XL, 1)
    dl::Int = size(XL, 2) 
    R::AbstractArray{T} = fill(1f0 / K, n, K)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    seg::AbstractArray{Int} = Flux.onecold(R', 1:K)
    μ::AbstractArray{T} = XL' * R
    ΣH::AbstractArray = [cholesky!(Hermitian(cov(XH) + I * 1f-6)) for k in 1:K]
    ΣL::AbstractArray = [cholesky!(Hermitian(cov(XL) + I * 1f-6)) for k in 1:K]
    U::AbstractArray{T} = qr!(randn(eltype(XH), dh, dh)).Q[:, 1:dl]
    ω::T = convert(eltype(XH), 10f0) # penalty rate
    llh::AbstractArray{T} = convert(Array{eltype(XH)}, fill(-Inf32, 10))
end

"""
Interface - to be changed
"""
function MrfMixGauss(X::AbstractArray{T}, adj::AbstractArray, K::Int, ω::T=convert(T, 10f0); 
    tol::T=convert(T, 1f-6), maxiter::Int=1000
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
    tol::T=convert(T, 1f-6), maxiter::Int=1000
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

function MrfMixGauss!(
    model::Union{MRFBatchSeg{T}, MRFBatch{T}, PairedMRFBatchSeg{T}, PairedMRFBatch{T}};
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T <: Real
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Markov Random Field Gaussian mixture...", dt=0.1, spinner=true)
    iter = 1
    while iter < maxiter
        iter += 1
        L[iter] = batch(model)
        incr = (L[iter] - L[iter-1]) / L[iter-1]
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        if abs(incr) < tol
            ProgressMeter.finish!(prog)
            model.llh = copy(L[2:iter])
            return model
        end
    end
    ProgressMeter.finish!(prog)
    iter == maxiter || @warn "Not converged after $(maxiter) steps."
    model.llh = copy(L[2:iter])
    return model
end

function MrfMixGauss!(
    model::AbstractVector{Union{MRFBatchSeg{T}, MRFBatch{T}, PairedMRFBatchSeg{T}, PairedMRFBatch{T}}};
    tol::T=convert(T, 1f-6), maxiter::Int=10000
) where T <: Real
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Markov Random Field Gaussian mixture...", dt=0.1, spinner=true)
    iter = 1
    while iter < maxiter
        iter += 1
        L[iter] = model |> eachrow |> Map(x -> batch(x)) |> Broadcasting() |> Folds.sum
        incr = (L[iter] - L[iter-1]) / L[iter-1]
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        if abs(incr) < tol
            ProgressMeter.finish!(prog)
            model.llh = copy(L[2:iter])
            return model
        end
    end
    ProgressMeter.finish!(prog)
    iter == maxiter || @warn "Not converged after $(maxiter) steps."
    model.llh = copy(L[2:iter])
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
    expect!(model, XHo, XLo)
end

function batch(model::PairedMRFBatchSeg{T}) where T <: Real
    XHo = deepcopy(model.XH)
    XLo = deepcopy(model.XL)
    # M step
    maximise!(model, XHo, XLo)
    # E step
    segment!(model)
    expect!(model, XHo, XLo)
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
    # update μ
    updateμ!(model, XHo, XLo)
    #updateU!(model, XHo)
    #updateΣ!(model, XHo, XLo)
    #mul!(model.μ, model.XL', model.R)
    #model.μ ./= model.nk'
    # update Σ
    @inbounds for k ∈ 1:model.K
        copyto!(XLo, model.XL)
        XLo .-= transpose(view(model.μ, :, k))
        XLo .*= sqrt.(view(model.R, :, k))
        model.ΣL[k] = cholesky!(Hermitian(XLo' * XLo ./ model.nk[k]) + I * 1f-5)
    end

    @inbounds for k ∈ 1:model.K
        mul!(XHo, model.XH, model.U')
        XHo .-= transpose(view(model.μ, :, k))
        XHo .*= sqrt.(view(model.R, :, k))
        model.ΣH[k] = cholesky!(Hermitian(XHo' * XHo ./ model.nk[k]) + I * 1f-5)
    end
end

function updateμ!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    @inbounds for k ∈ 1:model.K
        μk = view(model.μ, :, k)
        Rk = view(model.R, :, k)
        mul!(XHo, model.XH, model.U')
        copyto!(XLo, model.XL)
        rdiv!(XHo, model.ΣH[k])
        rmul!(XHo, model.ΣL[k].U')
        rmul!(XHo, model.ΣL[k].U)
        mul!(μk, transpose(XHo .+ XLo), Rk)
        @debug "μk" k μk
        ldiv!(cholesky!(Hermitian((model.ΣL[k].U' * model.ΣL[k].U) .+ (model.ΣH[k].U' * model.ΣH[k].U))), μk)
        lmul!(model.ΣH[k].U, μk)
        lmul!(model.ΣH[k].U', μk)
        @debug "μk" k μk
    end
    #mul!(model.μ, model.XL', model.R)
    model.μ ./= model.nk'
    model.μ[findall(isnan, model.μ)] .= 0
end

function updateU!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, Xo::AbstractArray{T}
) where T <: Real
    fill!(model.U, 0)
    @inbounds for k ∈ model.K
        Rk = view(model.R, :, k)
        copyto!(Xo, model.XH)
        Xo .*= sqrt.(Rk)
        model.U .+= transpose(Xo) * Xo
    end
    u, _, v = svd!(model.μ * transpose(model.R) * model.XH * model.U)
    mul!(model.U, u, v')
end

function updateΣ!(
    model::Union{PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    @inbounds for k ∈ model.K
        μk = view(model.μ, :, k)
        Rk = view(model.R, :, k)
        mul!(XHo, model.XH, model.U')
        XHo .-= μk'
        copyto!(XLo, model.XL)
        XLo .-= μk'
        map([XHo, XLo]) do x
            x .*= sqrt.(Rk)
        end
        model.ΣH[k] = cholesky!(I * 1f-6 + Hermitian((XHo' * XHo) ./ model.nk[k]))
        model.ΣL[k] = cholesky!(I * 1f-6 + Hermitian((XLo' * XLo) ./ model.nk[k]))
    end

    @inbounds for k ∈ 1:model.K
        copyto!(XLo, model.XL)
        XLo .-= transpose(view(model.μ, :, k))
        XLo .*= sqrt.(view(model.R, :, k))
        model.ΣL[k] = cholesky!(Hermitian(XLo' * XLo ./ model.nk[k]) + I * 1f-5)
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
        Rk .+= logdet(model.Σ[k]) + model.d * log(2π)
        Rk .*= -0.5f0
    end
    @debug "R" model.R
    copyto!(model.E1, model.R)
    # log prior
    logPrior!(model)
    model.R .+= @avx log.(model.E2)
    copyto!(model.llhmap, model.R)
    l = sum(Flux.logsumexp(model.R, dims=2)) / model.n
    #@info "model.R" model.R maximum(model.R)
    Flux.softmax!(model.R, dims=2)
    @debug "R" model.R
    return l
end

function expect!(
    model::Union{PairedMRFBatch{T}, PairedMRFBatchSeg{T}}, XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T<:Real
    #@inbounds for k ∈ 1:model.K
    #    Rk = view(model.R, :, k)
    #    μk = view(model.μ, :, k)
    #    # Gauss llh
    #    mul!(XHo, model.XH, model.U')
    #    copyto!(XLo, model.XL)
    #    # demean
    #    map([XHo, XLo]) do x
    #        x .-= μk'
    #    end
    #    @debug "diag" findall(isnan, XHo) findall(isnan, XLo)
    #    copyto!(Rk, diag((XHo / model.ΣH[k]) * XHo') .+ diag((XLo / model.ΣL[k]) * XLo'))
    #    Rk .+= logdet(model.ΣH[k]) .+ logdet(model.ΣL[k]) .+ (model.dh + model.dl) * log(2π)
    #    Rk .*= -0.5f0
    #    logPrior!(Rk, model, k)
    #end

    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        μk = view(model.μ, :, k)
        # Gauss llh
        copyto!(XLo, model.XL)
        XLo .-= μk'
        copyto!(Rk, diag((XLo / model.ΣL[k]) * XLo'))
        Rk .+= logdet(model.ΣL[k]) + model.dl * log(2π)
        Rk .*= -0.5f0
    end
    # log prior
    logPrior!(model)
    @debug "R" model.R
    l = sum(Flux.logsumexp(model.R, dims=2)) / model.n
    Flux.softmax!(model.R, dims=2)
    @debug "R" model.R
    return l
end

function logPrior!(model::Union{MRFBatchSeg{T}, PairedMRFBatchSeg{T}}) where T <: Real
    for k ∈ 1:model.K
        Ek = view(model.E2, :, k)
        @inbounds for v ∈ 1:model.n
            #Rk[v] -= sum([(model.seg[idx] != k) * model.f[v][kkk] for (kkk, idx) ∈ enumerate(model.adj[v])])
            Ek[v] = -model.ω * sum( (model.seg[collect(model.adj[v])] .!= k) .* model.f[v] )
            #Ek[v] = -model.ω * sum((model.seg[collect(model.adj[v])] .!= k))
        end
    end
    Flux.softmax!(model.E2, dims=2)
    #Rk .+= log(model.nk[k]/model.n)
end

function logPrior!(model::Union{MRFBatch{T}, PairedMRFBatch{T}}) where T <: Real
    for k ∈ 1:model.K
        Ek = view(model.E2, :, k)
        @inbounds for v ∈ 1:model.n
            #Rk[v] += model.ω * sum([model.R[idx, k] for idx ∈ model.adj[v]])
            Ek[v] = model.ω * sum(model.R[collect(model.adj[v]), k] .* model.f[v])
        end
    end
    model.E2 ./= sum(model.E2, dims=2)
    #Rk .+= log(model.nk[k]/model.n)
end

function segment!(model::Union{MRFBatchSeg{T}, PairedMRFBatchSeg{T}}) where T<:Real
    copyto!(model.seg, Flux.onecold(model.R', 1:model.K))
    model
end
