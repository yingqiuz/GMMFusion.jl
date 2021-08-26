@with_kw immutable struct Priors{T<:Real}
    K::Int
    α::AbstractArray{T} = ones(T, K)
    β::AbstractArray{T} = ones(T, K)
    ν::AbstractARray{T}
    m::AbstractArray{T}
    W::AbstractArray
    logW::AbstractArray{T}
end

@with_kw immutable struct Batch{T<:Real}
    K::Int
    X::AbstractArray{T}
    R::AbstractArray{T} = fill(1f0/K, size(X, 1), K)
    α::AbstractArray{T}
    β::AbstractArray{T}
    ν::AbstractArray{T}
    m::AbstractArray{T}
    W::AbstractArray
    logW::AbstractArray{T}
end

"""
Variational Bayesian Gaussian Mixture Model
"""
function MixGaussVb(
    X::AbstractArray{T}, prior::Priors{T};
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp, 
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    @debug "prior" prior
    K = prior.K
    # init 
    model = Batch(K=K, X=X, α=prior.α, β=prior.β, ν=prior.ν, m=prior.m, W=prior.W, logW=prior.logW)
    MixGaussVb!(model, prior; tol=tol, maxiter=maxiter)
end

function MixGaussVb!(
    model::Batch{T}, prior::Priors{T};
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, d = size(model.X)
    K = model.K
    model.K == prior.K || throw(DimensionMismatch("Prior K and model K mismatch."))
    # allocate memory for temporary matrices
    Xo = copy(X)
    nk = zeros(T, K)
    # allocate memory for lowerbound
    lb = fill(-Inf32, maxiter)
    prog = ProgressUnknown("Running Variational Bayesian Gaussian mixture...", dt=0.1, spinner=true)
    incr = NaN32
    for iter ∈ 2:maxiter
        # E-step
        ProgressMeter.next!(
            prog; showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        @debug "model" model
        # M-step
        M!(model, nk, prior)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        #@info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            ProgressMeter.finish!(prog)
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function M!(model::Batch{T}, nk::AbstractArray{T}, Xo::AbstractArray{T}, prior::Priors{T}) where T <: Real
    # posterior parameters
    sum!(nk, model.R')
    model.α .= prior.α .+ nk
    model.β .= prior.β .+ nk
    model.ν .= prior.ν .+ nk
    # update m
    mul!(model.m, model.X', model.R)
    model.m .+= prior.β' .* prior.m
    model.m ./= model.β
    for k ∈ 1:K
        mk = view(model.m, :, k)
        mk0 = view(prior.m, :, k)
        copyto!(Xo, model.X)
        Xo .-= mk'
        Xo .*= sqrt.(view(model.R, :, k))
        mul!(model.W[k], Xo', Xo)
        model.W[k] .+= prior.W[k] .+ prior.β[k] .* (mk .- mk0) .* (mk .- mk0)'
    end
end

function E!(
    model::Batch{T}, prior::Priors{T}, 
) where T <: Real
    n, d = size(model.X)
    Xo = copy(model.X)
    for k ∈ 1:K 
        ## 
        Rk = view(model.R, :, k)
        mk = view(model.m, :, k)
        copyto!(Xo, X)
        Xo .-= mk'
        copyto!(Rk, diag(Xo * (W \ Xo')))
        Rk .+= d / model.β[k] + d * log(2π)
    end
    model.R .-= sum(digamma.((model.v' .- collect(1:d) .+ 1)./2), dims=1) .- logdet(W) .+ d * log(2)
    model.R ./= -2
    model.R .+= digamma(model.α') .- digamma(sum(model.α))
    softmax!(model.R, dims=2)
end

function LowerBound(model::Batch{T}, prior::Priors{T}) where T <: Real
    n, d = size(model.X)
    K = size(model.R, 2)
    # Epz - Eqz
    L = 0 - sum(model.R .* log.(model.R))
    # Eppi - Eqpi
    L += loggamma(sum(prior.α)) - sum(loggamma.(prior.α)) 
    L -= loggamma(sum(model.α)) - sum(loggamma.(model.α))
    # Epmu - Eqmu
    L += d * sum(log.(prior.β)) / 2 - d*sum(log.(mode.β)) / 2
    # EpLambda - EqLambda
    L -= sum(0.5f0 .* prior.v .* (prior.logW .+ d*log(2))) + sum(loggamma.((prior.v' .- collect(1:d) .+ 1) ./ 2))
    L += sum(0.5f0 .* model.v .* (model.logW .+ d*log(2))) + sum(loggamma.((model.v' .- collect(1:d) .+ 1) ./ 2))
    # Epx
    L += -0.5f0 * d * n * log(2π)
    return L
end
