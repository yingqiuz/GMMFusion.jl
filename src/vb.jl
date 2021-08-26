@with_kw mutable struct Prior{T<:Real}
    K::Int
    d::Int = size(m, 1)
    α::AbstractArray{T} = ones(T, K)
    β::AbstractArray{T} = ones(T, K)
    ν::AbstractArray{T} = ones(T, K) .* (d+1)
    m::AbstractArray{T}
    W::AbstractVector{AbstractArray{T}} = [zeros(T, d, d) + I for k in 1:K]
    chol::AbstractArray=[cholesky!(Hermitian(w + I * 1f-6)) for w in W]
    logW::AbstractArray{T}=[logdet(x) for x in chol]
end

@with_kw mutable struct Batch{T<:Real}
    K::Int
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    X::AbstractArray{T}
    R::AbstractArray{T}
    α::AbstractArray{T}
    β::AbstractArray{T}
    ν::AbstractArray{T}
    m::AbstractArray{T}
    W::AbstractArray{AbstractArray{T}}
    chol::AbstractArray=[cholesky!(Hermitian(w + I * 1f-6)) for w in W]
    logW::AbstractArray{T}
    lb::T=convert(T, -Inf)
end

"""
Variational Bayesian Gaussian Mixture Model
"""
function MixGaussVb(
    X::AbstractArray{T}, prior::Prior{T}, Rinit::AbstractArray{T}=fill(1f0/prior.K, size(X, 1), prior.K);
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T <: Real
    @debug "prior" prior
    K = prior.K
    n, d = size(X)
    prior.d == d || throw(DimensionMismatch("Dimension of priors and data mismatch."))
    # init 
    model = Batch(K=K, n=n, d=d, X=X, R=Rinit, α=prior.α, β=prior.β, ν=prior.ν, m=prior.m, W=prior.W, chol=prior.chol, logW=prior.logW)
    MixGaussVb!(model, prior; tol=tol, maxiter=maxiter)
end

function MixGaussVb!(
    model::Batch{T}, prior::Prior{T};
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T <: Real
    model.K == prior.K || throw(DimensionMismatch("Prior K and model K mismatch."))
    # allocate memory for temporary matrices
    Xo = copy(model.X)
    nk = zeros(T, model.K)
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
        maximise!(model, nk, Xo, prior)
        # E-step
        expect!(model, Xo)
        lb[iter] = LowerBound(model, prior)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        #@info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            ProgressMeter.finish!(prog)
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return model
        end
    end
end

function maximise!(model::Batch{T}, nk::AbstractArray{T}, Xo::AbstractArray{T}, prior::Prior{T}) where T <: Real
    # posterior parameters
    sum!(nk, model.R')
    model.α .= prior.α .+ nk
    model.β .= prior.β .+ nk
    model.ν .= prior.ν .+ nk
    # update posterior mean
    mul!(model.m, model.X', model.R)
    model.m .+= prior.β' .* prior.m
    model.m ./= model.β'
    for k ∈ 1:model.K
        mk = view(model.m, :, k)
        mk0 = view(prior.m, :, k)
        copyto!(Xo, model.X)
        Xo .-= mk'
        Xo .*= sqrt.(view(model.R, :, k))
        mul!(model.W[k], Xo', Xo)
        model.W[k] .+= prior.W[k] .+ prior.β[k] .* (mk .- mk0) .* (mk .- mk0)'
        model.chol[k] = cholesky!(Hermitian(model.W[k] + I * 1f-6))
        model.logW[k] = logdet(model.chol[k])
    end
end

function expect!(
    model::Batch{T}, Xo::AbstractArray{T},
) where T <: Real
    d = model.d
    for k ∈ 1:model.K 
        ## EQ
        Rk = view(model.R, :, k)
        mk = view(model.m, :, k)
        copyto!(Xo, model.X)
        Xo .-= mk'
        copyto!(Rk, diag((Xo / model.chol[k]) * Xo'))
    end
    model.R .*= model.ν'
    model.R .+= d ./ model.β' .+ d*log(2f0π)
    # minus ElogLambda
    model.R .-= sum(digamma.((model.ν' .- collect(1:d) .+ 1)./2), dims=1) .- model.logW' .+ d * log(2f0)
    model.R .*= -0.5f0
    model.R .+= digamma.(model.α') .- digamma(sum(model.α))
    Flux.softmax!(model.R, dims=2)
end

function LowerBound(model::Batch{T}, prior::Prior{T}) where T <: Real
    # Epz - Eqz
    lb = @avx 0 - sum(model.R .* log.(model.R))
    # Eppi - Eqpi
    lb += loggamma(sum(prior.α)) - sum(loggamma.(prior.α)) 
    lb -= loggamma(sum(model.α)) - sum(loggamma.(model.α))
    # Epmu - Eqmu
    lb += 0.5f0d * sum(log.(prior.β)) - 0.5f0d * sum(log.(model.β))
    # EpLambda - EqLambda
    lb -= EpLambda(prior) - EpLambda(model)
    #lb += sum(0.5f0 .* model.ν .* (-1f0 .* model.logW .+ d*log(2f0))) + sum(logmvgamma.(model.d, 0.5f0 .* model.ν))
    # Epx
    lb += -0.5f0 * d * n * log(2f0π)
    model.lb = lb
    return lb
end

EpLambda(x) = sum(0.5f0 .* x.ν .* (-1f0 .* x.logW .+ d*log(2f0))) + sum(logmvgamma.(x.d, 0.5f0 .* x.ν))

function evidence(model::Batch{T}, prior::Prior{T}) where T <: Real
    nk = sum(model.R, 1)[:]
    # Epz - Eqz
end