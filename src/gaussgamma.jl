@with_kw mutable struct GaussGammaBatch{T<:Real}
    K::Int
    X::AbstractVector{T}
    #index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = convert(Array{eltype(X)}, fill(1f0 / K, n, K))
    nk::AbstractVector{T} = vec(sum(R, dims=1))
    w::AbstractVector{T} = nk ./ n
    m::AbstractVector{T} = R' * X ./ nk
    μ::T = m[1]  # Gaussian
    σ²::T = var(sqrt.(view(R, :, 1)) .* (X .- μ)) * n / nk[1]  # Gaussian
    θ::T = var(sqrt.(view(R, :, 2)) .* (X .- m[2])) * n / nk[2] / m[2]  # vec(var(sqrt.(R) .* (X .- m'), dims=1)) .* n ./ (R' * X)
    α::T = m[2] / θ # (R' * X ./ nk) ./ θ
    #Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    llhmap::AbstractArray{T} = zeros(eltype(X), n, K)
end

@with_kw mutable struct MrfGaussGammaBatch{T<:Real}
    K::Int
    X::AbstractVector{T}
    adj::AbstractArray
    #index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = convert(Array{eltype(X)}, fill(1f0 / K, n, K))
    E2::AbstractArray{T} = copy(R)
    nk::AbstractVector{T} = vec(sum(R, dims=1))
    w::AbstractVector{T} = nk ./ n
    seg::AbstractArray{Int} = Flux.onecold(R', 1:K)
    m::AbstractVector{T} = R' * X ./ nk
    μ::T = m[1]  # Gaussian
    σ²::T = var(sqrt.(view(R, :, 1)) .* (X .- μ)) * n / nk[1]  # Gaussian
    θ::T = var(sqrt.(view(R, :, 2)) .* (X .- m[2])) * n / nk[2] / m[2]  # vec(var(sqrt.(R) .* (X .- m'), dims=1)) .* n ./ (R' * X)
    α::T = m[2] / θ # (R' * X ./ nk) ./ θ
    ω::T = convert(eltype(X), 10f0) # penalty rate
    #ρ::T = convert(eltype(X), 1f0) # length scale
    #Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    llhmap::AbstractArray{T} = zeros(eltype(X), n, K)
end

function MixGaussGamma!(
    model::Union{GaussGammaBatch{T}, MrfGaussGammaBatch{T}}; 
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T<:Real
    # likelihood vector
    L = fill(convert(eltype(model.X), -Inf32), maxiter)
    # progress bar
    prog = ProgressUnknown("Running Gamma mixture model...", dt=0.1, spinner=true)
    iter = 1
    #bar = deepcopy(model.nk)
    #α₀ = rand(eltype(model.X), model.K)
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

function batch(model::Union{GaussGammaBatch{T}, MrfGaussGammaBatch{T}}) where T <: Real
    # M step
    maximise!(model)
    # E step
    expect!(model)
end

function maximise!(model::Union{GaussGammaBatch{T}, MrfGaussGammaBatch{T}}) where T <: Real
    # posterior parameters
    # update mixing weights
    sum!(model.nk, model.R')
    #model.nk .+= 1f-8
    copyto!(model.w, model.nk ./ model.n)
    @debug "model.R" model.R
    # update μ, σ²
    mul!(model.m, model.R' ./ model.nk, model.X)
    model.μ = model.m[1]
    model.σ² = var(sqrt.(view(model.R, :, 1)) .* (model.X .- model.μ)) * model.n / model.nk[1]
    # update θ, α
    bar = sum(view(model.R, :, 2) .* @avx log.(model.X .+ 1f-8)) ./ model.nk[2] - log(model.m[2] + 1f-8)
    α₀ = model.α + rand(eltype(model.α))
    while ((abs(model.α - α₀) / α₀) > 1f-5)
        α₀ = copy(model.α)
        model.α = 1 / (1f-8 + 1 / α₀ + (bar + log(α₀) - digamma(α₀)) / (α₀ ^ 2 * (1 / α₀ - polygamma(1, α₀))))
        #@info "model.α[k]" k model.α[k]
        model.α += 1f-8
    end
    #mul!(model.θ, model.R' ./ model.nk, model.X)
    #mul!(bar, model.R' ./ model.nk, @avx log.(model.X .+ 1f-6))
    #bar .-= @avx log.(model.θ .+ 1f-6)
    #updateα!(model, bar, α₀, 1f-4)
    # update β
    model.θ = model.m[2] / model.α
    #@info "model.θ" model.θ
end

function updateα!(
    model::Union{GaussGammaBatch{T}, MrfGaussGammaBatch{T}}, 
    bar::AbstractArray{T}, 
    α₀::AbstractArray{T}, 
    tol::T=convert(T, 1f-5)
) where T<:Real
    copyto!(α₀, model.α .+ rand(eltype(α₀), model.K))
    #@info "model.α" α₀ model.α bar
    @inbounds for k ∈ 1:model.K
        while ((abs(model.α[k] - α₀[k]) / α₀[k]) > tol)
            #@info "model.α[k]" k model.α[k]
            α₀[k] = copy(model.α[k])
            model.α[k] = 1 / (1f-6 + 1 / α₀[k] + (bar[k] + log(α₀[k]) - digamma(α₀[k])) / (α₀[k] ^ 2 * (1 / α₀[k] - polygamma(1, α₀[k]))))
            #@info "model.α[k]" k model.α[k]
            model.α[k] += 1f-6
            # α[k] = invdigamma(bar[k] + log(α[k]))
        end
    end
    @debug "model.α" α₀ model.α
end

function expect!(model::GaussGammaBatch{T}) where T <: Real
    # llh map of each component
    copyto!(view(model.R, :, 1), logpdf.(Normal(model.μ, sqrt(model.σ²)), model.X))
    copyto!(view(model.R, :, 2), logpdf.(Gamma(model.α, model.θ), model.X))
    copyto!(model.llhmap, model.R)
    #@info "R, w" model.R model.w
    model.R .+= @avx log.(model.w')
    #@debug "R" model.R
    #l = sum(@avx log.(sum(model.R, dims=2))) / model.n
    l = sum(Flux.logsumexp(model.R, dims=2)) / model.n
    #@debug "model.R" model.R maximum(model.R)
    Flux.softmax!(model.R, dims=2)
    return l
end

function expect!(model::MrfGaussGammaBatch{T}) where T<:Real
    # llh map of each component
    copyto!(view(model.R, :, 1), logpdf.(Normal(model.μ, sqrt(model.σ²)), model.X))
    copyto!(view(model.R, :, 2), logpdf.(Gamma(model.α, model.θ), model.X))
    copyto!(model.llhmap, model.R)
    logPrior!(model)
    model.R .+= @avx log.(model.E2)
    @debug "R" model.R
    l = sum(Flux.logsumexp(model.R, dims=2)) / model.n
    #@debug "model.R" model.R maximum(model.R)
    Flux.softmax!(model.R, dims=2)
    #l = sum(@avx log.(sum(model.R, dims=2))) / model.n
    #@debug "model.R" model.R maximum(model.R)
    #model.R ./= sum(model.R, dims=2)
    return l
end

function logPrior!(model::MrfGaussGammaBatch{T}) where T <: Real
    segment!(model)
    for k ∈ 1:model.K
        Ek = view(model.E2, :, k)
        @inbounds for v ∈ 1:model.n
            #Rk[v] -= sum([(model.seg[idx] != k) * model.f[v][kkk] for (kkk, idx) ∈ enumerate(model.adj[v])])
            #Ek[v] = -model.ω * sum( (model.seg[collect(model.adj[v])] .!= k) .* model.f[v] )
            Ek[v] = -model.ω * sum((model.seg[collect(model.adj[v])] .!= k))
        end
    end
    Flux.softmax!(model.E2, dims=2)
    #Rk .+= log(model.nk[k]/model.n)
end