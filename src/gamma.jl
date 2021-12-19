@with_kw mutable struct GammaBatch{T<:Real}
    K::Int
    X::AbstractVector{T}
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = convert(Array{eltype(X)}, fill(1f0 / K, n, K))
    nk::AbstractVector{T} = vec(sum(R, dims=1))
    w::AbstractVector{T} = nk ./ n
    m::AbstractVector{T} = R' * X ./ nk
    θ::AbstractVector{T} = vec(var(sqrt.(R) .* (X .- m'), dims=1)) .* n ./ (R' * X)
    α::AbstractVector{T} = (R' * X ./ nk) ./ θ
    #Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    llhmap::AbstractArray{T} = zeros(eltype(X), n, K)
end

@with_kw mutable struct MrfGammaBatch{T<:Real}
    K::Int
    X::AbstractVector{T}
    adj::AbstractArray
    index::AbstractArray{Int} = findall(x -> x>1f-3, std(X, dims=1)[:])
    n::Int = size(X, 1)
    d::Int = size(X, 2)
    R::AbstractArray{T} = convert(Array{eltype(X)}, fill(1f0 / K, n, K))
    E2::AbstractArray{T} = copy(R)
    nk::AbstractVector{T} = vec(sum(R, dims=1))
    w::AbstractVector{T} = nk ./ n
    seg::AbstractArray{Int} = Flux.onecold(R', 1:K)
    m::AbstractVector{T} = R' * X ./ nk
    θ::AbstractVector{T} = vec(var(sqrt.(R) .* (X .- m'), dims=1)) .* n ./ (R' * X)
    α::AbstractVector{T} = (R' * X ./ nk) ./ θ
    ω::T = convert(eltype(X), 10f0) # penalty rate
    σ::T = convert(eltype(X), 1f0) # length scale
    #Σ::AbstractArray = [cholesky!(Hermitian(cov(X) + I * 1f-6)) for k in 1:K]
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    llhmap::AbstractArray{T} = zeros(eltype(X), n, K)
end

function mixGamma!(
    model::Union{GammaBatch{T}, MrfGammaBatch{T}}; 
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T<:Real
    # likelihood vector
    L = fill(-Inf32, maxiter)
    # progress bar
    prog = ProgressUnknown("Running Gamma mixture model...", dt=0.1, spinner=true)
    iter = 1
    bar = deepcopy(model.nk)
    α₀ = rand(eltype(model.X), model.K)
    while iter < maxiter
        iter += 1
        L[iter] = batch(model, bar, α₀)
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

function batch(model::Union{GammaBatch{T}, MrfGammaBatch{T}}, bar::AbstractArray{T}, α₀::AbstractArray{T}) where T <: Real
    # M step
    maximise!(model, bar, α₀)
    # E step
    expect!(model)
end

function maximise!(model::Union{GammaBatch{T}, MrfGammaBatch{T}}, bar::AbstractArray{T}, α₀::AbstractArray{T}) where T <: Real
    # posterior parameters
    # update mixing weights
    sum!(model.nk, model.R')
    copyto!(model.w, model.nk ./ model.n)
    @debug "model.R" model.R
    # update α
    mul!(model.θ, model.R' ./ model.nk, model.X)
    @info "model.θ" model.θ
    mul!(bar, model.R' ./ model.nk, @avx log.(model.X .+ 1f-6))
    @info "bar" bar
    bar .-= @avx log.(model.θ .+ 1f-6)
    @info "bar" bar
    updateα!(model, bar, α₀, 1f-4)
    # update β
    model.θ ./= model.α
    @info "model.θ" model.θ
end

function updateα!(model::Union{GammaBatch{T}, MrfGammaBatch{T}}, bar::AbstractArray{T}, α₀::AbstractArray{T}, tol::T=convert(T, 1f-5)) where T<:Real
    copyto!(α₀, rand(eltype(α₀), model.K))
    #@info "model.α" α₀ model.α bar
    @inbounds for k ∈ 1:model.K
        while ((abs(model.α[k] - α₀[k]) / α₀[k]) > tol)
            α₀[k] = copy(model.α[k])
            model.α[k] = 1 / (1 / α₀[k] + (bar[k] + log(α₀[k]) - digamma(α₀[k])) / (α₀[k] ^ 2 * (1 / α₀[k] - polygamma(1, α₀[k]))))
            model.α[k] += 1f-6
            #@info "model.α[k]" model.α[k]
            # α[k] = invdigamma(bar[k] + log(α[k]))
        end
    end
    @info "model.α" α₀ model.α
end

function expect!(model::GammaBatch{T}) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        # Gamma pdf
        copyto!(Rk, pdf.(Gamma(model.α[k], model.θ[k]), model.X))
    end
    model.R .*= model.w'
    @debug "R" model.R
    l = sum(@avx log.(sum(model.R, dims=2))) / model.n
    #@info "model.R" model.R maximum(model.R)
    model.R ./= sum(model.R, dims=2)
    # deal with inf
    @debug "R" model.R
    return l
end

function expect!(model::MrfGammaBatch{T}) where T<:Real
    @inbounds for k ∈ 1:model.K
        Rk = view(model.R, :, k)
        # Gamma pdf
        copyto!(Rk, pdf.(Gamma(model.α[k], model.θ[k]), model.X))
    end
    logPrior!(model)
    model.R .*= model.E2
    @debug "R" model.R
    l = sum(@avx log.(sum(model.R, dims=2))) / model.n
    #@info "model.R" model.R maximum(model.R)
    model.R ./= sum(model.R, dims=2)
    # deal with inf
    @debug "R" model.R
    return l
end

function logPrior!(model::MrfGammaBatch{T}) where T <: Real
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