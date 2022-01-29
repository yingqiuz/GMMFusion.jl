@with_kw mutable struct MrfBatch{T<:Real}
    K::Int
    adj::AbstractArray
    R::AbstractArray{T}
    n::Int = size(R, 1)
    seg::AbstractArray{Int} = zeros(Int, n)
    nk::AbstractArray{T} = vec(sum(R, dims=1))
    ω::T = convert(eltype(X), 1f0) # penalty rate
    llh::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, 10))
    E1::AbstractArray{T} = zeros(eltype(X), n, K)
    E2::AbstractArray{T} = ones(eltype(X), n) .* (nk' ./ n)
    llhmap::AbstractArray{T} = convert(Array{eltype(X)}, fill(-Inf32, n, K))
    # record history
    Rhistory::AbstractArray{T} = fill(-NaN32, n, 10)
    E1history::AbstractArray{T} = fill(-NaN32, n, 10)
    E2history::AbstractArray{T} = fill(-NaN32, n, 10)
    llhmaphistory::AbstractArray{T} = fill(-NaN32, n, 10)
end

function MrfBatch!(
    model::MrfBatch{T}; 
    tol::T=convert(T, 1f-6), maxiter::Int=1000
) where T<:Real
    # likelihood vector
    L = fill(convert(eltype(model.X), -Inf32), maxiter)
    # progress bar
    prog = ProgressUnknown("Running Gamma mixture model...", dt=0.1, spinner=true)
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

function batch(model::MrfBatch{T}) where T<: Real
    segment!(model)
    # add mrf
    logPrior!(model)
    model.R .+= @avx log.(model.E2)
    copyto!(model.llhmap, model.R)
    l = sum(Flux.logsumexp(model.R, dims=2)) / model.n
    #@info "model.R" model.R maximum(model.R)
    Flux.softmax!(model.R, dims=2)
    @debug "R" model.R
    return l
end