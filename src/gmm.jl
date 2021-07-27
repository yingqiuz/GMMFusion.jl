using Base: Real
struct GMM
    K::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    w::AbstractVector                      # weights: n
    μ::AbstractArray                       # means: n x d
    Σ::AbstractVector{AbstractArray} # diagonal covmatariances n x d, or Vector n of d x d full covmatariances
    #hist::Array{History}           # history of this GMM
end

"""
interface normal version
"""
function EM(
    X::AbstractArray{T}, K::Int;
    init::Union{AbstractArray{T}, Nothing}=nothing,
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    # init 
    R = kmeans(X', K; init=init, tol=tol, max_iters=maxiter)
    w = convert(Array{T}, reshape(counts(R.assignments) ./ n, 1, K))  # cluster size
    μ = copy(R.centers)
    Σ = [cholesky!(cov(X)) for k ∈ 1:K]
    R = [x == k ? 1 : 0 for x ∈ R.assignments, k ∈ 1:K]
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), copy(X), w, μ, Σ; 
        tol=tol, maxiter=maxiter)
end

function EM!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractArray{T}, 
    μ::AbstractMatrix{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}};
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, d = size(X)
    n2, K = size(R)
    n == n2 || throw(DimensionMismatch("Dimension of X and R mismatch."))
    # allocate memory for temporary matrices
    Xo = copy(X)
    #covmat = zeros(T, n, n)

    # allocate memory for llh
    llh = Vector{T}(undef, maxiter)
    fill!(llh, -Inf32)
    prog = ProgressMeter.ProgressUnknown()
    @showprogress 0.1 "EM for gmm..." for iter ∈ 2:maxiter
        # E-step
        @debug "R" R
        @debug "w" w
        @debug "μ" μ
        @debug "Σ" Σ
        llh[iter] = E!(R, X, w, μ, Σ, Xo)
        # M-step
        M!(w, μ, Σ, R, X, Xo)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        @info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function M!(
    w::AbstractArray{T}, μ::AbstractMatrix{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractMatrix{T}, X::AbstractMatrix{T}, Xo::AbstractMatrix{T}
) where T <: Real
    n, K = size(R)
    # udpate parameters
    w .= sum(R, dims=1) # remember to div by n
    @debug "w" w
    mul!(μ, transpose(X), R)
    μ ./= w
    # update Σ
    Threads.@threads for k ∈ 1:K
        #copy!(Xo, X)
        Xtmp = deepcopy(X)
        Xtmp .-= transpose(view(μ, :, k))
        Xtmp .*= sqrt.(view(R, :, k))
        Σ[k] = cholesky!(Xtmp' * Xtmp ./ w[1, k] + I * 1f-8)
    end
    w ./= n
end

"""
interface batched version
"""
function EM(
    X::Vector{A} where A <: AbstractArray{T}, K::Int;
    init::Union{AbstractArray{T, 2}, Nothing}=nothing,
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    nlist = [size(x, 1) for x ∈ X]
    n = sum(nlist)
    d = size(X[1], 2)
    R = kmeans(Elkan(), vcat(X...)', K; k_init=init, tol=tol, max_iters=1000)
    @debug "R" R
    μ = deepcopy(R.centers)
    @debug "μ" μ
    # recalculate assignment
    #w = mapreduce(r -> counts(r.assignments), +, R)
    w = convert(Array{T}, reshape(counts(R.assignments) ./ n, 1, K))
    @debug "w" w
    R = [convert(
            Array{T}, 
            [
                x == k ? 1 : 0 
                for x ∈ R.assignments[sum(nlist[1:i]) - ntmp + 1:sum(nlist[1:i])], k ∈ 1:K
            ]
        ) for (i, ntmp) ∈ enumerate(nlist)]
    @debug "R" R
    C = [cholesky!(I + zeros(T, d, d)) for k ∈ 1:K]
    Σ = [I + zeros(T, d, d) for k ∈ 1:K]
    updateΣ!(C, Σ, μ, R, X, deepcopy(X), w)
    @debug "Σ" Σ
    @debug "C" C
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(R, X, w, μ, C, Σ; tol=tol, maxiter=maxiter)
end

"""
batched version
"""
function EM!(
    R::AbstractVector{A} where A <: AbstractArray{T}, 
    X::AbstractVector{A} where A <: AbstractArray{T}, 
    w::AbstractMatrix{T}, μ::AbstractMatrix{T}, 
    C::AbstractVector{B} where B <: Cholesky{T, Matrix{T}},
    Σ::AbstractVector{A} where A <: AbstractArray{T};
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n_bs = size(X, 1)
    n = sum([size(x, 1) for x ∈ X])
    #n2, K = size(R)
    #n == n2 || throw(DimensionMismatch("Dimension of X and R mismatch."))
    # allocate memory for temporary matrices
    Xo = deepcopy(X)
    #covmat = zeros(T, n, n)
    # allocate memory for llh
    llh = zeros(T, maxiter)
    llh[1] = -Inf32
    prog = ProgressUnknown("EM for gmm...", spinner=true)
    incr = NaN
    for iter ∈ 2:maxiter
        ProgressMeter.next!(prog, spinner="🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"; showvalues = [(:iter, iter-1), (:incr, incr)])
        # E-step
        @debug "R" R
        #llh[iter] = Folds.sum(
        #    E!(r, x, w, μ, C, xo) for (r, x, xo) ∈ zip(R, X, Xo)
        #)
        @showprogress 0.1 "$(iter-1) E step" for (r, x, xo) ∈ zip(R, X, Xo)
        #end
            llh[iter] += E!(r, x, w, μ, C, xo)
            #@info "X" X
        end
        llh[iter] /= n 
        @info "llh $(iter) " llh[iter]
        @debug "R" R
        # update the coefficients and store in wp
        fill!(w, 0)
        @inbounds for bt ∈ 1:n_bs
            w .+= sum(R[bt], dims=1)
            #@info "X" X
        end
        # now update other params M step
        M!(μ, C, Σ, w, X, Xo, R)
        w ./= n
        @debug "R" R
        @debug "w" w
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        
        if abs(incr) < tol || iter == maxiter
            ProgressMeter.finish!(prog)
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function E!(
    R::AbstractArray{T},  X::AbstractArray{T}, w::AbstractArray{T},
    μ::AbstractArray{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Xo::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    @debug "μ" μ
    Threads.@threads for k ∈ 1:K
        expectation!(
            view(R, :, k), X, Xo, view(μ, :, k), Σ[k]
        )
    end
    R .+= log.(w)
    llh = logsumexp(R, dims=2)
    R .-= llh
    R .= exp.(R)
    @debug "R" R
    return sum(llh)
end

function expectation!(
    Rk::AbstractVector{T},
    X::AbstractArray{T},
    Xo::AbstractArray{T},
    μ::AbstractVector{T},
    C::Cholesky{T, Matrix{T}}
) where T <: Real
    n, d = size(X)
    #copyto!(Xo, X)
    Xo = deepcopy(X)
    Xo .-= μ'
    #@debug "X Xo" X sum(Xo, dims=1)
    #C = cholesky!(Hermitian(Σ))
    fill!(Rk, -logdet(C) / 2 - log(2π) * d / 2)
    #rdiv!(Xo, C.U)
    #@turbo for nn ∈ 1:n, dd ∈ 1:d
    #    Rk[nn] -= Xo[nn, dd] ^ 2 / 2 
    #end
    #mul!(covmat, Xo, C \ transpose(Xo))
    #@debug "covmat" diag(covmat)
    Rk .-= diag(Xo * (C \ transpose(Xo))) ./ 2
end

function M!(
    μ::AbstractMatrix{T}, C::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Σ::AbstractVector{B} where B <: AbstractArray{T}, w::AbstractMatrix{T},
    X::AbstractVector{B} where B <: AbstractArray{T}, 
    Xo::AbstractVector{B} where B <: AbstractArray{T}, 
    R::AbstractVector{B} where B <: AbstractArray{T},
) where T <: Real
    updateμ!(μ, R, X, w)
    @debug "μ" μ
    updateΣ!(C, Σ, μ, R, X, Xo, w)
end

function updateμ!(
    μ::AbstractArray{T}, R::AbstractVector{A} where A <: AbstractArray{T}, 
    X::AbstractVector{A} where A <: AbstractArray{T}, 
    w::AbstractMatrix{T},
) where T <: Real
    fill!(μ, 0)
    for (x, r) ∈ zip(X, R)
        μ .+= transpose(x) * r
    end
    μ ./= w
end

function updateΣ!(
    C::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Σ::AbstractVector{B} where B <: AbstractArray{T}, 
    μ::AbstractArray{T}, R::AbstractVector{B} where B <: AbstractArray{T}, 
    X::AbstractVector{B} where B <: AbstractArray{T}, 
    Xo::AbstractVector{B} where B <: AbstractArray{T}, 
    w::AbstractMatrix{T},
) where T <: Real
    K = size(μ, 2)
    n_bs = size(X, 1)
    # update Σ
    @showprogress 0.5 "update Σ..." for k ∈ 1:K
        fill!(Σ[k], 0)
        μk = @view μ[:, k]
        @inbounds for bt ∈ 1:n_bs
            copyto!(Xo[bt], X[bt])
            Xo[bt] .-= μk'
            Σ[k] .+= transpose(Xo[bt]) * Diagonal(view(R[bt], :, k)) * Xo[bt]
        end
        Σ[k] ./= w[1, k]
        C[k] = cholesky!(Hermitian(Σ[k] + I * 1f-6))
        @debug "Σ[$(k)]" Σ[k]
    end
end