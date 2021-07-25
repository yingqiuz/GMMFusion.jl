using Base: Real
struct GMM
    n::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    w::AbstractVector                      # weights: n
    μ::AbstractArray                       # means: n x d
    Σ::AbstractVector{AbstractArray} # diagonal covmatariances n x d, or Vector n of d x d full covmatariances
    #hist::Array{History}           # history of this GMM
end

function EM(
    X::AbstractArray{T}, Xtest::AbstractArray{T}, K::Int;
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp,
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, d = size(X)
    # init 
    R = kmeans(X', K; init=init, tol=tol, maxiter=maxiter)
    w = convert(Array{T}, reshape(counts(R) ./ n, 1, K))  # cluster size
    μ = copy(R.centers)
    Σ = [cholesky!(cov(X)) for k ∈ 1:K]
    R = [x == k ? 1 : 0 for x ∈ assignments(R), k ∈ 1:K]
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), copy(X), w, μ, Σ; 
        tol=tol, maxiter=maxiter)
    ntest = size(Xtest, 1)
    Rtest = zeros(T, ntest, K)
    covmat = zeros(T, ntest, ntest)
    Xo = copy(Xtest)
    E!(Rtest, Xtest, w, μ, Σ, Xo, covmat)
    return Rtest
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
    covmat = zeros(T, n, n)

    # allocate memory for llh
    llh = Vector{T}(undef, maxiter)
    fill!(llh, -Inf32)
    @showprogress 0.1 "EM for gmm..." for iter ∈ 2:maxiter
        # E-step
        @debug "R" R
        @debug "w" w
        @debug "μ" μ
        @debug "Σ" Σ
        llh[iter] = E!(R, X, w, μ, Σ, Xo, covmat)
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

function E!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractArray{T},
    μ::AbstractArray{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Xo::AbstractArray{T}, covmat::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    @inbounds for k ∈ 1:K
        expectation!(
            view(R, :, k), X, Xo, covmat,
            view(μ, :, k), Σ[k]
        )
    end
    @debug "R" R
    R .+= log.(w)
    @debug "R" R
    llh = logsumexp(R, dims=2)
    R .-= llh
    R .= exp.(R)
    @debug "R" R
    return sum(llh) / n
end

function expectation!(
    Rk::AbstractVector{T},
    X::AbstractArray{T},
    Xo::AbstractArray{T},
    covmat::AbstractMatrix{T},
    μ::AbstractVector{T},
    C::Cholesky{T, Matrix{T}}
) where T <: Real
    n, d = size(X)
    copyto!(Xo, X)
    Xo .-= μ'
    #@debug "X Xo" X sum(Xo, dims=1)
    #C = cholesky!(Hermitian(Σ))
    fill!(Rk, -logdet(C) / 2 - log(2π) * d / 2)
    mul!(covmat, Xo, C \ transpose(Xo))
    @debug "covmat" diag(covmat)
    Rk .-= diag(covmat) ./ 2
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
    @inbounds for k ∈ 1:K
        copy!(Xo, X)
        Xo .-= transpose(view(μ, :, k))
        Xo .*= sqrt.(view(R, :, k))
        Σ[k] = cholesky!(Xo' * Xo ./ w[1, k] + I * 1f-8)
    end
    w ./= n
end

function EM(
    R::KmeansResult{Matrix{T}, T, Int}, X::AbstractArray{T}, Xtest::AbstractArray{T}, K::Int;
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, d = size(X)
    # init
    a = assignments(R)
    μ = Matrix{T}(undef, d, K)
    for k ∈ 1:K
        μ[:, k] .= mean(X[findall(x -> x == k, a), :], dims=1)[:]
    end
    Σ = [cholesky!(cov(X)) for k ∈ 1:K]
    w = convert(Array{T}, reshape(counts(R) ./ n, 1, K))  # cluster size
    R = [x == k ? 1 : 0 for x ∈ a, k ∈ 1:K]
    

    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), copy(X), w, μ, Σ; 
        tol=tol, maxiter=maxiter)
    ntest = size(Xtest, 1)
    Rtest = zeros(T, ntest, K)
    covmat = zeros(T, ntest, ntest)
    Xo = copy(Xtest)
    E!(Rtest, Xtest, w, μ, Σ, Xo, covmat)
    return Rtest
end