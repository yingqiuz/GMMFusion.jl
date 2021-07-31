function predict(
    model::GMM{T}, 
    Xtest::AbstractArray{T}
) where T <: Real
    K = model.K
    ntest = size(Xtest, 1)
    Rtest = zeros(T, ntest, K)
    covmat = zeros(T, ntest, ntest)
    Xo = copy(Xtest)
    E!(Rtest, Xtest, model.w, model.μ, model.Σ, Xo, covmat)
    return Rtest
end

function EM(
    X::AbstractArray{T}, K::Int;
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp,
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, d = size(X)
    # init 
    R = kmeans(X', K; init=init, tol=tol, maxiter=maxiter)
    w = convert(Array{T}, counts(R) ./ n)  # cluster size
    μ = copy(R.centers)
    Σ = [cholesky!(cov(X)) for k ∈ 1:K]
    R = [x == k ? 1 : 0 for x ∈ assignments(R), k ∈ 1:K]
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), copy(X), w, μ, Σ; 
        tol=tol, maxiter=maxiter)
    return GMM(K, d, w, μ, Σ)
end

function EM!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractVector{T}, 
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
    prog = ProgressUnknown("Running EM...", spinner=true)
    incr = NaN32
    for iter ∈ 2:maxiter
        # E-step
        ProgressMeter.next!(
            prog; spinner="🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛",
            showvalues = [(:iter, iter-1), (:incr, incr)]
        )
        @debug "R" R
        @debug "w" w
        @debug "μ" μ
        @debug "Σ" Σ
        llh[iter] = E!(R, X, w, μ, Σ, Xo, covmat)
        # M-step
        M!(w, μ, Σ, R, X, Xo)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        #@info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            ProgressMeter.finish!(prog)
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function E!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractVector{T},
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
    @avx R .+= log.(w')
    @debug "R" R
    llh = logsumexp(R, dims=2)
    R .-= llh
    @avx R .= exp.(R)
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
    w::AbstractVector{T}, μ::AbstractMatrix{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractMatrix{T}, X::AbstractMatrix{T}, Xo::AbstractMatrix{T}
) where T <: Real
    n, K = size(R)
    # udpate parameters
    sum!(w, R')
    #w .= vec(sum(R, dims=1)) # remember to div by n
    @debug "w" w
    mul!(μ, transpose(X), R)
    μ ./= w'
    # update Σ
    @inbounds for k ∈ 1:K
        copy!(Xo, X)
        Xo .-= transpose(view(μ, :, k))
        Xo .*= sqrt.(view(R, :, k))
        Σ[k] = cholesky!(Xo' * Xo ./ w[k] + I * 1f-8)
    end
    w ./= n
end

function EM(
    R::KmeansResult{Matrix{T}, T, Int}, X::AbstractArray{T}, K::Int;
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
    w = convert(Array{T}, counts(R) ./ n)  # cluster size
    R = [x == k ? 1 : 0 for x ∈ a, k ∈ 1:K]
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), copy(X), w, μ, Σ; 
        tol=tol, maxiter=maxiter)
    return GMM(K, d, w, μ, Σ)
end