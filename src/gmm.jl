using Base: Real
struct GMM
    n::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    w::AbstractVector                      # weights: n
    μ::AbstractArray                       # means: n x d
    Σ::Union{AbstractArray, AbstractVector{AbstractArray}} # diagonal covariances n x d, or Vector n of d x d full covariances
    #hist::Array{History}           # history of this GMM
end

function EM(
    X::AbstractArray{T}, K::Int;
    kw...
) where T <: Real
    n, d = size(X, 2)
    # initialisation
    μ = zeros(T, K, d)
    Σ = [zeros(T, d, d) + I for k ∈ 1:K]
    w = ones(T, K) ./ K
    R = Matrix{T}(undef, n, K)
    fill!(R, 1 / K)

    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(R, copy(X), w, μ, Σ; kw...)
end

function EM!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractVector{T}, 
    μ::AbstractArray{T}, Σ::AbstractVector{AbstractArray{T}};
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X)
    n2, K = size(R)
    n == n2 || throw(DimensionMismatch("Dimension of X and R mismatch."))
    # allocate memory for temporary matrices
    Xo = copy(X)
    tmat = zeros(T, d, d)

    # allocate memory for llh
    llh = Vector{T}(undef, maxiter)
    fill!(llh, -Inf32)

    @showprogress 0.1 "EM for gmm..." for iter ∈ 2:maxiter
        # M-step
        M!(w, μ, Σ, R, X, Xo)
        # E-step
        llh[iter] = E!(R, X, w, μ, Σ, Xo, tmat)
        incr = llh[iter] - llh[iter-1]
        if abs(incr / llh[iter-1]) < tol || iter == maxiter
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function E!(
    R::AbstractArray{T}, X::AbstractArray{T}, w::AbstractVector{T},
    μ::AbstractArray{T}, Σ::AbstractVector{AbstractArray{T}},
    Xo::AbstractArray{T}, tmat::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    @inbounds for k ∈ 1:K
        expectation!(
            view(R, :, k), tmat, Xo, X, 
            view(μ, :, k), 
            view(Σ, :, :, k)
        )
    end
    R .+= log.(w')
    llh = sum(R)
    softmax!(R)
    return llh
end

function expectation!(
    kllh::AbstractVector{T},
    tmat::AbstractArray{T},
    Xo::AbstractArray{T},
    X::AbstractArray{T},
    μ::AbstractArray{T},
    Σ::AbstractArray{T}
) where T <: Real
    fill!(kllh, 0)
    n, d = size(X)
    copyto!(tmat, Σ)
    copyto!(Xo, X)
    Xo .-= μ'
    C = cholesky!(tmat)
    mul!(tmat, Xo, ldiv!(C, transpose(Xo)))
    #ldiv!(tmat, C, Xo)
    kllh .-= (diag(tmat) ./ 2) .+ (logdet(C) / 2 + log(2π) * d / 2)
end

function M!(
    w::AbstractVector{T}, μ::AbstractMatrix{T}, Σ::AbstractVector{AbstractArray{T}},
    R::AbstractMatrix{T},
    X::AbstractMatrix{T}, Xo::AbstractMatrix{T},
) where T <: Real
    n, K = size(R, 2)
    # udpate parameters
    w .= sum(R, dims=1)  # remember to div by n
    mul!(μ, transpose(X), R)
    μ ./= w'
    # update Σ
    fill!(Σ, 0)
    for k ∈ 1:K
        #Σk = @view Σ[k]
        copy!(Xo, X)
        Xo .-= view(μ, :, k)
        Xo .*= sqrt.(view(R, :, k))
        mul!(Σ[k], Xo', Xo)
        Σ[k] ./= w[k]
        Σ[k] += I * 1f-8
    end
    w ./= n
end