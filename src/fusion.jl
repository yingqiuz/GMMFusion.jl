"""
interface
"""
function EM(
    XH::AbstractArray{T}, XL::AbstractArray{T}, XLtest::AbstractArray{T}, K::Int;
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp,
    tol::T=convert(T, 1e-6), maxiter::Int=1000, transform_high::Bool=false
) where T <: Real
    n, d = size(XH)
    n == size(XL, 1) || throw(DimensionMismatch("Size of XH and XL mismatch."))
    # init - high quality
    RH = kmeans(XH', K; init=init, tol=tol, maxiter=maxiter)
    a = assignments(RH)
    w = convert(Array{T}, reshape(counts(RH) ./ n, 1, K))  # cluster size
    μH = copy(RH.centers)
    @debug "μH" μH
    ΣH = [cholesky!(cov(XH)) for k ∈ 1:K]
    R = convert(Array{T}, [x == k ? 1 : 0 for x ∈ a, k ∈ 1:K])

    # init - low quality
    #RL = kmeans(XL', K; init=init, tol=tol, maxiter=maxiter)
    #μL = copy(RL.centers)
    μL = Matrix{T}(undef, d, K)
    for k ∈ 1:K
        μL[:, k] .= mean(XL[findall(x -> x == k, a), :], dims=1)[:]
    end
    @debug "μL" μL
    ΣL = [cholesky!(cov(XL)) for k ∈ 1:K]
    # temporary variables to reduce memory allocation
    XHo = copy(XH)
    XLo = copy(XL)
    U = Matrix{T}(undef, d, d)

    if transform_high
        # find initial transformation U
        updateU!(U, XH, XHo, R, μL, K)
        @debug "U" U
        # update ΣH, ΣL
        updateΣ!(ΣH, ΣL, XH, XHo, XL, XLo, w .* n, μL, U, R, K)
        @debug "U" U
        @debug "w" w
        @debug "ΣL" ΣL
        @debug "ΣH" ΣH
        EM!(R, XH, XL, w, μL, ΣH, ΣL, U; tol=tol, maxiter=maxiter)
        #return FusedGMM(K, d, w, μH, ΣH, ΣL, U)
        ntest = size(XLtest, 1)
        Rtest = zeros(T, ntest, K)
        covmat = zeros(T, ntest, ntest)
        Xo = copy(XLtest)
        E!(Rtest, XLtest, w, μL, ΣL, Xo, covmat)
        return Rtest
    else
        #zu, _, v = svd!(μH * μL')
        #U = u * v'
        updateU!(U, XL, XLo, R, μH, K)
        @debug "U" U
        # update ΣH, ΣL
        updateΣ!(ΣL, ΣH, XL, XLo, XH, XHo, w .* n, μH, U, R, K)
        @debug "U" U
        @debug "w" w
        @debug "ΣL" ΣL
        @debug "ΣH" ΣH
        EM!(R, XL, XH, w, μH, ΣL, ΣH, U; tol=tol, maxiter=maxiter)
        #return FusedGMM(K, d, w, μH, ΣH, ΣL, U)
        ntest = size(XLtest, 1)
        Rtest = zeros(T, ntest, K)
        covmat = zeros(T, ntest, ntest)
        Xo = copy(XLtest * U')
        copyto!(XLtest, Xo)
        E!(Rtest, XLtest, w, μH, ΣL, Xo, covmat)
        return Rtest
    end
end

"""
main algorithm
"""
function EM!(
    R::AbstractArray{T}, X1::AbstractArray{T}, X2::AbstractArray{T},
    w::AbstractMatrix{T}, μ::AbstractMatrix{T}, 
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    U::AbstractMatrix{T}; tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X1)
    n == size(X2, 1) || throw(DimensionMismatch("Dimensions of X1 and X2 mismatch."))
    d == size(X2, 2) || throw(DimensionMismatch("Dimensions of X1 and X2 mismatch."))
    K = size(R, 2)

    # likelihood
    llh = fill(-Inf32, maxiter)
    # pre-allocate memory for centralised data
    X1o = copy(X1)
    X2o = copy(X2)
    cov1, cov2 = [zeros(T, n, n) for _ ∈ 1:2]

    @showprogress 0.1 "EM..." for iter ∈ 2:maxiter 
        # E-step
        llh[iter] = E!(R, X1, X2, w, μ, Σ1, Σ2, X1o, X2o, cov1, cov2, U)
        @debug "llh" llh[iter]
        # M-step
        M!(w, μ, U, Σ1, Σ2, R, X1, X2, X1o, X2o)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        @debug "U" U
        @debug "w" w
        @debug "μ" μ
        @debug "Σ2" Σ2
        @debug "Σ1" Σ1
        @info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end

end

"""
E step
"""
function E!(
    R::AbstractArray{T}, X1::AbstractArray{T}, X2::AbstractArray{T}, 
    w::AbstractArray{T}, μ::AbstractArray{T}, 
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    X1o::AbstractArray{T}, X2o::AbstractArray{T}, 
    cov1::AbstractArray{T}, cov2::AbstractArray{T},
    U::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    @inbounds for k ∈ 1:K
        expectation!(
            view(R, :, k), X1, X2, X1o, X2o, cov1, cov2,
            view(μ, :, k), Σ1[k], Σ2[k], U
        )
    end
    R .+= log.(w)
    llh = logsumexp(R, dims=2)
    R .-= llh
    @avx R .= exp.(R)
    return sum(llh) / n
end

function expectation!(
    Rk::AbstractVector{T},
    X1::AbstractArray{T}, X2::AbstractArray{T},
    X1o::AbstractArray{T}, X2o::AbstractArray{T},
    cov1::AbstractMatrix{T}, cov2::AbstractMatrix{T},
    μ::AbstractVector{T},
    C1::Cholesky{T, Matrix{T}}, C2::Cholesky{T, Matrix{T}},
    U::AbstractArray{T}
) where T <: Real
    _, d1 = size(X1)
    _, d2 = size(X2)
    # centralise data
    copyto!(X2o, X2)
    X2o .-= μ'
    copyto!(X1o, X1 * U')
    X1o .-= μ'
    #@debug "X Xo" X sum(Xo, dims=1)
    #CL, CH = map(x -> cholesky!(Hermitian(x)), [ΣL, ΣH])
    fill!(Rk, -logdet(C1) / 2 - log(2π) * d1 / 2 - logdet(C2) / 2 - log(2π) * d2 / 2)
    mul!(cov2, X2o, C2 \ transpose(X2o))
    mul!(cov1, X1o, C1 \ transpose(X1o))
    #@debug "covmat" diag(covmat)
    Rk .-= diag(cov2) ./ 2 .+ diag(cov1) ./ 2
end

function M!(
    w::AbstractMatrix{T}, μ::AbstractMatrix{T}, U::AbstractArray{T},
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractArray{T}, X1::AbstractArray{T}, X2::AbstractArray{T},
    X1o::AbstractArray{T}, X2o::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    # udpate parameters
    w .= sum(R, dims=1) # remember to div by n
    # update μL
    updateμ!(μ, R, X1, X1o, X2, X2o, w, Σ1, Σ2, U, K)
    # update U
    updateU!(U, X1, X1o, R, μ, K)
    # update ΣH, ΣL
    updateΣ!(Σ1, Σ2, X1, X1o, X2, X2o, w, μ, U, R, K)
    w ./= n
end

function updateμ!(
    μ::AbstractArray{T}, R::AbstractArray{T}, X1::AbstractArray{T}, X1o::AbstractArray{T},
    X2::AbstractArray{T}, X2o::AbstractArray{T}, w::AbstractArray{T},
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    U::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    # update μL
    @inbounds for k ∈ 1:K
        μk = view(μ, :, k)
        Rk = view(R, :, k)
        #CL, CH = map(x -> cholesky!(Hermitian(x)), [ΣL[k], ΣH[k]])
        copyto!(X1o, X1 * U')
        copyto!(X2o, X2)
        rdiv!(X1o, Σ1[k])
        rdiv!(X2o, Σ2[k])
        mul!(μk, transpose(X1o + X2o), Rk)
        ldiv!(cholesky!(LinearAlgebra.inv!(Σ1[k]) + LinearAlgebra.inv!(Σ2[k])), μk)
        μk ./= w[1, k]
    end
end

function updateU!(
    U::AbstractArray{T},
    X::AbstractArray{T}, Xo::AbstractArray{T},
    R::AbstractArray{T}, μ::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    fill!(U, 0)
    @inbounds for k ∈ 1:K
        Rk = view(R, :, k)
        copyto!(Xo, X)
        Xo .*= sqrt.(Rk)
        #lmul!(Diagonal(sqrt.(Rk)), Xo)
        U .+= transpose(Xo) * Xo
    end
    #C = cholesky!(U)
    #mul!(U, μL * transpose(R), XH)
    #rdiv!(U, C)
    # or orthogonal
    u, _, v = svd!(μ * transpose(R) * X * U)
    mul!(U, u, v')
end

function updateΣ!(
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    X1::AbstractArray{T}, X1o::AbstractArray{T},
    X2::AbstractArray{T}, X2o::AbstractArray{T}, 
    w::AbstractArray{T}, μ::AbstractArray{T}, U::AbstractArray{T},
    R::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    @inbounds for k ∈ 1:K
        μk = @view μ[:, k]
        copyto!(X1o, X1 * U')
        X1o .-= μk'
        copyto!(X2o, X2)
        X2Lo .-= μk'
        map([X1o, X2o]) do x
            x .*= sqrt.(view(R, :, k))
        end
        #Xo .*= sqrt.(view(R, :, k))
        Σ1[k] = cholesky!((X1o' * X1o) ./ w[1, k] + I * 1f-8)
        Σ2[k] = cholesky!((X2o' * X2o) ./ w[1, k] + I * 1f-8)
        #update!(ΣH[k], XoH, w[1, k])
        #update!(ΣL[k], XoL, w[1, k])
    end
end