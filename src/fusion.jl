"""
interface
"""
function EM(
    XH::AbstractArray{T}, XL::AbstractArray{T}, XLtest::AbstractArray{T}, K::Int;
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp,
    tol::T=convert(T, 1e-6), maxiter::Int=10000
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

    EM!(R, XH, XL, w, μH, μL, ΣH, ΣL; tol=tol, maxiter=maxiter)
    ntest = size(XLtest, 1)
    Rtest = zeros(T, ntest, K)
    covmat = zeros(T, ntest, ntest)
    Xo = copy(XLtest)
    E!(Rtest, XLtest, w, μL, ΣL, Xo, covmat)
    return Rtest
end

"""
main algorithm
"""
function EM!(
    R::AbstractArray{T}, XH::AbstractArray{T}, XL::AbstractArray{T},
    w::AbstractMatrix{T}, μH::AbstractMatrix{T}, μL::AbstractMatrix{T}, 
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}};
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(XH)
    n == size(XL, 1) || throw(DimensionMismatch("Dimensions of XH and XL mismatch."))
    d == size(XL, 2) || throw(DimensionMismatch("Dimensions of XH and XL mismatch."))
    K = size(R, 2)

    # likelihood
    llh = fill(-Inf32, maxiter)
    # pre-allocate memory for centralised data
    XHo = copy(XH)
    XLo = copy(XL)
    covH, covL = [zeros(T, n, n) for _ ∈ 1:2]

    @showprogress 0.1 "EM..." for iter ∈ 2:maxiter 
        # E-step
        llh[iter] = E!(R, XH, w, μH, ΣH, XHo, covH)
        #llh[iter] = E!(R, XH, XL, w, μL, ΣH, ΣL, XHo, XLo, covH, covL)
        @debug "llh" llh[iter]
        # M-step
        M!(w, μH, ΣH, R, XH, XHo)
        M!(μL, ΣL, R, XL, XLo)
        #M!(w, μL, U, ΣH, ΣL, R, XH, XL, XHo, XLo)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        @debug "U" U
        @debug "w" w
        @debug "μL" μL
        @debug "ΣL" ΣL
        @debug "ΣH" ΣH
        @info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end

end

function M!(
    μ::AbstractMatrix{T}, Σ::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractMatrix{T}, X::AbstractMatrix{T}, Xo::AbstractMatrix{T}
) where T <: Real
    n, K = size(R)
    # udpate parameters
    @debug "w" w
    mul!(μ, transpose(X), R)
    μ ./= (w .* n)
    # update Σ
    @inbounds for k ∈ 1:K
        copy!(Xo, X)
        Xo .-= transpose(view(μ, :, k))
        Xo .*= sqrt.(view(R, :, k))
        Σ[k] = cholesky!(Xo' * Xo ./ (w[1, k] .* n) + I * 1f-8)
    end
end