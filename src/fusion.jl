"""
interface
"""
function EM(
    XH::AbstractArray{T}, XL::AbstractArray{T}, K::Int;
    init::Union{Symbol, SeedingAlgorithm, AbstractVector{<:Integer}}=:kmpp,
    tol::T=convert(T, 1e-6), maxiter::Int=1000
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

    # find initial transformation U
    u, _, v = svd!(μL * μH')
    U = u * v'
    @debug "U" U

    XHo = copy(XH)
    XLo = copy(XL)

    updateU!(U, XH, XHo, R, μL, K)
    # update ΣH, ΣL
    updateΣ!(ΣH, ΣL, XH, XHo, XL, XLo, w .* n, μL, U, R, K)
    @debug "U" U
    @debug "w" w
    @debug "μL" μL
    @debug "ΣL" ΣL
    @debug "ΣH" ΣH
    EM!(R, XH, XL, w, μL, ΣH, ΣL, U; tol=tol, maxiter=maxiter)
end

"""
main algorithm
"""
function EM!(
    R::AbstractArray{T}, XH::AbstractArray{T}, XL::AbstractArray{T},
    w::AbstractMatrix{T}, μL::AbstractMatrix{T}, 
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    U::AbstractMatrix{T}; tol::T=convert(T, 1e-6), maxiter::Int=10000
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
        llh[iter] = E!(R, XH, XL, w, μL, ΣH, ΣL, XHo, XLo, covH, covL, U)
        @debug "llh" llh[iter]
        # M-step
        M!(w, μL, U, ΣH, ΣL, R, XH, XL, XHo, XLo)
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

"""
E step
"""
function E!(
    R::AbstractArray{T}, XH::AbstractArray{T}, XL::AbstractArray{T}, 
    w::AbstractArray{T}, μL::AbstractArray{T}, 
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    XHo::AbstractArray{T}, XLo::AbstractArray{T}, 
    covH::AbstractArray{T}, covL::AbstractArray{T},
    U::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    @inbounds for k ∈ 1:K
        expectation!(
            view(R, :, k), XH, XL, XHo, XLo, covH, covL,
            view(μL, :, k), ΣH[k], ΣL[k], U
        )
    end
    R .+= log.(w)
    llh = logsumexp(R, dims=2)
    R .-= llh
    R .= exp.(R)
    return sum(llh) / n
end

function expectation!(
    Rk::AbstractVector{T},
    XH::AbstractArray{T}, XL::AbstractArray{T},
    XHo::AbstractArray{T}, XLo::AbstractArray{T},
    covH::AbstractMatrix{T}, covL::AbstractMatrix{T},
    μL::AbstractVector{T},
    CH::Cholesky{T, Matrix{T}}, CL::Cholesky{T, Matrix{T}},
    U::AbstractArray{T}
) where T <: Real
    n, d = size(XL)
    # centralise data
    copyto!(XLo, XL)
    XLo .-= μL'
    copyto!(XHo, XH * U')
    XHo .-= μL'
    #@debug "X Xo" X sum(Xo, dims=1)
    #CL, CH = map(x -> cholesky!(Hermitian(x)), [ΣL, ΣH])
    fill!(Rk, -logdet(CL) / 2 - log(2π) * d - logdet(CH) / 2)
    mul!(covL, XLo, CL \ transpose(XLo))
    mul!(covH, XHo, CH \ transpose(XHo))
    #@debug "covmat" diag(covmat)
    Rk .-= diag(covL) ./ 2 .+ diag(covH) ./ 2
end

function M!(
    w::AbstractMatrix{T}, μL::AbstractMatrix{T}, U::AbstractArray{T},
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractArray{T}, XH::AbstractArray{T}, XL::AbstractArray{T},
    XHo::AbstractArray{T}, XLo::AbstractArray{T}
) where T <: Real
    n, K = size(R)
    # udpate parameters
    w .= sum(R, dims=1) # remember to div by n
    # update μL
    updateμ!(μL, R, XH, XHo, XL, XLo, w, ΣH, ΣL, U, K)
    # update U
    updateU!(U, XH, XHo, R, μL, K)
    # update ΣH, ΣL
    updateΣ!(ΣH, ΣL, XH, XHo, XL, XLo, w, μL, U, R, K)
    w ./= n
end

function updateμ!(
    μL::AbstractArray{T}, R::AbstractArray{T}, XH::AbstractArray{T}, XHo::AbstractArray{T},
    XL::AbstractArray{T}, XLo::AbstractArray{T}, w::AbstractArray{T},
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    U::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    # update μL
    @inbounds for k ∈ 1:K
        μk = view(μL, :, k)
        Rk = view(R, :, k)
        #CL, CH = map(x -> cholesky!(Hermitian(x)), [ΣL[k], ΣH[k]])
        copyto!(XHo, XH * U')
        copyto!(XLo, XL)
        rdiv!(XHo, ΣH[k])
        rdiv!(XLo, ΣL[k])
        mul!(μk, transpose(XHo + XLo), Rk)
        ldiv!(cholesky!(LinearAlgebra.inv!(ΣH[k]) + LinearAlgebra.inv!(ΣL[k])), μk)
        μk ./= w[1, k]
    end
end

function updateU!(
    U::AbstractVector{A} where A <: AbstractArray{T},
    XH::AbstractArray{T}, XHo::AbstractArray{T},
    R::AbstractArray{T}, μL::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    @inbounds for k ∈ 1:K
        Rk = view(R, :, k)
        copyto!(XHo, XH)
        lmul!(Diagonal(sqrt.(Rk)), XHo)
        C = cholesky!(transpose(XHo) * XHo)
        mul!(U[k], view(μL, :, k), Rk' * XH)
        rdiv!(U[k], C)
        # orthogonalise?
        #u, _, v = svd!(view(μL, :, k) * Rk' * XH * transpose(XHo) * XHo)
        #mul!(U[k], u, v')
    end
end

function updateU!(
    U::AbstractArray{T},
    XH::AbstractArray{T}, XHo::AbstractArray{T},
    R::AbstractArray{T}, μL::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    fill!(U, 0)
    @inbounds for k ∈ 1:K
        Rk = view(R, :, k)
        copyto!(XHo, XH)
        lmul!(Diagonal(sqrt.(Rk)), XHo)
        U .+= transpose(XHo) * XHo
    end
    #C = cholesky!(U)
    #mul!(U, μL * transpose(R), XH)
    #rdiv!(U, C)
    # or orthogonal
    u, _, v = svd!(μL * transpose(R) * XH * U)
    mul!(U, u, v')
end

function updateΣ!(
    ΣH::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    ΣL::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    XH::AbstractArray{T}, XHo::AbstractArray{T},
    XL::AbstractArray{T}, XLo::AbstractArray{T}, 
    w::AbstractArray{T}, μL::AbstractArray{T}, U::AbstractArray{T},
    R::AbstractArray{T}, K::Int=size(R, 2)
) where T <: Real
    @inbounds for k ∈ 1:K
        μk = @view μL[:, k]
        copyto!(XHo, XH * U')
        XHo .-= μk'
        copyto!(XLo, XL)
        XLo .-= μk'
        map([XHo, XLo]) do x
            x .*= sqrt.(view(R, :, k))
        end
        #Xo .*= sqrt.(view(R, :, k))
        ΣH[k] = cholesky!((XHo' * XHo) ./ w[1, k] + I * 1f-8)
        ΣL[k] = cholesky!((XLo' * XLo) ./ w[1, k] + I * 1f-8)
        #update!(ΣH[k], XoH, w[1, k])
        #update!(ΣL[k], XoL, w[1, k])
    end
end