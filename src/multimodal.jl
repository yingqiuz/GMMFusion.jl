"""
interface normal version
"""
function EM(
    X1::AbstractArray{T}, X2::AbstractArray{T}, K::Int;
    init::Union{AbstractArray{T}, Nothing}=nothing,
    tol::T=convert(T, 1e-6), maxiter::Int=10000
) where T <: Real
    n, d = size(X2)
    # init 
    R = kmeans(X1', K; init=init, tol=tol, max_iters=maxiter)
    w = convert(Array{T}, reshape(counts(R.assignments) ./ n, 1, K))  # cluster size
    μ1 = copy(R.centers)
    Σ1 = [cholesky!(cov(X1)) for k ∈ 1:K]
    @debug "μ1" μ1
    # μ2, Σ2
    μ2 = Matrix{T}(undef, d, K)
    for k ∈ 1:K
        μ2[:, k] .= mean(X2[findall(x -> x == k, R.assignments), :], dims=1)[:]
    end
    @debug "μ2" μ2
    Σ2 = [cholesky!(cov(X2)) for k ∈ 1:K]
    R = [x == k ? 1 : 0 for x ∈ R.assignments, k ∈ 1:K]
    #model = GMM(d, K, ones(T, K) ./ K, μ, Σ)
    EM!(convert(Array{T}, R), deepcopy(X1), deepcopy(X2), w, μ1, μ2, Σ1, Σ2; 
        tol=tol, maxiter=maxiter)
end

function EM!(
    R::AbstractArray{T}, X1::AbstractArray{T}, X2::AbstractArray{T}, 
    w::AbstractArray{T}, μ1::AbstractMatrix{T}, μ2::AbstractMatrix{T}, 
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}};
    tol::T=convert(T, 1e-6), maxiter::Int=1000
) where T <: Real
    n, _ = size(X1)
    n2, K = size(R)
    n == n2 || throw(DimensionMismatch("Dimension of X and R mismatch."))
    # allocate memory for temporary matrices
    #X1o = copy(X1)
    #X2o = copy(X2)
    #covmat = zeros(T, n, n)

    # allocate memory for llh
    llh = Vector{T}(undef, maxiter)
    fill!(llh, -Inf32)
    prog = ProgressMeter.ProgressUnknown()
    @showprogress 0.1 "EM for gmm..." for iter ∈ 2:maxiter
        # E-step
        @debug "R" R
        @debug "w" w
        @debug "μ" μ1, μ2
        @debug "Σ" Σ1, Σ2
        llh[iter] = E!(R, X1, X2, w, μ1, μ2, Σ1, Σ2)
        # M-step
        M!(w, μ1, μ2, Σ1, Σ2, R, X1, X2)
        incr = (llh[iter] - llh[iter-1]) / llh[iter-1]
        @info "iteration $(iter-1), incr" incr
        if abs(incr) < tol || iter == maxiter
            iter != maxiter || @warn "Not converged after $(maxiter) steps"
            return R
        end
    end
end

function E!(
    R::AbstractArray{T},  X1::AbstractArray{T}, X2::AbstractArray{T}, 
    w::AbstractArray{T}, μ1::AbstractArray{T}, μ2::AbstractArray{T}, 
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}},
) where T <: Real
    n, K = size(R)
    Threads.@threads for k ∈ 1:K
        Rk = view(R, :, k)
        expectation!(
            Rk, X1, X2, view(μ1, :, k), view(μ2, :, k), Σ1[k], Σ2[k]
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
    X1::AbstractArray{T}, X2::AbstractArray{T},
    μ1::AbstractVector{T}, μ2::AbstractVector{T},
    C1::Cholesky{T, Matrix{T}}, C2::Cholesky{T, Matrix{T}}
) where T <: Real
    _, d1 = size(X1)
    _, d2 = size(X2)
    #copyto!(X1o, X1)
    X1o = deepcopy(X1)
    X1o .-= μ1'
    #copyto!(X2o, X2)
    X2o = deepcopy(X2)
    X2o .-= μ2'
    #@debug "X Xo" X sum(Xo, dims=1)
    #C = cholesky!(Hermitian(Σ))
    fill!(Rk, -logdet(C1) / 2 -logdet(C2) / 2 - log(2π) * d1 / 2 - log(2π) * d2 / 2)
    #rdiv!(Xo, C.U)
    #@turbo for nn ∈ 1:n, dd ∈ 1:d
    #    Rk[nn] -= Xo[nn, dd] ^ 2 / 2 
    #end
    #mul!(covmat, X1o, C1 \ transpose(X1o))
    #@debug "covmat" diag(covmat)
    Rk .-= diag(X1o * (C1 \ transpose(X1o))) ./ 2

    #mul!(covmat, X2o, C2 \ transpose(X2o))
    #@debug "covmat" diag(covmat)
    Rk .-= diag(X2o * (C2 \ transpose(X2o))) ./ 2
end


function M!(
    w::AbstractArray{T}, μ1::AbstractMatrix{T}, μ2::AbstractMatrix{T}, 
    Σ1::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    Σ2::AbstractVector{A} where A <: Cholesky{T, Matrix{T}}, 
    R::AbstractMatrix{T}, 
    X1::AbstractMatrix{T}, X2::AbstractMatrix{T}, 
) where T <: Real
    n, K = size(R)
    # udpate parameters
    w .= sum(R, dims=1) # remember to div by n
    @debug "w" w
    mul!(μ1, transpose(X1), R)
    μ1 ./= w
    mul!(μ2, transpose(X2), R)
    μ2 ./= w
    # update Σ
    Threads.@threads for k ∈ 1:K
        Rk = view(R, :, k)
        #copy!(X1o, X1)
        X1o = deepcopy(X1)
        X1o .-= transpose(view(μ1, :, k))
        X1o .*= sqrt.(Rk)
        Σ1[k] = cholesky!(X1o' * X1o ./ w[1, k] + I * 1f-8)

        #copy!(X2o, X2)
        X2o = deepcopy(X2)
        X2o .-= transpose(view(μ2, :, k))
        X2o .*= sqrt.(Rk)
        Σ2[k] = cholesky!(X2o' * X2o ./ w[1, k] + I * 1f-8)
    end
    w ./= n
end