module GMMFusion

using Distributions
using LinearAlgebra
using Statistics, StatsBase, StatsFuns
using Distributed
using ProgressMeter
using Clustering
using LoopVectorization
using Parameters
using SpecialFunctions
using Flux: softmax!, onecold, onehot

export EM, EM!, GMM, FusedGMM, predict

struct GMM{T<:Real}
    K::Int
    d::Int
    w::AbstractVector{T}
    μ::AbstractArray{T}
    Σ::AbstractVector
end

struct FusedGMM{T<:Real}
    K::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    w::AbstractVector{T}               # weights: n
    μ::AbstractArray{T}                       # means: n x d
    ΣH::AbstractVector
    ΣL::AbstractVector
    U::AbstractArray{T}
    #hist::Array{History}           # history of this GMM
end

include("gmm.jl")
include("fusion.jl")
include("vb.jl")

end