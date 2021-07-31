module GMMFusion
using Distributions: LinearAlgebra
using Statistics: LinearAlgebra
using Distributions
using LinearAlgebra, Statistics
using StatsBase, StatsFuns
using Distributed
using ProgressMeter
using Clustering
using LoopVectorization

export EM, EM!, fusion, fusion!, GMM, FusedGMM

include("gmm.jl")
include("fusion.jl")
include("fusion2.jl")

end