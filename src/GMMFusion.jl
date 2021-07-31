module GMMFusion
using Distributions: LinearAlgebra
using Statistics: LinearAlgebra
using Distributions
using LinearAlgebra, Statistics
using StatsBase, StatsFuns
using Distributed
using ProgressMeter
using Clustering

export EM, EM!, fusion, fusion!

include("gmm.jl")
include("fusion.jl")
include("fusion2.jl")

end