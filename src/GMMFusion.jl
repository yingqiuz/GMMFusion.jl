module GMMFusion
using Distributions
using LinearAlgebra, Statistics
using StatsBase, StatsFuns
using Distributed
using ProgressMeter
using NIfTI, HDF5
using Folds
using OnlineStats: Mean
using LoopVectorization
using ParallelKMeans

export EM, EM!

include("gmm.jl")
include("fusion.jl")
end