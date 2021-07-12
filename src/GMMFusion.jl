module GMMFusion
export EM, EM!
using Distributions
using LinearAlgebra, Statistics
using StatsBase, StatsFuns
using Distributed
using ProgressMeter
using Clustering

include("gmm.jl")

end