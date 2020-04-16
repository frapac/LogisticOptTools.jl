module LogisticOptTools

using LinearAlgebra, Statistics, SparseArrays

BLAS.set_num_threads(1)

# Dataset object
abstract type AbstractDataset{T <: Real} end
include("utils.jl")
include("loggers.jl")
include("io/libsvm_parser.jl")

include("loss.jl")
include("penalty.jl")

include("logistic/dense_logistic.jl")
include("logistic/sparse_logistic.jl")
include("logistic/dual_logistic.jl")

include("model.jl")

end # module
