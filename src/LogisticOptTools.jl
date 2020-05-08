module LogisticOptTools

using LinearAlgebra, Statistics, SparseArrays
using Optim

include("utils.jl")
include("loggers.jl")
include("io/libsvm_parser.jl")

include("dataset.jl")
include("loss.jl")
include("penalty.jl")

include("logistic/dense_logistic.jl")
include("logistic/sparse_logistic.jl")
include("logistic/dual_logistic.jl")

include("model.jl")

end # module
