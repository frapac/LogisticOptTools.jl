import Base: length

# Define data for logistic regression problem
mutable struct DualLogitData{T <: Real} <: AbstractDataset{T}
    X::AbstractArray{T}
    y::Vector{T}
    x_pred::Vector{T}
end
function DualLogitData(X::Array{T, 2}, y::Vector{T}) where T
    n_feats = size(X, 2)
    return DualLogitData(X, y, zeros(T, n_feats))
end
function DualLogitData(libsmv_file::String; scale_data=false)
    dataset = parse_libsvm(libsmv_file)
    # Convert to dense matrix
    X = to_dense(dataset)
    if scale_data
        scale!(NormalScaler(), X)
    end
    y = dataset.labels
    # Special preprocessing for covtype
    format_label!(y)
    return DualLogitData(X, y)
end

ndata(dat::DualLogitData) = length(dat.y)
nfeatures(dat::DualLogitData) = size(dat.X, 2)

lowerbound(data::DualLogitData) = -0.5 * data.y .- 0.5
upperbound(data::DualLogitData) = -0.5 * data.y .+ 0.5

function predict!(data::DualLogitData, x)
    mul!(data.x_pred, data.X', x)
end

"""
Compute logistic loss of given vector parameter `λ ∈  R^n`.

## Complexity
O(n)

"""
function loss(ω::AbstractVector{T}, data::DualLogitData{T}) where T
    res = zero(T)
    @inbounds for i in 1:ndata(data)
        res += loss(LogLoss(), ω[i], data.y[i])
    end
    return -res
end

"""
Compute gradient of logistic loss for given parameter `λ ∈  R^n`.

## Complexity
O(n)

"""
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T}, data::DualLogitData{T}) where T
    n = ndata(data)
    invn = one(T) / n
    @inbounds for j in 1:n
        λtmp = ω[j] * data.y[j]
        grad[j] = data.y[j] * (log1p(λtmp) - log(-λtmp))
    end
    return nothing
end

function hessvec!(hv::AbstractVector{T}, ω::AbstractVector{T},
                  vec::AbstractVector{T}, data::DualLogitData{T}) where T
    n = ndata(data)
    invn = one(T) / n
    @inbounds for j in 1:n
        λtmp = ω[j] * data.y[j]
        hv[j] = (-1.0 / λtmp + 1.0 / (1.0 + λtmp)) * vec[j]
    end
    return nothing
end

