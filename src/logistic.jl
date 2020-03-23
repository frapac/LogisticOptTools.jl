# Define logistic loss
# TODO
# [ ] Memoize expit(-yₚ[i] y[i])

import Base: length

# Define data for logistic regression problem
mutable struct LogitData{T <: Real} <: AbstractDataset{T}
    X::Array{T, 2}
    y::Vector{T}
    y_pred::Vector{T}
    hash_x::UInt64
end

function LogitData(X::Array{T, 2}, y::Vector{T}) where T
    @assert size(X, 1) == size(y, 1)
    return LogitData(X, y, zeros(T, size(y, 1)), UInt64(0))
end

function LogitData(libsmv_file::String; scale_data=false)
    dataset = parse_libsvm(libsmv_file)
    # Convert to dense matrix
    X = to_dense(dataset)
    if scale_data
        scale!(NormalScaler(), X)
    end
    y = dataset.labels
    # Special preprocessing for covtype
    format_label!(y)
    return LogitData(X, y, zeros(size(y, 1)), UInt64(0))
end

length(dat::LogitData) = length(dat.y)
dim(dat) = size(dat.X, 2)

function _update_ypred!(data::LogitData{T}, x::AbstractVector{T}) where T
    new_hash = hash(x)
    if new_hash != data.hash_x
        mul!(data.y_pred, data.X, x)
        data.hash_x = new_hash
    end
end

"""
Compute logistic loss of given vector parameter `ω ∈  R^p`.

## Complexity
O(n)

"""
function loss(ω::AbstractVector{T}, data::LogitData{T}) where T
    # Sanity check
    @assert length(ω) == dim(data)
    # Compute dot product
    _update_ypred!(data, ω)
    res = zero(T)
    @inbounds for i in 1:length(data)
        res += loss(Logit(), data.y_pred[i], data.y[i])
    end
    return one(T) / length(data) * res
end

"""
Compute gradient of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T}, data::LogitData{T}) where T
    # Sanity check
    @assert length(ω) ==  length(grad) == dim(data)
    # Reset grad
    fill!(grad, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    invn = -one(T) / length(data)
    @inbounds for j in 1:length(data)
        tmp = invn * data.y[j] * expit(-data.y_pred[j] * data.y[j])
        @inbounds for i in 1:length(ω)
            grad[i] += tmp * data.X[j, i]
        end
    end
    return nothing
end

"""
Compute Hessian of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p * (p-1) /2)

"""
function hess!(hess::AbstractVector{T}, ω::AbstractVector{T}, data::LogitData{T}) where T
    # Sanity check
    p = dim(data)
    @assert length(ω) == p
    @assert length(hess) == p * (p + 1) / 2
    # Reset Hessian
    fill!(hess, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    n = length(data)
    invn = one(T) / n
    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (one(T) - σz)
        count = 1
        for j in 1:p
            for k in j:p
                @inbounds hess[count] += cst * data.X[i, j] * data.X[i, k]
                count += 1
            end
        end
    end
    return nothing
end

"""
Compute diagonal of Hessian for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function diaghess!(diagh::AbstractVector{T}, ω::AbstractVector{T}, data::LogitData{T}) where T
    # Sanity check
    p = dim(data)
    @assert length(ω) == length(diagh) == p
    # Reset diagh
    fill!(diagh, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    n = length(data)
    invn = one(T) / n
    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (one(T) - σz)
        for j in 1:p
            @inbounds diagh[j] += cst * data.X[i, j] * data.X[i, j]
        end
    end
    return nothing
end

"""
Compute Hessian-vector product of logistic loss for given vector
parameter `ω ∈  R^p`.

## Complexity
O(n x 2 x p)

"""
function hessvec!(hessvec::AbstractVector{T}, ω::AbstractVector{T},
                  vec::AbstractVector{T}, data::LogitData{T}) where T
    # Sanity check
    p = dim(data)
    @assert length(ω) == length(vec) == length(hessvec) == p
    # Reset diagh
    fill!(hessvec, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    invn = one(T) / length(data)
    @inbounds for i in 1:length(data)
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (T(1) - σz)
        acc = zero(T)
        @inbounds for j in 1:p
            acc += data.X[i, j] * vec[j]
        end
        pσ = cst * acc
        @inbounds for j in 1:p
            hessvec[j] += pσ * data.X[i, j]
        end
    end
    return nothing
end
