# Define logistic loss

import Base: length

# Define data for logistic regression problem
mutable struct SparseLogitData{T <: Real} <: AbstractDataset{T}
    # Store X as sparse matrix
    X::AbstractSparseMatrix{T, Int}
    y::Vector{T}
    y_pred::Vector{T}
    hash_x::UInt64
    # Cache values of exp(y[i] * ypred[i])
    cache_σ::Vector{T}
end

function SparseLogitData(X::AbstractSparseMatrix{T, Int}, y::Vector{T}) where T
    @assert size(X, 1) == size(y, 1)
    return SparseLogitData(X, y, zeros(T, size(y, 1)), UInt64(0), zeros(T, size(y, 1)))
end

function SparseLogitData(libsmv_file::String)
    dataset = parse_libsvm(libsmv_file)
    # Convert to sparse matrix
    X = to_sparse(dataset)
    y = copy(dataset.labels)
    format_label!(y)
    return SparseLogitData(X, y, zeros(size(y, 1)), UInt64(0),
                           zeros(size(y, 1)))
end
length(dat::SparseLogitData) = length(dat.y)

function _update_ypred!(data::SparseLogitData{T}, x::AbstractVector{T}) where T
    new_hash = hash(x)
    if new_hash != data.hash_x
        mul!(data.y_pred, data.X, x)
        data.hash_x = new_hash
        # Update σ
        @inbounds for j in 1:length(data)
            data.cache_σ[j] = expit(-data.y_pred[j] * data.y[j])
        end
    end
end

"""
Compute logistic loss of given vector parameter `ω ∈  R^p`.

## Complexity
O(n)

"""
function loss(ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    # Compute dot product
    _update_ypred!(data, ω)
    res = zero(T)
    # Could not take into account sparsity pattern here
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
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    fill!(grad, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    invn = -one(T) / length(data)
    n = dim(data)

    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)
    for ncol in 1:size(data.X, 2)
        @inbounds for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            @inbounds tmp = invn * data.y[i] * data.cache_σ[i]
            @inbounds grad[ncol] += tmp * vals
        end
    end
    return nothing
end

"""
Compute Hessian of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p * (p-1) /2)

"""
function hess!(hess::AbstractVector{T}, ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    error("Currently  not implemented")
    fill!(hess, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    n = length(data)
    p = dim(data)
    invn = one(T) / n

    rows = rowvals(data.X)
    vals = nonzeros(data.X)

    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (one(T) - σz)

        xi = data.X[i, :]
        #= for i, vi in findnz(xi) =#
        #=     for j, vj in findnz(xi) =#
        #=         count = i * (j-1) + j =#
        #=         @inbounds hess[count] += cst * vi * vj =#
        #=     end =#
        #= end =#
    end
    return nothing
end

"""
Compute Hessian-vector product of logistic loss for given vector
parameter `ω ∈  R^p`.

## Complexity
O(n * 2 * p)

"""
function hessvec!(hessvec::AbstractVector{T}, ω::AbstractVector{T},
                  vec::AbstractVector{T}, data::SparseLogitData{T}) where T
    fill!(hessvec, zero(T))
    # Compute dot product
    _update_ypred!(data, ω)
    p = dim(data)
    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)

    invn = one(T) / length(data)

    acc = zeros(length(data))
    for ncol in 1:size(data.X, 2)
        @inbounds for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            @inbounds cst = invn * data.cache_σ[i] * (1.0 - data.cache_σ[i])
            @inbounds acc[i] += vec[ncol] * vals * cst
        end
    end

    for ncol in 1:size(data.X, 2)
        @inbounds for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            @inbounds hessvec[ncol] += acc[i] * vals
        end
    end
    return nothing
end