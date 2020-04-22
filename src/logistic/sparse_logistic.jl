# Define logistic loss
# TODO
# [ ] Take into account slope


# Define data for logistic regression problem
struct SparseLogitData{T <: Real} <: AbstractDataset{T}
    # Store X as sparse matrix
    X::AbstractSparseMatrix{T, Int}
    y::Vector{T}
    y_pred::Vector{T}
    # Cache values of exp(y[i] * ypred[i])
    cache_σ::Vector{T}
end
function SparseLogitData(X::AbstractSparseMatrix{T, Int}, y::Vector{T}) where T
    @assert size(X, 1) == size(y, 1)
    return SparseLogitData(X, y, zeros(T, size(y, 1)), zeros(T, size(y, 1)))
end
function SparseLogitData(libsmv_file::String)
    dataset = parse_libsvm(libsmv_file)
    # Convert to sparse matrix
    X = to_sparse(dataset)
    y = copy(dataset.labels)
    format_label!(y)
    return SparseLogitData(X, y, zeros(size(y, 1)),
                           zeros(size(y, 1)))
end

ndata(dat::SparseLogitData) = length(dat.y)
nfeatures(data::SparseLogitData) = size(data.X, 2)

function predict!(data::SparseLogitData{T}, x::AbstractVector{T}) where T
    mul!(data.y_pred, data.X, x)
    # Update σ
    @inbounds for j in 1:ndata(data)
        data.cache_σ[j] = expit(-data.y_pred[j] * data.y[j])
    end
end

"""
Compute logistic loss of given vector parameter `ω ∈  R^p`.

## Complexity
O(n)

"""
function loss(ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    res = zero(T)
    n = ndata(data)
    # Could not take into account sparsity pattern here
    @inbounds for i in 1:n
        res += loss(Logit(), data.y_pred[i], data.y[i])
    end
    return one(T) / n * res
end

"""
Compute gradient of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    invn = -one(T) / ndata(data)
    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)
    for ncol in 1:nfeatures(data)
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
function hessian!(hess::AbstractVector{T}, ω::AbstractVector{T}, data::SparseLogitData{T}) where T
    error("Currently  not implemented")
    n = ndata(data)
    p = nfeatures(data)
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
    p = nfeatures(data)
    n = ndata(data)
    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)
    invn = one(T) / n

    acc = zeros(n)
    for ncol in 1:p
        @inbounds for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            @inbounds cst = invn * data.cache_σ[i] * (1.0 - data.cache_σ[i])
            @inbounds acc[i] += vec[ncol] * vals * cst
        end
    end
    for ncol in 1:p
        @inbounds for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            @inbounds hessvec[ncol] += acc[i] * vals
        end
    end
    return nothing
end
