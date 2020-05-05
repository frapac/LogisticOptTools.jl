# Define logistic loss

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
function SparseLogitData(libsmv_file::String; scale_data=false)
    dataset = parse_libsvm(libsmv_file)
    # Convert to sparse matrix
    X = to_sparse(dataset)
    if scale_data
        scale!(NormalScaler(), X)
    end
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
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T},
                   data::SparseLogitData{T}, fit_intercept=false) where T
    # Sanity check
    @assert length(ω) == length(grad) == (nfeatures(data) + fit_intercept)
    invn = -one(T) / ndata(data)
    rowsv = SparseArrays.rowvals(data.X)
    xvals = SparseArrays.nonzeros(data.X)
    @inbounds for ncol in 1:nfeatures(data)
        acc = zero(T)
        @simd for j ∈  SparseArrays.nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            acc += invn * data.y[i] * data.cache_σ[i] * vals
        end
        grad[ncol] = acc
    end

    if fit_intercept
        for j in 1:ndata(data)
            @inbounds grad[end] += invn * data.y[j] * data.cache_σ[j]
        end
    end

    return nothing
end

"""
Compute Hessian of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p * (p-1) /2)

"""
function hessian!(hess::AbstractVector{T}, ω::AbstractVector{T},
                  data::SparseLogitData{T}, fit_intercept=false) where T
    error("Currently  not implemented")
end

"""
Compute Hessian-vector product of logistic loss for given vector
parameter `ω ∈  R^p`.

## Complexity
O(n * 2 * p)

"""
function hessvec!(hessvec::AbstractVector{T}, ω::AbstractVector{T},
                  vec::AbstractVector{T}, data::SparseLogitData{T},
                  fit_intercept=false) where T
    p = nfeatures(data)
    n = ndata(data)
    @assert length(ω) == length(vec) == length(hessvec) == (p + fit_intercept)
    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)
    invn = one(T) / n

    acc = zeros(n)
    @inbounds for ncol in 1:p
        for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            cst = invn * data.cache_σ[i] * (1.0 - data.cache_σ[i])
            acc[i] += vec[ncol] * vals * cst
        end
    end
    @inbounds for ncol in 1:p
        @simd for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            hessvec[ncol] += acc[i] * vals
        end
    end

    if fit_intercept
        # TODO: simplify computation of Hv for intercept
        @inbounds for ncol in 1:p
            for j in nzrange(data.X, ncol)
                i = rowsv[j]
                vals = xvals[j]
                σz = data.cache_σ[i]
                cst = invn * σz * (T(1) - σz)
                hessvec[ncol] += cst * vals * vec[end]
            end
        end
        sumcst = T(0)
        @inbounds for i in 1:n
            σz = data.cache_σ[i]
            sumcst += invn * σz * (T(1) - σz)
        end
        hessvec[end] += sum(acc) + sumcst * vec[end]
    end
    return nothing
end

"""
Compute diagonal of Hessian for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function diaghess!(diagh::AbstractVector{T}, ω::AbstractVector{T},
                   data::SparseLogitData{T}, fit_intercept=false) where T
    # Sanity check
    n = ndata(data)
    p = nfeatures(data)
    @assert length(ω) == length(diagh) == (p + fit_intercept)
    invn = one(T) / n
    rowsv = rowvals(data.X)
    xvals = nonzeros(data.X)
    @inbounds for ncol in 1:p
        acc = zero(T)
        for j in nzrange(data.X, ncol)
            i = rowsv[j]
            vals = xvals[j]
            σz = data.cache_σ[i]
            acc += invn * σz * (one(T) - σz) * vals^2
        end
        diagh[ncol] = acc
    end

    if fit_intercept
        for j in 1:n
            σz = data.cache_σ[i]
            diagh[end] += invn * σz * (one(T) - σz)
        end
    end
    return nothing
end
