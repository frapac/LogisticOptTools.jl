# Define logistic loss
# TODO
# [ ] Memoize expit(-yₚ[i] y[i])

import Base: length

# Define data for logistic regression problem
struct LogitData{T <: Real} <: AbstractDataset{T}
    # Features x_ij
    X::Array{T, 2}
    # Labels y_i
    y::Vector{T}
    # Current prediction. Should be equal to dot(X, p) with p
    # current parameters.
    y_pred::Vector{T}
end

function LogitData(X::Array{T, 2}, y::Vector{T}) where T
    @assert size(X, 1) == size(y, 1)
    return LogitData(X, y, zeros(T, size(y, 1)))
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
    return LogitData(X, y, zeros(size(y, 1)))
end

nfeatures(dat::LogitData) = size(dat.X, 2)
ndata(dat::LogitData) = length(dat.y)

function predict!(data::LogitData{T}, x::AbstractVector{T}, intercept=NaN) where T
    mul!(data.y_pred, data.X, x)
    if isfinite(intercept)
        data.y_pred .+= intercept
    end
    return
end

"""
Compute logistic loss of given vector parameter `ω ∈  R^p`.

## Complexity
O(n)

"""
function loss(ω::AbstractVector{T}, data::LogitData{T}) where T
    # Compute dot product
    res = zero(T)
    @inbounds for i in 1:ndata(data)
        res += loss(Logit(), data.y_pred[i], data.y[i])
    end
    return one(T) / ndata(data) * res
end

"""
Compute gradient of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function gradient!(grad::AbstractVector{T}, ω::AbstractVector{T},
                   data::LogitData{T}, fit_intercept::Bool=false) where T
    nfeats = nfeatures(data)
    n = ndata(data)
    invn = -one(T) / n
    @inbounds for j in 1:n
        tmp = invn * data.y[j] * expit(-data.y_pred[j] * data.y[j])
        for i in 1:nfeats
            grad[i] += tmp * data.X[j, i]
        end
        if fit_intercept
            grad[end] += tmp
        end
    end
    return nothing
end

"""
Compute Hessian of logistic loss for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p * (p-1) /2)

"""
function hessian!(hess::AbstractVector{T}, ω::AbstractVector{T}, data::LogitData{T},
                  fit_intercept::Bool=false) where T
    # Sanity check
    p = nfeatures(data)
    n = ndata(data)
    invn = one(T) / n
    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (one(T) - σz)
        count = 1
        for j in 1:p
            for k in j:p
                hess[count] += cst * data.X[i, j] * data.X[i, k]
                count += 1
            end
            if fit_intercept
                hess[count] += cst * data.X[i, j]
                count += 1
            end
        end
        if fit_intercept
            hess[count] += cst
        end
    end
    return nothing
end

"""
Compute diagonal of Hessian for given vector parameter `ω ∈  R^p`.

## Complexity
O(n * p)

"""
function diaghess!(diagh::AbstractVector{T}, ω::AbstractVector{T},
                   data::LogitData{T}, fit_intercept::Bool=false) where T
    # Sanity check
    n = ndata(data)
    p = nfeatures(data)
    @assert length(ω) == length(diagh)
    invn = one(T) / n
    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (one(T) - σz)
        for j in 1:p
            diagh[j] += cst * data.X[i, j] * data.X[i, j]
        end
        if fit_intercept
            diagh[end] += cst
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
                  vec::AbstractVector{T}, data::LogitData{T}, fit_intercept::Bool=false) where T
    p = nfeatures(data)
    n = ndata(data)
    invn = one(T) / n
    @inbounds for i in 1:n
        σz = expit(-data.y_pred[i] * data.y[i])
        cst = invn * σz * (T(1) - σz)
        acc = zero(T)
        @simd for j in 1:p
            acc += data.X[i, j] * vec[j]
        end
        pσ = cst * acc
        for j in 1:p
            hessvec[j] += pσ * data.X[i, j]
        end

        if fit_intercept
            for j in 1:p
                hessvec[j] += cst * data.X[i, j] * vec[end]
            end
            hessvec[end] += pσ + cst * vec[end]
        end
    end
    return nothing
end
