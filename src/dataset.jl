
# Dataset object
abstract type AbstractDataset{T <: Real} end

function nfeatures(data::AbstractDataset) end

function ndata(data::AbstractDataset) end

function predict!(data::AbstractDataset, x::AbstractVector, intercept) end

function loss(x::AbstractVector, data::AbstractDataset) end

function gradient!(grad::AbstractVector, x::AbstractVector,
                   data::AbstractDataset, fit_intercept::Bool) end

function hessian!(hess::AbstractVector, x::AbstractVector,
                   data::AbstractDataset, fit_intercept::Bool) end

function diaghess!(diagh::AbstractVector, x::AbstractVector,
                   data::AbstractDataset, fit_intercept::Bool) end

function hessvec!(hv::AbstractVector, x::AbstractVector,
                  data::AbstractDataset, fit_intercept::Bool) end

struct EmptyDataset{T <: Real}  <: AbstractDataset{T} end

ndata(::EmptyDataset) = 0
nfeatures(::EmptyDataset) = 0
loss(::EmptyDataset) = NaN

