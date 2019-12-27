
abstract type AbstractDataset{T <: Real} end

abstract type AbstractModel end

# define logistic problem
struct GeneralizedLinearModel{T <: Real} <: AbstractModel
    data::AbstractDataset{T}
    penalty::AbstractPenalty
    logger::AbstractLogger
end
GeneralizedLinearModel(data::AbstractDataset{T}, penalty::AbstractPenalty) where T = GeneralizedLinearModel(data, penalty, NoLogger())

function loss(θ::AbstractVector{T}, model::GeneralizedLinearModel{T}) where T
    obj = loss(θ, model.data) + model.penalty(θ)
    return obj
end

function gradient!(grad::AbstractVector{T}, θ::AbstractVector{T}, model::GeneralizedLinearModel{T}) where T
    gradient!(grad, θ, model.data)
    gradient!(grad, θ, model.penalty)
end

function hess!(hess::AbstractVector{T}, x::AbstractVector{T}, model::GeneralizedLinearModel{T}) where T
    hess!(hess, x, model.data)
    hess!(hess, x, model.penalty)
end

function hessvec!(hessvec::AbstractVector{T}, θ::AbstractVector{T}, vec::AbstractVector{T}, model::GeneralizedLinearModel{T}) where T
    hessvec!(hessvec, θ, vec, model.data)
    hessvec!(hessvec, θ, vec, model.penalty)
end
