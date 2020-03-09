# Model object
abstract type AbstractModel end

##################################################
# PRIMAL
##################################################
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

##################################################
# DUAL
##################################################
# Define dual model
# Currently, it suports only L2 penalty
struct DualGeneralizedLinearModel{T <: Real} <: AbstractModel
    data::AbstractDataset{T}
    penalty::L2Penalty
    logger::AbstractLogger
end
DualGeneralizedLinearModel(data::AbstractDataset{T}, penalty::L2Penalty) where T = DualGeneralizedLinearModel(data, penalty, NoLogger())

update!(model::DualGeneralizedLinearModel, x::AbstractVector) = _update_xpred!(model.data, x)

function loss(θ::AbstractVector{T},
              model::DualGeneralizedLinearModel{T}) where T
    update!(model, θ)
    obj = loss(θ, model.data) + model.penalty(model.data.x_pred)
    return obj
end

function gradient!(grad::AbstractVector{T}, θ::AbstractVector{T},
                   model:: DualGeneralizedLinearModel{T}) where T
    update!(model, θ)
    gradient!(grad, θ, model.data)
    # compute gradient of penalty ||X^\top θ||_2^2 in two steps
    gpenal = zeros(length(model.data.x_pred))
    gradient!(gpenal, model.data.x_pred, model.penalty)
    grad .+= model.data.X * gpenal
end

function hess!(hess::AbstractVector{T},
               x::AbstractVector{T},
               model::DualGeneralizedLinearModel{T}) where T
    update!(model, θ)
    hess!(hess, x, model.data)
    hess!(hess, x, model.penalty)
end

function hessvec!(hv::AbstractVector{T},
                  θ::AbstractVector{T},
                  vec::AbstractVector{T},
                  model::DualGeneralizedLinearModel{T}) where T
    update!(model, θ)
    hessvec!(hv, θ, vec, model.data)
    hessvec!(hv, θ, vec, model.penalty)
end
