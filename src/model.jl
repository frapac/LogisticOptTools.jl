# Model object
abstract type AbstractModel end

##################################################
# PRIMAL
##################################################
const NULL_HASH = UInt64(0)

# define logistic problem
# TODO: add inheritance from MMI.Probabilistic
mutable struct LogisticRegressor{T <: Real} <: AbstractModel
    data::AbstractDataset{T}
    penalty::AbstractPenalty
    hash_x::UInt64
    fit_intercept::Bool
    # To log evolution during optimization process
    logger::AbstractLogger
end
function LogisticRegressor(data::AbstractDataset{T},
                           penalty::AbstractPenalty) where T
    return LogisticRegressor(data, penalty, NULL_HASH, false , NoLogger())
end
function LogisticRegressor(X::AbstractArray{T, 2}, y::AbstractVector{T};
                           penalty=L2Penalty(0.0),
                           logger=NoLogger(),
                           fit_intercept=false) where T
    Dataset = issparse(X) ? SparseLogitData : LogitData
    data = Dataset(X, y)
    return LogisticRegressor(data, penalty, NULL_HASH, fit_intercept , logger)
end

function update!(model::LogisticRegressor, x)
    new_hash = hash(x)
    # Update prediction only if we test a new x
    if new_hash != model.hash_x
        predict!(model.data, x)
        model.hash_x = new_hash
    end
end

function loss(x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    update!(model, x)
    obj = loss(x, model.data) + model.penalty(x)
    return obj
end

function gradient!(grad::AbstractVector{T}, x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Reset gradient
    fill!(grad, 0.0)
    update!(model, x)
    gradient!(grad, x, model.data)
    gradient!(grad, x, model.penalty)
end

function hessian!(hess::AbstractVector{T}, x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Reset hessian
    fill!(hess, 0.0)
    update!(model, x)
    hessian!(hess, x, model.data)
    hessian!(hess, x, model.penalty)
end

function hessvec!(hessvec::AbstractVector{T}, x::AbstractVector{T}, vec::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Reset hessian
    fill!(hessvec, 0.0)
    update!(model, x)
    hessvec!(hessvec, x, vec, model.data)
    hessvec!(hessvec, x, vec, model.penalty)
end

function generate_callbacks(model::LogisticRegressor)
    f = x -> loss(x, model)
    grad! = (g, x) -> gradient!(g, x, model)
    hess! = (h, x) -> hessian!(h, x, model)
    hessvec! = (h, x, v) -> hessian!(h, x, v, model)
    return f, grad!, hess!, hessvec!
end

##################################################
# DUAL
##################################################
# Define dual model
# Currently, it suports only L2 penalty
mutable struct DualLogisticRegressor{T <: Real} <: AbstractModel
    data::AbstractDataset{T}
    penalty::L2Penalty
    logger::AbstractLogger
end
DualLogisticRegressor(data::AbstractDataset{T}, penalty::L2Penalty) where T = DualLogisticRegressor(data, penalty, NoLogger())

update!(model::DualLogisticRegressor, x::AbstractVector) = _update_xpred!(model.data, x)

function loss(θ::AbstractVector{T},
              model::DualLogisticRegressor{T}) where T
    update!(model, θ)
    obj = loss(θ, model.data) + model.penalty(model.data.x_pred)
    return obj
end

function gradient!(grad::AbstractVector{T}, θ::AbstractVector{T},
                   model:: DualLogisticRegressor{T}) where T
    update!(model, θ)
    gradient!(grad, θ, model.data)
    # compute gradient of penalty ||X^\top θ||_2^2 in two steps
    gpenal = zeros(length(model.data.x_pred))
    gradient!(gpenal, model.data.x_pred, model.penalty)
    grad .+= model.data.X * gpenal
end

function hess!(hess::AbstractVector{T},
               x::AbstractVector{T},
               model::DualLogisticRegressor{T}) where T
    update!(model, θ)
    hess!(hess, x, model.data)
    hess!(hess, x, model.penalty)
end

function hessvec!(hv::AbstractVector{T},
                  θ::AbstractVector{T},
                  vec::AbstractVector{T},
                  model::DualLogisticRegressor{T}) where T
    update!(model, θ)
    hessvec!(hv, θ, vec, model.data)
    hessvec!(hv, θ, vec, model.penalty)
end
