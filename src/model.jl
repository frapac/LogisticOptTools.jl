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
    # Penalty to apply (L0, L1 or L2)
    penalty::AbstractPenalty
    # Hash for current x to avoid computing twice some operations
    hash_x::UInt64
    # Shall we fit intercept as well?
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
    return LogisticRegressor(data, penalty, NULL_HASH, fit_intercept ,logger)
end

nfeatures(model::LogisticRegressor) = nfeatures(model.data)

function update!(model::LogisticRegressor, x)
    new_hash = hash(x)
    # Update prediction only if we test a new x
    if new_hash != model.hash_x
        nfeat = nfeatures(model)
        if model.fit_intercept
            predict!(model.data, view(x, 1:nfeat), x[end])
        else
            predict!(model.data, x)
        end
        model.hash_x = new_hash
    end
end

function loss(x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Sanity check
    update!(model, x)
    if model.fit_intercept
        w = @view x[1:nfeatures(model)]
        obj = loss(w, model.data) + model.penalty(w)
    else
        obj = loss(x, model.data) + model.penalty(x)
    end
    return obj
end

function gradient!(grad::AbstractVector{T}, x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Sanity check
    @assert length(x) == length(grad)
    # Reset gradient
    fill!(grad, 0.0)
    update!(model, x)
    if model.fit_intercept
        gradient!(grad, x, model.data, true)
        p = nfeatures(model)
        g = @view grad[1:p]
        w = @view x[1:p]
        gradient!(g, w, model.penalty)
    else
        gradient!(grad, x, model.data)
        gradient!(grad, x, model.penalty)
    end
end

function hessian!(hess::AbstractVector{T}, x::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Sanity check
    p = length(x)
    @assert length(hess) == p * (p + 1) / 2
    # Reset hessian
    fill!(hess, 0.0)
    update!(model, x)
    if model.fit_intercept
        hessian!(hess, x, model.data, true)
        hindex = Int[]
        p = nfeatures(model)
        # TODO: remove computation of hindex here.
        count = 1
        for j in 1:p
            for i in j:p
                push!(hindex, count)
                count += 1
            end
            count += 1
        end
        h = @view hess[hindex]
        w = @view x[1:p]
        hessian!(h, w, model.penalty)
    else
        hessian!(hess, x, model.data)
        hessian!(hess, x, model.penalty)
    end
end

function hessvec!(hessvec::AbstractVector{T}, x::AbstractVector{T}, vec::AbstractVector{T}, model::LogisticRegressor{T}) where T
    # Sanity check
    @assert length(vec) == length(hessvec)
    @assert length(vec) == length(x)
    # Reset hessian
    fill!(hessvec, 0.0)
    # Update yₚ
    update!(model, x)
    if model.fit_intercept
        hessvec!(hessvec, x, vec, model.data, true)
        p = nfeatures(model)
        hv = @view hessvec[1:p]
        v = @view vec[1:p]
        w = @view x[1:p]
        hessvec!(hv, w, v, model.penalty)
    else
        hessvec!(hessvec, x, vec, model.data)
        hessvec!(hessvec, x, vec, model.penalty)
    end
end

function diaghess!(diagh::AbstractVector{T}, x::AbstractVector{T},
                   model::LogisticRegressor{T}) where T
    fill!(diagh, 0.0)
    # Update yₚ
    update!(model, x)
    if model.fit_intercept
        diaghess!(diagh, x, model.data, true)
        p = nfeatures(model)
        dh = @view diagh[1:p]
        w = @view x[1:p]
        diaghess!(dh, w, model.penalty)
    else
        diaghess!(diagh, x, model.data)
        diaghess!(diagh, x, model.penalty)
    end
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
    hash_λ::UInt64
    logger::AbstractLogger
end
DualLogisticRegressor(data::AbstractDataset{T}, penalty::L2Penalty) where T = DualLogisticRegressor(data, penalty, NULL_HASH, NoLogger())

function DualLogisticRegressor(X::AbstractArray{T, 2}, y::AbstractVector{T};
                               penalty=L2Penalty(0.0),
                               logger=NoLogger()) where T
    if issparse(X)
        error("Sparse arrays not yet supported for dual logistic regression")
    end
    data = DualLogitData(X, y)
    return DualLogisticRegressor(data, penalty, NULL_HASH, logger)
end

function update!(model::DualLogisticRegressor, x::AbstractVector)
    new_hash = hash(x)
    if new_hash != model.hash_λ
        predict!(model.data, x)
        model.hash_λ = new_hash
    end
end

function loss(x::AbstractVector{T},
              model::DualLogisticRegressor{T}) where T
    update!(model, x)
    obj = loss(x, model.data) + model.penalty(model.data.x_pred)
    return obj
end

function gradient!(grad::AbstractVector{T}, x::AbstractVector{T},
                   model:: DualLogisticRegressor{T}) where T
    fill!(grad, 0.0)
    update!(model, x)
    gradient!(grad, x, model.data)
    # compute gradient of penalty ||X^\top θ||_2^2 in two steps
    gpenal = zeros(length(model.data.x_pred))
    gradient!(gpenal, model.data.x_pred, model.penalty)
    grad .+= model.data.X * gpenal
end

function hessian!(hess::AbstractVector{T},
                  x::AbstractVector{T},
                  model::DualLogisticRegressor{T}) where T
    fill!(hess, 0.0)
    update!(model, x)
    hessian!(hess, x, model.data)
    hessian!(hess, x, model.penalty)
end

function hessvec!(hv::AbstractVector{T},
                  x::AbstractVector{T},
                  vec::AbstractVector{T},
                  model::DualLogisticRegressor{T}) where T
    fill!(hv, 0.0)
    update!(model, x)
    hessvec!(hv, x, vec, model.data)
    hessvec!(hv, x, vec, model.penalty)
end

function generate_callbacks(model::DualLogisticRegressor)
    f = x -> loss(x, model)
    grad! = (g, x) -> gradient!(g, x, model)
    hess! = (h, x) -> hessian!(h, x, model)
    hessvec! = (h, x, v) -> hessian!(h, x, v, model)
    return f, grad!, hess!, hessvec!
end
