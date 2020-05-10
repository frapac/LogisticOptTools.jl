
using MathOptInterface
const MOI = MathOptInterface

struct LogisticEvaluator <: MOI.AbstractNLPEvaluator
    logreg::AbstractRegression
    enable_hessian::Bool
    enable_hessian_vector_product::Bool
end

function MOI.features_available(d::LogisticEvaluator)
    if d.enable_hessian
        return [:Grad, :Jac, :Hess]
    elseif d.enable_hessian_vector_product
        return [:Grad, :Jac, :HessVec]
    else
        return [:Grad, :Jac]
    end
end

function MOI.initialize(d::LogisticEvaluator,
                        requested_features::Vector{Symbol})
end

function MOI.eval_objective(d::LogisticEvaluator, x)
    return loss(x, d.logreg)
end

function MOI.eval_objective_gradient(d::LogisticEvaluator, grad_f, x)
    gradient!(grad_f, x, d.logreg)
end

# upper triangle only
function MOI.hessian_lagrangian_structure(d::LogisticEvaluator)
    @assert d.enable_hessian
    p = length(d.logreg)
    structure = Tuple{Int64,Int64}[]
    for i in 1:p
        for j in i:p
            push!(structure, (i, j))
        end
    end
    return structure
end

function MOI.eval_hessian_lagrangian(d::LogisticEvaluator, H, x, σ, μ)
    @assert d.enable_hessian
    hessian!(H, x, d.logreg)
end

function MOI.eval_hessian_lagrangian_product(d::LogisticEvaluator, h, x, v, σ, μ)
    @assert d.enable_hessian_vector_product
    hessvec!(h, x, v, d.logreg)
end

function nlp_model(model::MOI.ModelLike, evaluator, logreg::LogisticRegressor)
    MOI.empty!(model)

    nvariables = nfeatures(logreg)
    if logreg.fit_intercept
        nvariables += 1
    end
    # add variables
    start = zeros(nvariables)
    v = MOI.add_variables(model, nvariables)
    for i in 1:nvariables
        MOI.set(model, MOI.VariablePrimalStart(), v[i], start[i])
    end
    # set NLP structure
    block_data = MOI.NLPBlockData([], evaluator, true)
    MOI.set(model, MOI.NLPBlock(), block_data)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
end

