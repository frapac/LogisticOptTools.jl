
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
    n = length(d.logreg)
    return loss(x[1:n], d.logreg)
end

function MOI.eval_objective_gradient(d::LogisticEvaluator, grad_f, x)
    n = length(d.logreg)
    gradient!(view(grad_f, 1:n), x[1:n], d.logreg)
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

function MOI.eval_constraint(d::LogisticEvaluator, cons, x)
end

function MOI.eval_constraint_jacobian(d::LogisticEvaluator, grad_f, x)
end

function MOI.eval_hessian_lagrangian(d::LogisticEvaluator, H, x, σ, μ)
    @assert d.enable_hessian
    n = length(d.logreg)
    hessian!(H, x[1:n], d.logreg)
end

function MOI.eval_hessian_lagrangian_product(d::LogisticEvaluator, h, x, v, σ, μ)
    @assert d.enable_hessian_vector_product
    n = length(d.logreg)
    hessvec!(view(h, 1:n), x[1:n], v[1:n], d.logreg)
end

# TODO: pass to a NormOneCone bridge
function _linearize_penalty!(model::MOI.ModelLike,
                             parameters::Vector{MOI.VariableIndex},
                             logreg::LogisticRegressor)
    nvariables = length(logreg)
    n_cons = nvariables
    linear_penal = MOI.add_variables(model, nvariables)
    for i in 1:n_cons
        # 0 <= z_i + w_i,
        c_lb = MOI.ScalarAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}.([1.0, 1.0], [parameters[i], linear_penal[i]]), 0.0)
        c = MOI.add_constraint(model, c_lb, MOI.GreaterThan(0.0))
        # -z_i + w_i <= 0,      for i = 0, ..., n-1
        c_ub = MOI.ScalarAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}.([1.0, -1.0], [parameters[i], linear_penal[i]]), 0.0)
        c = MOI.add_constraint(model, c_ub, MOI.LessThan(0.0))
    end

    # TODO: find a way to handle linear objective in NLPevaluator
    ocoefs = fill(logreg.penalty.constant, nvariables)
    oindex = linear_penal
    objf = MOI.ScalarAffineFunction{Float64}(MOI.ScalarAffineTerm{Float64}.(ocoefs, oindex), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
end

function nlp_model(model::MOI.ModelLike, evaluator, logreg::LogisticRegressor)
    MOI.empty!(model)
    @assert MOI.supports(model, MOI.NLPBlock())

    nvariables = length(logreg)
    # add variables
    start = zeros(nvariables)
    w = MOI.add_variables(model, nvariables)
    for i in 1:nvariables
        MOI.set(model, MOI.VariablePrimalStart(), w[i], start[i])
    end
    # set NLP structure
    block_data = MOI.NLPBlockData([], evaluator, true)
    MOI.set(model, MOI.NLPBlock(), block_data)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    if isa(logreg.penalty, LinearizedL1Penalty)
        _linearize_penalty!(model, w, logreg)
    end
    return w
end

function conic_model(model::MOI.ModelLike, logreg::LogisticRegressor)
    MOI.empty!(model)
    @assert MOI.supports_constraint(model, MOI.VectorAffineFunction{Float64}, MOI.ExponentialCone)

    X = logreg.data.X
    y = logreg.data.y

    n = ndata(logreg.data)
    nvariables = length(logreg)
    # add variables
    start = zeros(nvariables)
    w = MOI.add_variables(model, nvariables)
    t = MOI.add_variables(model, nvariables)

    for j in 1:nvariables
        MOI.set(model, MOI.VariablePrimalStart(), v[j], start[j])
    end

    for i in 1:n
        # temporary variables
        z = MOI.add_variables(model, 2)
        # z1, z2 >= 0
        MOI.add_constraint.(model, MOI.SingleVariable.(z), MOI.GreaterThan(0.0))
        # z1 + z2 <= 1
        c_sum = MOI.ScalarAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}.([1.0, 1.0], z), 0.0)
        MOI.add_constraint(model, c_sum, MOI.LessThan(1.0))

        # (z1, 1, u - t ) ∈  K_exp
        terms = MOI.VectorAffineTerm{Float64}[]
        vat  = MOI.VectorAffineTerm{Float64}(3, MOI.ScalarAffineTerm{Float64}(1.0, z[1]))
        push!(terms, vat)
        vat  = MOI.VectorAffineTerm{Float64}(1, MOI.ScalarAffineTerm{Float64}(-1.0, t[i]))
        push!(terms, vat)
        for j in eachindex(X[i, :])
            x_ij = X[i, j]
            if x_ij != 0.0
                x_ij *= y[i]
                vat  = MOI.VectorAffineTerm{Float64}(1, MOI.ScalarAffineTerm{Float64}(x_ij, w[j]))
                push!(terms, vat)
            end
        end

        constants = [0.0, 1.0, 1.0]
        vaf = MOI.VectorAffineFunction{Float64}(vat, constants)
        vc = MOI.add_constraint(model, vaf, MOI.ExponentialCone())

        # (z2, 1, - t ) ∈  K_exp
        terms = MOI.VectorAffineTerm{Float64}[]
        vat  = MOI.VectorAffineTerm{Float64}(3, MOI.ScalarAffineTerm{Float64}(1.0, z[2]))
        push!(terms, vat)
        vat  = MOI.VectorAffineTerm{Float64}(1, MOI.ScalarAffineTerm{Float64}(-1.0, t[i]))
        push!(terms, vat)
        constants = [0.0, 1.0, 1.0]
        vaf = MOI.VectorAffineFunction{Float64}(vat, constants)
        vc = MOI.add_constraint(model, vaf, MOI.ExponentialCone())
    end

    # add penalty
    #
    # add objective
    ocoefs = fill(1.0 / n, nvariables)
    objf = MOI.ScalarAffineFunction{Float64}(MOI.ScalarAffineTerm{Float64}.(ocoefs, t), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
end

