push!(LOAD_PATH, "..")

using LinearAlgebra
using JuMP, Optim
using LogisticOptTools
using Xpress, GLPK

const LOT = LogisticOptTools

SOLVER = optimizer_with_attributes(Xpress.Optimizer, "PRESOLVE" => 0,
                                   "MIPPRESOLVE" => 0)
SOLVER = () -> Xpress.Optimizer(
        PRESOLVE = 0,
        CUTSTRATEGY = 0,
        HEURSTRATEGY = 0,
        SYMMETRY = 0,
        OUTPUTLOG = 3
    )

sparsity(s::Vector; tol=0.5) = findall(x -> x > tol, s)

function oracle(X, y, γ, s)
    pattern = sparsity(s)
    println(pattern)
    Xs = X[:, pattern]
    glm = LOT.DualLogisticRegressor(Xs, y, penalty=LOT.L2Penalty(γ))
    f, gradient!, _, _ = LOT.generate_callbacks(glm)
    algo = LBFGS()
    options = Optim.Options(iterations=250, g_tol=1e-5, show_trace=false)
    lower = LOT.lowerbound(glm.data)
    upper = LOT.upperbound(glm.data)
    x0 = 0.5 * (lower .+ upper)
    res = Optim.optimize(f, gradient!, lower, upper,
                         x0, Fminbox(algo), options)
    c = res.minimum
    α♯ = res.minimizer
    ∇c = zeros(length(s))

    @inbounds for i in eachindex(s)
        atmp = dot(X[:, i], α♯)
        ∇c[i] = -1/ γ * atmp^2
    end
    return -c, ∇c
end

function benders(X, y, s0, γ, k; solver=SOLVER)
    n, p = size(X)
    c0, ∇c0 = oracle(X, y, γ, s0)

    benders_model = Model(solver)

    # Variables s and t
    @variable(benders_model, 0 <= s[i=1:p] <= 1, Bin, start=s0[i])
    @variable(benders_model, 0 <= t , start=c0)
    # Global constraints
    @constraint(benders_model, t >= c0 + dot(∇c0, s - s0))
    @constraint(benders_model, sum(s) <= k)
    # Objective
    @objective(benders_model, Min, t)

    function lazy_constraint_callback(cb_data)
        s_current = [callback_value(cb_data, s[j]) for j in eachindex(s)]
        println(s_current)
        ck, ∇ck = oracle(X, y, γ, s_current)
        new_optimality_cons = @build_constraint(ck + dot(∇ck, s - s_current) <= t)
        MOI.submit(benders_model, MOI.LazyConstraint(cb_data), new_optimality_cons)
    end

    #= MOI.set(benders_model, MOI.LazyConstraintCallback(), lazy_constraint_callback) =#
    optimize!(benders_model)
    return benders_model
end

γ = 1e-2
DATASET = "diabetes"
#= DATASET = "covtype.libsvm.binary.bz2" =#
#= DATASET = "colon-cancer.bz2" =#
dataset = LOT.LogitData(DATASET, scale_data=true)
X, y = dataset.X, dataset.y

X = X[:, 1:2]
n, p = size(X)
dγ = 1.0 / (4.0 * γ * n)

k = 1
s0 = zeros(p)
s0[1:k] .= 1
m = benders(X, y, s0, dγ, k)
