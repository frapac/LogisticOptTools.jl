# Benchmark Optim.jl on logistic regression problems

using Optim, LineSearches
using LogisticOptTools
using Plots
const LOT = LogisticOptTools

# Choose dataset to load
DATASET = "colon-cancer.bz2"
# Specify whether to run LBFGSB
RUN_LBFGSB = true
# Choose penalty with given coefficient λ
PENALTY = LOT.L2Penalty(1e-5)

# First step: load data
if !isfile(DATASET)
    println("Download dataset from $LIBSVM_URL")
    download("$LIBSVM_URL/$DATASET", DATASET)
end
# Load dataset as dense matrices
include("dataset.jl")
#= X, y = real_dataset(DATASET) =#
X, y = fake_dataset(10000, 2000, correlation=10.0, intercept=false)


# Some plots utilities
"Display evolution of gradients along iteration."
function display_grad(results::Optim.OptimizationResults,
                      label::String,
                      p::Plots.Plot)
    exectime = [t.metadata["time"] for t in results.trace]
    exectime .-= exectime[1]
    grad = [t.g_norm for t in results.trace]
    plot!(p, exectime, log10.(grad), marker=true, label=label, markersize=1.)
end

"Display evolution of functions along iteration."
function display_func(results::Optim.OptimizationResults,
                      label::String,
                      p::Plots.Plot)
    exectime = [t.metadata["time"] for t in results.trace]
    exectime .-= exectime[1]
    values = [t.value for t in results.trace]
    values .-= values[end]
    plot!(p, exectime, log10.(values), marker=true, label=label, markersize=1.)
end

"Closure for evaluation callbacks"
function get_callback(dat::LOT.LogitData, penalty::LOT.AbstractPenalty)
    # Define logistic problem in closure for use in Optim.jl
    prob = LOT.GeneralizedLinearModel(dat, penalty)
    eval_f = x -> LOT.loss(x, prob)
    function eval_g(g, x)
        LOT.gradient!(g, x, prob)
        return nothing
    end
    return (eval_f, eval_g)
end
get_callback(X, y; penalty=LOT.L2Penalty(0.0)) = get_callback(LOT.LogitData(X, y), penalty)

"Benchmark algorithms from Optim.jl"
function benchmark(algos, options, X, y, penalty)
    n = size(X, 2)
    x = zeros(n)
    g = zeros(n)
    # Build evaluation callbacks
    f, g! = get_callback(X, y, penalty=penalty)
    # Activate compilation
    f(x)
    g!(g, x)

    results = []
    for algo in algos
        res = Optim.optimize(f, g!, zeros(size(X, 2)), algo, options)
        push!(results, res)
    end
    return results
end

bfgs1 = BFGS(alphaguess=InitialHagerZhang(α0=1.0),
             linesearch=LineSearches.HagerZhang())
bfgs2 = BFGS(alphaguess=InitialStatic(alpha=1.0),
             linesearch=LineSearches.MoreThuente())
cg = ConjugateGradient()

options = Optim.Options(iterations=250, store_trace=true, show_trace=false,
                        g_tol=1e-5, allow_f_increases=true)

results = benchmark([bfgs1, bfgs2, cg], options, X, y, PENALTY)

# Plot evolution of gradient
pg = plot(markershape=:rect)
display_grad.(results, ["Optim BFGS HZ", "Optim BFGS MT", "Optim CG"], Ref(pg))
xlabel!(pg, "Time (s)")
ylabel!(pg, "Gradient (log scale)")

# Plot evolution of function
pf = plot(markershape=:rect)
display_func.(results, ["Optim BFGS HZ", "Optim BFGS MT", "Optim CG"], Ref(pf))
xlabel!(pf, "Time (s)")
ylabel!(pf, "log(f - fopt) ")

if RUN_LBFGSB
    include("lbfgsb.jl")
    trace = bfgs_fit(LOT.LogitData(X, y), PENALTY, trace=true)
    t0 = trace.time[1]
    trace.time .-= t0
    fopt = trace.obj[end]

    plot!(pf, trace.time, log10.(trace.obj .- fopt), label="LBFGSB.jl",
          marker=true, markersize=1.0)
    plot!(pg, trace.time, log10.(trace.grad), label="LBFGSB.jl",
          marker=true, markersize=1.0)
end
