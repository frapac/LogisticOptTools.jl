# Benchmark Optim.jl on logistic regression problems

using Optim, LineSearches
using LogisticOptTools
using Plots

const LOT = LogisticOptTools

# Choose dataset to load
DATASET = "covtype.libsvm.binary.bz2"
# Choose penalty with given coefficient λ
PENALTY = LOT.L2Penalty(1e-2)

if !isfile(DATASET)
    println("Download dataset from $LIBSVM_URL")
    download("$LIBSVM_URL/$DATASET", DATASET)
end

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
                      f♯,
                      label::String,
                      p::Plots.Plot)
    exectime = [t.metadata["time"] for t in results.trace]
    exectime .-= exectime[1]
    values = [t.value for t in results.trace]
    values .-= f♯
    plot!(p, exectime, log10.(values), marker=true, label=label, markersize=1.)
end

"Benchmark algorithms from Optim.jl"
function benchmark(algos, options, logreg)
    p = LOT.nfeatures(logreg)
    x = zeros(p)
    g = zeros(p)
    f, g!, _, _ = LOT.generate_callbacks(logreg)
    # Activate precompilation
    f(x)
    g!(g, x)

    results = []
    for algo in algos
        res = Optim.optimize(f, g!, x, algo, options)
        push!(results, res)
    end
    return results
end

bfgs1 = BFGS(alphaguess=InitialHagerZhang(α0=1.0),
             linesearch=LineSearches.HagerZhang())
bfgs2 = BFGS(alphaguess=InitialStatic(alpha=1.0),
             linesearch=LineSearches.MoreThuente())
cg = ConjugateGradient()

options = Optim.Options(iterations=100, store_trace=true, show_trace=true,
                        g_tol=1e-5, allow_f_increases=true)

# Load data
dataset = LOT.LogitData(DATASET, scale_data=true)
# Build logistic regression
logreg = LOT.LogisticRegressor(dataset.X, dataset.y, penalty=PENALTY, fit_intercept=true)
results = benchmark([bfgs1, bfgs2, cg], options, logreg)

# Plot evolution of gradient
pg = plot(markershape=:rect)
display_grad.(results, ["Optim BFGS HZ", "Optim BFGS MT", "Optim CG"], Ref(pg))
xlabel!(pg, "Time (s)")
ylabel!(pg, "Gradient (log scale)")

# Plot evolution of function
pf = plot(markershape=:rect)
# Get optimum
f♯ = minimum([res.minimum for res in results])
display_func.(results, f♯, ["Optim BFGS HZ", "Optim BFGS MT", "Optim CG"], Ref(pf))
xlabel!(pf, "Time (s)")
ylabel!(pf, "log(f - fopt) ")
