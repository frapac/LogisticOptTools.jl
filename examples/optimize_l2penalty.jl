# Demonstrate how to optimize L2 penalty parameter with LogisticOptTools
# We follow the method introduced in:
# Optimizing for Generalization in Machine Learning with Cross-Validation Gradients
# S Barratt, R Sharma
# arXiv preprint arXiv:1805.07072

using Optim
using LogisticOptTools
using Plots

const LOT = LogisticOptTools

# Select dataset where to optimize l2 penalty parameter
SVM_DATASET = "diabetes"
dataset = LOT.LogitData(SVM_DATASET)

# Instantiate optimizer. We have an outer and an inner optimizers there.
options = Optim.Options(iterations=250, g_tol=1e-5, show_trace=false)
optimizer = LOT.L2PenaltyOptimizer(3, false, false, BFGS(), LBFGS(), options)

# Set initial parameter
γ0 = 1.0
# Optimize (be careful, problem is not necessary convex)!
l♯, γ♯ = LOT.fit!(optimizer, dataset.X, dataset.y, γ0)

# Show effectiveness of optimal parameter found
costs = []
# Range of γ to explore
t = range(-10, step=.1, stop=10.)
for i in  t
    γ = 10^i
    c, dc = LOT.crossvalid_loss(dataset.X, dataset.y, γ, penopt)
    push!(costs, c)
end

plot(t, costs)
vline!([log10(γ♯)])
xlabel!("ℓ2 parameter (log scale)")
ylabel!("Costs")
