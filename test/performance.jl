using LinearAlgebra, SparseArrays
using LogisticOptTools
using BenchmarkTools

const LOT = LogisticOptTools

function dense_callbacks(;n=2_000, p=2_000, intercept=false)
    A = randn(n, p)
    b = sign.(randn(n))
    x = randn(p)

    dat = LOT.LogitData(A, b)
    logit = LOT.LogisticRegressor(dat, LOT.L2Penalty(1.0))
    logit.fit_intercept = intercept

    println("Benchmark f")
    @time LOT.loss(x, dat)
    @time LOT.loss(x, logit)

    println("Benchmark ∇f")
    g = zeros(p)
    @time LOT.gradient!(g, x, dat)
    @time LOT.gradient!(g, x, logit)

    println("Benchmark ∇^2f v")
    hvec = zeros(p)
    vec = zeros(p)
    @time LOT.hessvec!(hvec, x, vec, dat)
    @time LOT.hessvec!(hvec, x, vec, logit)

    println("Benchmark diag(∇^2f)")
    @time LOT.diaghess!(hvec, x, dat)

    println("Benchmark ∇^2f")
    p = 54
    A = randn(p, p)
    b = sign.(randn(p))
    dat = LOT.LogitData(A, b)
    logit = LOT.LogisticRegressor(dat, LOT.L2Penalty(1.0))
    hess = zeros(div(p * (p+1), 2))
    x = randn(p)
    @time LOT.hessian!(hess, x, dat)
    @time LOT.hessian!(hess, x, logit)
end

function sparse_callbacks(;n=2_000, p=2_000)
    # sparse matrix
    A = sprandn(n, p, .25)
    # dense vector
    b = sign.(randn(n))

    dat = LOT.SparseLogitData(A, b)
    logit = LOT.LogisticRegressor(dat, LOT.L2Penalty(1.0))

    n = LOT.nfeatures(dat)
    x = randn(p)
    println("Benchmark f")
    LOT.loss(x, dat)
    @time LOT.loss(x, dat)

    println("Benchmark ∇f")
    g = zeros(p)
    LOT.gradient!(g, x, dat)
    @time LOT.gradient!(g, x, dat)

    println("Benchmark ∇^2f v")
    hvec = zeros(p)
    vec = zeros(p)
    LOT.hessvec!(hvec, x, vec, dat)
    @time LOT.hessvec!(hvec, x, vec, dat)

    #= println("Benchmark ∇^2f") =#
    #= p = 54 =#
    #= A = randn(T, p, p) =#
    #= b = sign.(randn(T, p)) =#
    #= dat = LOT.LogitData(A, b) =#
    #= logit = LOT.GeneralizedLinearModel(dat, LOT.L2Penalty(1.0)) =#
    #= hess = zeros(T, div(p * (p+1), 2)) =#
    #= x = randn(T, p) =#
    #= LOT.hess!(hess, x, dat) =#
    #= @time LOT.hess!(hess, x, dat) =#
    #= @time LOT.hess!(hess, x, logit) =#
end
