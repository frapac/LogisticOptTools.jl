
using Test, Random
using Statistics, SparseArrays, LinearAlgebra
using Optim
using LogisticOptTools

const LOT = LogisticOptTools

Random.seed!(2713)
const SVM_DATASET = joinpath(@__DIR__, "diabetes.txt")

function fake_dataset(n_samples=100, n_features=10)
    n_samples, n_features = 10^4, 10^3
    X = randn(n_samples, n_features)
    w = randn(n_features)
    y = sign.(X * w)
    X .+= 0.8 * randn(n_samples, n_features) # add noise
    X .+= 100.0 # this makes it correlated by adding a constant term
    X = hcat(X, ones(n_samples, 1))
    return X, y
end

function wrap_optim(dat::LOT.LogitData, penalty::LOT.AbstractPenalty)
    prob = LOT.GeneralizedLinearModel(dat, penalty, LOT.Tracer{Float64}())
    eval_f = x -> LOT.loss(x, prob)
    function eval_g(g, x)
        LOT.gradient!(g, x, prob)
        push!(prob.logger.grad, norm(g, Inf))
        push!(prob.logger.time, time())
        return nothing
    end
    return (eval_f, eval_g)
end

@testset "Utils functions" begin
    # Test expit
    @testset "Evaluation functions" begin
        for t in [-1.0, 0.0, 1.0]
            @test LOT.expit(t) == 1.0 / (1.0 + exp(-t))
            @test LOT.log1pexp(t) == -log(1.0 + exp(-t))
        end
    end
    @testset "Preprocessing" begin
        n, m = 100, 10
        dataset = randn(n, m)
        LOT.scale!(LOT.NormalScaler(), dataset)
        @test mean(dataset, dims=1) ≈ zeros(1, m) atol=1e-8
        @test std(dataset, dims=1) ≈ ones(1, m)
    end
end

@testset "LIBSVM parser" begin
    # Load dataset from LIBSVM URL
    if !isfile(SVM_DATASET)
        download("$(LOT.LIBSVM_URL)/diabetes", SVM_DATASET)
    end
    # Parse diabetes either with Float32 or Float64
    for type_ in [Float32, Float64]
        dat = LOT.parse_libsvm(SVM_DATASET, type_)
        @test isa(dat, LOT.LibSVMData{Int, type_})
        X = LOT.to_dense(dat)
        @test isa(X, Array{type_, 2})
        Xs = LOT.to_sparse(dat)
        @test issparse(Xs)
    end
end

@testset "Penalty" begin
    n = 10
    λ = 1.0
    x0 = zeros(n)
    x1 = ones(n)
    g = zeros(n)
    vec = zeros(n)
    hess = zeros(div(n * (n-1), 2))
    for penalty in [LOT.L2Penalty(λ), LOT.L1Penalty(λ)]
        @test penalty(x0) == 0.0
        @test penalty(x1) == penalty.constant * sum(n)
        LOT.gradient!(g, x0, penalty)
        LOT.hess!(hess, x0, penalty)
        LOT.hessvec!(g, x0, vec, penalty)
    end
end

@testset "Fitting" begin
    # Load dataset
    svm_data = LOT.parse_libsvm(SVM_DATASET, Float64)
    X = LOT.to_dense(svm_data)
    y = svm_data.labels
    for glm in [LOT.LogitData]
        data = glm(X, y)
        @test LOT.length(data) == length(y)
        @test LOT.dim(data) == size(X, 2)
        # Build callbacks
        f, g! = wrap_optim(data, LOT.L2Penalty(0.0))
        algo = BFGS()
        options = Optim.Options(iterations=250, g_tol=1e-5)
        res_joptim = Optim.optimize(f, g!, zeros(LOT.dim(data)), algo, options)
        Optim.converged(res_joptim)
    end
end
