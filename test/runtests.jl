
using Test, Random
using Statistics, SparseArrays, LinearAlgebra
using Optim
using LogisticOptTools

const LOT = LogisticOptTools
const SVM_DATASET = joinpath(@__DIR__, "diabetes.txt")
Random.seed!(2713)


function fake_dataset(n_samples=100, n_features=10)
    X = randn(n_samples, n_features)
    w = randn(n_features)
    y = sign.(X * w)
    X .+= 0.8 * randn(n_samples, n_features) # add noise
    X .+= 100.0 # this makes it correlated by adding a constant term
    X = hcat(X, ones(n_samples, 1))
    return X, y
end

function wrap_optim(dat::LOT.AbstractDataset, penalty::LOT.AbstractPenalty)
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

        y1 = [-2, 2, 2, -2]
        LOT.format_label!(y1)
        @test isequal(y1, [-1, 1, 1, -1])
        y2 = [-1, 3, 3]
        LOT.format_label!(y2)
        @test isequal(y2, [-1, 1, 1])
        y3 = [-1, 0, 0]
        LOT.format_label!(y3)
        @test isequal(y3, [-1, 1, 1])
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

@testset "Primal model" begin
    # Test dataset
    svm_data = LOT.parse_libsvm(SVM_DATASET, Float64)
    X = LOT.to_dense(svm_data)
    y = svm_data.labels

    @testset "Constructor" begin
        d1 = @inferred LOT.LogitData(X, y)
        @test sort(unique(d1.y)) == [-1.0, 1.0]
        d2 = @inferred LOT.LogitData(SVM_DATASET, scale_data=false)
        @test sort(unique(d2.y)) == [-1.0, 1.0]
        # Import should be the same when data are not scaled.
        @test hash(d1.X) == hash(d2.X)
        d3 = @inferred LOT.LogitData(SVM_DATASET, scale_data=true)
        @test sort(unique(d3.y)) == [-1.0, 1.0]
    end

    data = LOT.LogitData(X, y)
    # Input dimension
    n = LOT.dim(data)
    @test isa(n, Int)
    # Initial point
    x0 = zeros(n)
    # vector for hess-vec product
    vec = zeros(n)
    # Allocate vectors for gradient, Hessian and diag-Hessian
    g = zeros(n)
    hess = zeros(div(n * (n+1), 2))
    diagh = zeros(n)

    @testset "Primal logistic regression" begin
        @test LOT.loss(x0, data) ≈ -LOT.log1pexp(0.0)
        LOT.gradient!(g, x0, data)
        LOT.hess!(hess, x0, data)
        LOT.hessvec!(g, x0, vec, data)
        LOT.diaghess!(diagh, x0, data)
    end

    @testset "Penalty" begin
        # Penalty parameter
        fill!(g, 0.0)
        fill!(hess, 0.0)
        fill!(diagh, 0.0)
        λ = 1.0
        for penalty in [LOT.L2Penalty(λ),
                        LOT.L1Penalty(λ),
                        LOT.LinearizedL1Penalty(λ),
                        LOT.L0Penalty(1.0)]
            @test penalty(x0) == 0.0
            LOT.gradient!(g, x0, penalty)
            @test all(g .== 0.0)
            LOT.hess!(hess, x0, penalty)
            LOT.hessvec!(g, x0, vec, penalty)
            LOT.diaghess!(diagh, x0, penalty)
            @test all(diagh .== 2.0)
        end
    end

    @testset "Fitting logistic regression (primal)" begin
        # Load dataset
        svm_data = LOT.parse_libsvm(SVM_DATASET, Float64)
        X1 = LOT.to_dense(svm_data)
        X2 = LOT.to_sparse(svm_data)
        y = svm_data.labels
        # Solution when data are scaled
        solution = 0.6084979240046
        for (X, glm) in zip([X1, X2], [LOT.LogitData, LOT.SparseLogitData])
            data = glm(X, y)
            @test LOT.length(data) == length(y)
            @test LOT.dim(data) == size(X, 2)
            # Build callbacks
            f, g! = wrap_optim(data, LOT.L2Penalty(0.0))
            algo = BFGS()
            options = Optim.Options(iterations=250, g_tol=1e-5)
            res_joptim = Optim.optimize(f, g!, zeros(LOT.dim(data)), algo, options)
            @test res_joptim.minimum ≈ solution
            @test Optim.converged(res_joptim)
        end
    end
end

@testset "Dual model" begin
    svm_data = LOT.parse_libsvm(SVM_DATASET, Float64)
    X = LOT.to_dense(svm_data)
    y = svm_data.labels
    data = LOT.DualLogitData(X, y)
    n = length(data)
    penalty = LOT.L2Penalty(0.0)
    @testset "Dual Logit" begin
        λ = zeros(n)
        @test LOT.loss(λ, data) == 0.0
        λ = -data.y
        @test LOT.loss(λ, data) == 0.0
        g = zeros(n)
        LOT.gradient!(g, λ, data)
    end
    @testset "Dual GLM" begin
        glm = LOT.DualGeneralizedLinearModel(data, penalty)
        λ = zeros(n)
        @test LOT.loss(λ, glm) == 0.0
        g = zeros(n)
        LOT.gradient!(g, λ, glm)
    end
    @testset "Fitting (dual)" begin
        glm = LOT.DualGeneralizedLinearModel(data, penalty)
        f = x -> LOT.loss(x, glm)
        gradient! = (g, x) -> LOT.gradient!(g, x, glm)
        algo = BFGS()
        options = Optim.Options(iterations=250, g_tol=1e-5)
        lower = LOT.lowerbound(data)
        upper = LOT.upperbound(data)
        x0 = 0.5 * (lower .+ upper)
        res_joptim = Optim.optimize(f, gradient!, lower, upper,
                                    x0, Fminbox(algo), options)
        @test Optim.converged(res_joptim)
        println(res_joptim.minimum)
    end
end
