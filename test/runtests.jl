
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

    @testset "Constructor dense dataset" begin
        d1 = @inferred LOT.LogitData(X, y)
        @test sort(unique(d1.y)) == [-1.0, 1.0]
        d2 = @inferred LOT.LogitData(SVM_DATASET, scale_data=false)
        @test sort(unique(d2.y)) == [-1.0, 1.0]
        # Import should be the same when data are not scaled.
        @test hash(d1.X) == hash(d2.X)
        d3 = @inferred LOT.LogitData(SVM_DATASET, scale_data=true)
        @test sort(unique(d3.y)) == [-1.0, 1.0]
    end

    @testset "Constructor sparse dataset" begin
        d2 = @inferred LOT.SparseLogitData(SVM_DATASET, scale_data=false)
        @test sort(unique(d2.y)) == [-1.0, 1.0]
        d2 = @inferred LOT.SparseLogitData(SVM_DATASET, scale_data=true)
        @test sort(unique(d2.y)) == [-1.0, 1.0]
    end

    @testset "Primal logistic regression" begin
        for Dataset in [LOT.LogitData, LOT.SparseLogitData]
            data = Dataset(SVM_DATASET)
            # Input dimension
            p = LOT.nfeatures(data)
            @test isa(p, Int)
            @test p == size(data.X, 2)
            # Initial point
            x0 = zeros(p)
            # vector for hess-vec product
            vec = zeros(p)
            # Allocate vectors for gradient, Hessian and diag-Hessian
            g = zeros(p)
            hess = zeros(div(p * (p+1), 2))
            diagh = zeros(p)
            @test LOT.loss(x0, data) ≈ -LOT.log1pexp(0.0)
            LOT.gradient!(g, x0, data)
            LOT.hessvec!(g, x0, vec, data)
            LOT.diaghess!(diagh, x0, data)
            if isa(data, LOT.LogitData)
                LOT.hessian!(hess, x0, data)
            end
        end
    end

    @testset "Intercept" begin
        data = LOT.LogitData(SVM_DATASET)
        # +1 accounts for intercept
        p = LOT.nfeatures(data) + 1
        # Initial point
        x0 = zeros(p)
        # vector for hess-vec product
        vec = zeros(p)
        # Allocate vectors for gradient, Hessian and diag-Hessian
        g = zeros(p)
        hess = zeros(div(p * (p+1), 2))
        diagh = zeros(p)
        fit_intercept = true
        LOT.gradient!(g, x0, data, fit_intercept)
        LOT.hessvec!(g, x0, vec, data, fit_intercept)
        LOT.diaghess!(diagh, x0, data, fit_intercept)
        LOT.hessian!(hess, x0, data, fit_intercept)
    end

    @testset "Penalty" begin
        # Penalty parameter
        p = size(X, 2)
        x0 = zeros(p)
        g = zeros(p)
        hess = zeros(div(p * (p+1), 2))
        diagh = zeros(p)
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
            LOT.hessian!(hess, x0, penalty)
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
        options = Optim.Options(iterations=250, g_tol=1e-5)
        algo = BFGS()
        # No intercept
        for (X, glm) in zip([X1, X2], [LOT.LogitData, LOT.SparseLogitData])
            data = glm(X, y)
            @test LOT.ndata(data) == length(y)
            p = LOT.nfeatures(data)
            @test p == size(X, 2)
            # Build callbacks
            logreg = LOT.LogisticRegressor(X, y)
            f, g!, _, _ = LOT.generate_callbacks(logreg)
            x0 = zeros(p)
            res_joptim = Optim.optimize(f, g!, x0, algo, options)
            @test res_joptim.minimum ≈ solution
            @test Optim.converged(res_joptim)
            # Test dedicated shortcut function
            res_joptim = Optim.optimize(logreg, x0, algo, options)
            @test res_joptim.minimum ≈ solution
            @test Optim.converged(res_joptim)
        end

        # Intercept
        solution_intercept = 0.47099308415704666
        for (X, glm) in zip([X1, X2], [LOT.LogitData, LOT.SparseLogitData])
            data = glm(X, y)
            p = LOT.nfeatures(data)
            # Build callbacks
            logreg = LOT.LogisticRegressor(X, y, fit_intercept=true)
            f, g!, _, _ = LOT.generate_callbacks(logreg)
            x0 = zeros(p + 1)
            res_joptim = Optim.optimize(f, g!, x0, algo, options)
            @test res_joptim.minimum ≈ solution_intercept
            @test Optim.converged(res_joptim)
            # Test dedicated shortcut function
            res_joptim = Optim.optimize(logreg, x0, algo, options)
            @test res_joptim.minimum ≈ solution_intercept
            @test Optim.converged(res_joptim)
        end

        @testset "LogisticOptimizer utilities" begin
            # Test fit! function
            for (bias, sol) in zip([false, true], [solution, solution_intercept])
                logopt = LOT.LogisticOptimizer(algo=algo, options=options,
                                               fit_intercept=bias)
                p = size(X, 2) + bias
                res = LOT.fit!(logopt, X1, y, zeros(p))
                @test Optim.converged(res)
                @test res.minimum ≈ sol
            end
        end
    end
end

@testset "Test optimization of L2 penalty" begin
    dataset = LOT.LogitData(SVM_DATASET, scale_data=true)
    kfold = 5
    shuffle = false
    inner_algo = LBFGS()
    outer_algo = BFGS()
    options = Optim.Options()
    penopt = LOT.L2PenaltyOptimizer(kfold, shuffle, false,
                                    outer_algo, inner_algo, options)
    l♯, γ♯ = LOT.fit!(penopt, dataset.X, dataset.y, 1.0)
    @test l♯ ≈ 0.5536870145713175 atol=1e-4
    @test γ♯ ≈ 0.015463419098934016 atol=1e-4
end

@testset "Test dual model" begin
    svm_data = LOT.parse_libsvm(SVM_DATASET, Float64)
    X = LOT.to_dense(svm_data)
    y = svm_data.labels
    data = LOT.DualLogitData(X, y)
    n = LOT.ndata(data)
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
        λ = zeros(n)
        g = zeros(n)
        glm = LOT.DualLogisticRegressor(data, penalty)
        @test LOT.loss(λ, glm) == 0.0
        LOT.gradient!(g, λ, glm)

        glm = LOT.DualLogisticRegressor(X, y)
        @test LOT.loss(λ, glm) == 0.0
        LOT.gradient!(g, λ, glm)
    end
    @testset "Fitting (dual)" begin
        glm = LOT.DualLogisticRegressor(data, penalty)
        f, gradient!, _, _ = LOT.generate_callbacks(glm)
        algo = BFGS()
        options = Optim.Options(iterations=250, g_tol=1e-5)
        lower = LOT.lowerbound(data)
        upper = LOT.upperbound(data)
        x0 = 0.5 * (lower .+ upper)
        res_joptim = Optim.optimize(f, gradient!, lower, upper,
                                    x0, Fminbox(algo), options)
        @test Optim.converged(res_joptim)
    end
end
