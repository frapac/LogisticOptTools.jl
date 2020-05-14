using Test
using Ipopt, ECOS
using MathOptInterface

const MOI = MathOptInterface

function fit_nlp(constructor, logreg; hessian=false, hessvec=false)
    opt = constructor()
    MOI.set(opt, MOI.Silent(), true)
    evaluator = LOT.LogisticEvaluator(logreg, hessian, hessvec)
    LOT.load!(LOT.NonLinearFormulation(), opt, evaluator, logreg)
    MOI.optimize!(opt)
    obj = MOI.get(opt, MOI.ObjectiveValue())
    rc = MOI.get(model, MOI.TerminationStatus())
    MOI.empty!(opt)
    return rc, obj
end

function fit_conic(constructor, logreg)
    opt = constructor()
    MOI.set(opt, MOI.Silent(), true)
    cache = MOIU.UniversalFallback(MOIU.Model{Float64}())
    cached = MOIU.CachingOptimizer(cache, opt)
    bridged = MOIB.full_bridge_optimizer(cached, Float64)
    LOT.load!(LOT.ConicFormulation(), bridged, logreg)
    MOI.optimize!(bridged)
    obj = MOI.get(opt, MOI.ObjectiveValue())
    rc = MOI.get(model, MOI.TerminationStatus())
    MOI.empty!(opt)
    return rc, obj
end

@testset "Test MOI formulation" begin
    dataset = LOT.LogitData(SVM_DATASET, scale_data=true)

    obj_nopenalty = 0.530720398638368
    obj_l2penalty = 0.5463713982967904
    obj_l1penalty = 0.5566813776579644

    @testset "Test NLP formulation" begin
        logreg = LOT.LogisticRegressor(dataset.X, dataset.y, penalty=LOT.L2Penalty(0.0),
                                       fit_intercept=false)
        for activate_hessian in [false, true]
            rc, obj = fit_nlp(Ipopt.Optimizer, logreg, hessian=activate_hessian)
            @test rc == MOI.OPTIMAL
            @test obj ≈ obj_nopenalty
        end
        logreg.penalty = LOT.L2Penalty(1e-2)
        for activate_hessian in [false, true]
            rc, obj = fit_nlp(Ipopt.Optimizer, logreg, hessian=activate_hessian)
            @test rc == MOI.OPTIMAL
            @test obj ≈ obj_l2penalty
        end
    end

    @testset "Test Conic formulation" begin
        logreg = LOT.LogisticRegressor(dataset.X, dataset.y, penalty=LOT.L2Penalty(0.0),
                                       fit_intercept=false)
        # No penalty
        rc, obj = fit_conic(ECOS.Optimizer, logreg)
        @test rc == MOI.OPTIMAL
        @test obj ≈ obj_nopenalty

        # L2 penalty
        logreg.penalty = LOT.L2Penalty(1e-2)
        rc, obj = fit_conic(ECOS.Optimizer, logreg)
        @test rc == MOI.OPTIMAL
        # Test is currently broken as conic formulation uses |w|_2 as penalty,
        # whereas NLP formulation uses |w|_2^2
        @test_broken obj ≈ obj_nopenalty

        # L2 penalty
        logreg.penalty = LOT.L1Penalty(1e-2)
        rc, obj = fit_conic(ECOS.Optimizer, logreg)
        @test rc == MOI.OPTIMAL
        # Test is currently broken as conic formulation uses |w|_2 as penalty,
        # whereas NLP formulation uses |w|_2^2
        @test obj ≈ obj_l1penalty
    end
end

