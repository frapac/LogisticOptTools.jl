"Wrapper of LBFGSB for logistic regression."

using LBFGSB
using LinearAlgebra

function callback_builder(dat::Union{LOT.LogitData, LOT.GeneralizedLinearModel})
    eval_f = x -> LOT.loss(x, dat)
    function eval_g(g, x)
        LOT.gradient!(g, x, dat)
        push!(dat.logger.obj, LOT.loss(x, dat))
        push!(dat.logger.grad, norm(g, Inf))
        push!(dat.logger.time, time())
    end
    return (eval_f, eval_g)
end

function bfgs_fit(dat::LOT.LogitData, penalty=LOT.L2Penalty(0.0); trace=false)
    logit = LOT.GeneralizedLinearModel(dat, penalty, LOT.Tracer{Float64}())
    f, g! = callback_builder(logit)
    optimizer = L_BFGS_B(2048, 17)
    n = LOT.dim(dat)  # the dimension of the problem
    x = fill(Cdouble(0e0), n)  # the initial guess
    # set up bounds
    bounds = zeros(3, n)
    fout, xout = optimizer(f, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=1, maxfun=15000, maxiter=15000)
    return logit.logger
end
