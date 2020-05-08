
using Optim
import Optim: optimize

struct LogisticOptimizer
    penalty::AbstractPenalty
    dual::Bool
    fit_intercept::Bool
    algo::Optim.AbstractOptimizer
    options::Optim.Options
end

function LogisticOptimizer(; penalty=L2Penalty(0.0),
                             dual=false,
                             fit_intercept=false,
                             algo=Optim.BFGS(),
                             options=Optim.Options())
    return LogisticOptimizer(penalty, dual, fit_intercept, algo, options)
end

function LogisticRegressor(X::AbstractArray{T, 2}, y::AbstractVector{T},
                           logopt::LogisticOptimizer) where T
    Dataset = issparse(X) ? SparseLogitData : LogitData
    data = Dataset(X, y)
    return LogisticRegressor(data, logopt.penalty, NULL_HASH, logopt.fit_intercept, NoLogger())
end

function Optim.optimize(logreg::LogisticRegressor,
                        x0::AbstractVector,
                        algo::Optim.AbstractOptimizer,
                        options::Optim.Options)
    f, g!, _, _ = generate_callbacks(logreg)
    res = Optim.optimize(f, g!, x0, algo, options)
    return res
end

function fit!(logopt::LogisticOptimizer,
              X::AbstractArray{T, 2},
              y::AbstractVector{T},
              x0=zeros(size(X, 2))) where T
    logreg = LogisticRegressor(X, y, logopt)
    res = Optim.optimize(logreg, x0, logopt.algo, logopt.options)
    println(logreg.fit_intercept)
    return res
end
