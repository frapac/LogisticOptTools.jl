
using Optim
import Optim: optimize

# Fitting logistic regression
#
# TODO: add support for dual formulation
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
    return res
end


# Fitting optimal regression paramater (ℓ2)
abstract type AbstractL2PenaltyOptimizer end

struct L2PenaltyOptimizer
    kfold::Int
    shuffle::Bool
    fit_intercept::Bool
    outer_algo::Optim.AbstractOptimizer
    inner_algo::Optim.AbstractOptimizer
    options::Optim.Options
end
function L2PenaltyOptimizer( ; kfold=5,
                               shuffle=false,
                               fit_intercept=false,
                               outer_algo=BFGS(),
                               inner_algo=LBFGS(),
                               options=Optim.Options()
                            )
    return L2PenaltyOptimizer(kfold, shuffle, fit_intercept, outer_algo,
                              inner_algo, options)
end

function crossvalid_loss(X, y, γ, penopt::L2PenaltyOptimizer)
    kfold = penopt.kfold
    n, p = size(X)
    # Set-up permutation for cross validation
    if penopt.shuffle
        folds = randperm(n)
    else
        folds = vec(1:n)
    end
    #
    modk = div(n, kfold)
    start = 1
    stop = modk

    # Allocate a Hessian matrix
    ∇²f = zeros(p, p)
    # Get a view on upper-triangular
    h_train = @view ∇²f[triul(p)]
    # Allocate a gradient
    g_test = zeros(p)

    cost = 0.0
    ∇c   = 0.0
    w♯ = zeros(p)

    for i in 1:kfold
        # Split training and testing sets
        ind_train = folds[start:stop]
        ind_test = setdiff(1:n, ind_train)
        start += modk
        # Be careful we stay inside authorized indexes
        stop = min(stop + modk, n)
        Xtrain, ytrain = X[ind_train, :], y[ind_train]
        Xtest, ytest  = X[ind_test, :], y[ind_test]

        # First step: Training logreg
        logtrain = LogisticRegressor(Xtrain, ytrain, fit_intercept=false,
                                     penalty=L2Penalty(γ))
        # Inner optimization
        res = Optim.optimize(logtrain, w♯,
                            penopt.inner_algo,
                            penopt.options)
        p = res.minimizer
        # Compute Hessian
        hessian!(h_train, p, logtrain)

        # Second step: get prediction on testing subset
        # No penalty for testing!
        logtest = LogisticRegressor(Xtest, ytest, fit_intercept=false)
        loss_test = loss(p, logtest)
        # Compute gradient on testing set
        gradient!(g_test, p, logtest)

        # Get gradient w.r.t. current penalty
        ∇θ = - Symmetric(∇²f) \  p
        ∇c += 1 / kfold * dot(g_test, ∇θ)
        # Update cost
        cost += 1 / kfold * loss_test

        # Set hotstart for future computation
        w♯ .= p
    end
    return cost, ∇c
end

function fit!(penopt::L2PenaltyOptimizer,
              X::AbstractArray{T, 2},
              y::AbstractVector{T},
              γ0::Real) where T
    # Compute loss and gradient in a closure for Optim
    function fg!(f, g, x)
        c, ∇c = crossvalid_loss(X, y, x[1], penopt)
        if g != nothing
            g[1] = ∇c
        end
        if f != nothing
            return c
        end
    end
    res = Optim.optimize(Optim.only_fg!(fg!), [γ0], penopt.outer_algo, penopt.options)
    return res.minimum, res.minimizer[1]
end
