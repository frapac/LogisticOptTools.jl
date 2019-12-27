using LinearAlgebra, SparseArrays
using LogisticOptTools
const LOT = LogisticOptTools
BLAS.set_num_threads(1)

const LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

##################################################
# Some parameters
DATASET = "diabetes"
#= DATASET = "covtype.libsvm.binary.bz2" =#
LOAD_DATA     = true
FIT_KNITRO    = true
FIT_LBFGS     = false
FIT_LIBLINEAR = false
FIT_JOPTIM    = true
FIT_JSOOPTIM  = true
PENALTY       = 1e-4
PENALTY_TYPE  = "l2"
##################################################

function real_dataset(file)
    res = @time LOT.parse_libsvm(file)
    # Convert to dense matrix
    X = LOT.to_dense(res)
    # Some preprocessing
    LOT.scale!(LOT.NormalScaler(), X)
    y = copy(res.labels)
    # Special preprocessing for covtype
    y[y .== 2] .= -1
    return X, y
end

# If not here, download the file
if !isfile(DATASET)
    download("$LIBSVM_URL/$DATASET", DATASET)
end

# Load dataset as dense matrices
if LOAD_DATA
    X, y = real_dataset(DATASET)
    penalty = (PENALTY_TYPE == "l2") ? LOT.L2Penalty(PENALTY) : LOT.LinearizedL1Penalty(PENALTY)
end


# Warn: require KNITRO
if FIT_KNITRO
    using KNITRO
    res_kn = knfit(X, y, penalty=penalty, hessopt=2)
end


# Warn: require LIBLINEAR
if FIT_LIBLINEAR
    using LIBLINEAR
    if PENALTY_TYPE == "l2"
        res_liblinear = linear_train(y, X', C=1.0/(size(X, 1)*PENALTY), verbose=true, solver_type=Cint(0), eps=1e-5)
    else
        res_liblinear = linear_train(y, X', C=1.0/(size(X, 1)*PENALTY), verbose=true, solver_type=Cint(7), eps=1e-5)
    end
end


# Warn: require L-BFGS-B
# L-BFGS-B supports only smooth penalty
if FIT_LBFGS && PENALTY_TYPE == "l2"
    using LBFGSB
    function callback_builder(dat::Union{LOT.LogitData, LOT.LogisticProblem})
        eval_f = x -> LOT.loss(x, dat)
        function eval_g(g, x)
            LOT.gradient!(g, x, dat)
            push!(dat.logger.grad, norm(g, Inf))
            push!(dat.logger.time, time())
        end
        return (eval_f, eval_g)
    end

    function bfgs_fit(dat::LogitData, penalty=LOT.L2Penalty(0.0); trace=false)
        logit = LOT.GeneralizedLinearModel(dat, penalty, LOT.Tracer{Float64}())
        f, g! = callback_builder(logit)
        optimizer = L_BFGS_B(2048, 17)
        n = dim(dat)  # the dimension of the problem
        x = fill(Cdouble(0e0), n)  # the initial guess
        # set up bounds
        bounds = zeros(3, n)
        fout, xout = optimizer(f, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=1, maxfun=15000, maxiter=15000)
        return logit
    end

    res_lbfgsb = bfgs_fit(LogitData(X, y), penalty)
end


# Warn: require Optim.jl
# Optim supports only smooth penalty (currently)
if FIT_JOPTIM && PENALTY_TYPE == "l2"
    using Optim, LineSearches
    jcall(X, y; penalty=LOT.L2Penalty(0.0)) = jcall(LOT.LogitData(X, y), penalty)

    function jcall(dat::LOT.LogitData, penalty::LOT.AbstractPenalty)
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
    f, g! = jcall(X, y, penalty=penalty)
    algo = BFGS(alphaguess=InitialHagerZhang(α0=1.0), linesearch=LineSearches.HagerZhang())
    #= algo = BFGS(alphaguess=InitialStatic(alpha=1.0), linesearch=LineSearches.MoreThuente()) =#
    #= algo = BFGS(alphaguess=InitialStatic(alpha=1.0), linesearch=LineSearches.StrongWolfe()) =#
    options = Optim.Options(iterations=250, show_trace=true, g_tol=1e-5, allow_f_increases=true)
    res_joptim = Optim.optimize(f, g!, zeros(size(X, 2)), algo, options)
end


# Fit with JuliaSmoothOptimizer
if FIT_JSOOPTIM && PENALTY_TYPE == "l2"
    using JSOSolvers, NLPModels

    mutable struct LogReg <: AbstractNLPModel
        meta::NLPModelMeta
        counters::Counters
        data
    end

    function LogReg(X::Array{T, 2}, y::Vector{T}; penalty=LOT.L2Penalty(0.0)) where T
        n, d = size(X)
        meta = NLPModelMeta(d, x0=zeros(T, d),
                            name="Logit")

        return LogReg(meta, Counters(), LOT.GeneralizedLinearModel(LOT.LogitData(X, y), penalty, LOT.Tracer{Float64}()))
    end

    function NLPModels.obj(nlp :: LogReg, x :: AbstractVector)
        return LOT.loss(x, nlp.data)
    end

    function NLPModels.grad!(nlp :: LogReg, x :: AbstractVector, gx :: AbstractVector)
        LOT.gradient!(gx, x, nlp.data)
        push!(nlp.data.logger.grad, norm(gx, Inf))
        push!(nlp.data.logger.time, time())
        return gx
    end

    function NLPModels.hprod!(nlp :: LogReg, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0)
        LOT.hessvec!(Hv, x, v, nlp.data)
        return Hv
    end

    nlp = LogReg(X, y, penalty=penalty)
    res_jso = lbfgs(nlp)

end
