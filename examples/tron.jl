
using LogisticOptTools
using JSOSolvers, NLPModels

const LOT = LogisticOptTools

mutable struct LogReg <: AbstractNLPModel
    meta::NLPModelMeta
    counters::Counters
    logreg::LOT.LogisticRegressor
end

function LogReg(logreg)
    p = LOT.nfeatures(logreg)
    meta = NLPModelMeta(p, x0=zeros(p),
                        name="LogisticRegression")
    return LogReg(meta, Counters(), logreg)
end

function NLPModels.obj(nlp :: LogReg, x :: AbstractVector)
    return LOT.loss(x, nlp.logreg)
end

function NLPModels.grad!(nlp :: LogReg, x :: AbstractVector, gx :: AbstractVector)
    LOT.gradient!(gx, x, nlp.logreg)
    return gx
end

function NLPModels.hprod!(nlp :: LogReg, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0)
    LOT.hessvec!(Hv, x, v, nlp.logreg)
    return Hv
end

DATASET = "covtype.libsvm.binary.bz2"
if !isdefined(Main, :dataset)
    #= dataset = LOT.SparseLogitData(DATASET, scale_data=true) =#
    # Non-scaling data yields more difficult optimization problems
    dataset = LOT.LogitData(DATASET, scale_data=false)
end

penalty = LOT.L2Penalty(1e-2)
logreg = LOT.LogisticRegressor(dataset.X, dataset.y, penalty=penalty)

nlp = LogReg(logreg)
results = @time tron(nlp, atol=1e-6, max_time=300.0)
