"Wrapper of KNITRO for logistic regression."


function callbackEvalF(kc, cb, evalRequest, evalResult, dat)
    x = evalRequest.x
    evalResult.obj[1] = LOT.loss(x, dat)
    return 0
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, dat)
    ω = evalRequest.x
    LOT.gradient!(evalResult.objGrad, ω, dat)
    return 0
end

function callbackEvalHv!(kc, cb, evalRequest, evalResult, dat)
    h = evalRequest.x
    vec = evalRequest.vec
    LOT.hessvec!(evalResult.hessVec, h, vec, dat)
    return 0
end

function callbackEvalH!(kc, cb, evalRequest, evalResult, dat)
    x = evalRequest.x
    vec = evalRequest.vec
    LOT.hess!(evalResult.hess, x, dat)
    return 0
end

function callbackNewPoint(kc, x, lambda_, m)
    dat = m.callbacks[1].userparams[:data]
    # Query information about the current problem.
    dFeasError = KNITRO.KN_get_abs_opt_error(m)
    if isdefined(dat.logger)
        push!(dat.logger.grad, dFeasError)
        push!(dat.logger.time, time())
    end
    return 0
end

knfit(X, y; x0=zeros(size(X, 2)), penalty=LOT.L2Penalty(0.0), options...) = knfit(LOT.LogitData(X, y), x0, penalty, options)
function knfit(dat::LOT.LogitData, x0::AbstractVector, penalty::P, options) where P <: Union{LOT.L2Penalty, LOT.L1Penalty}
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, LOT.dim(dat))
    KNITRO.KN_set_var_primal_init_values(kc, x0)
    cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)
    KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)
    # Parse options
    for val in options
        optval = string(val.first)
        if optval == "trace" && val.second == true
            KNITRO.KN_set_newpt_callback(kc, callbackNewPoint)
            continue
        end
        if optval == "hessopt" && val.second == 5
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalHv!)
            KNITRO.KN_set_param(kc, "algorithm", 2)
        elseif optval == "hessopt" && val.second == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH!)
        end
        KNITRO.KN_set_param(kc, optval, val.second)
    end

    logit = LOT.GeneralizedLinearModel(dat, penalty, LOT.Tracer{Float64}())
    KNITRO.KN_set_cb_user_params(kc, cb, logit)

    nStatus = @time KNITRO.KN_solve(kc)
    w_opt = KNITRO.get_solution(kc)
    KNITRO.KN_free(kc)
    return w_opt, logit.logger
end

function knfit(dat::LOT.LogitData, x0, penalty::LOT.LinearizedL1Penalty, options)
    n_features = LOT.dim(dat)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, 2 * LOT.dim(dat))
    index_w = collect(Cint, 0:(n_features-1))
    index_z = collect(Cint, n_features:(2*n_features-1))
    w0 = x0
    z0 = -w0
    KNITRO.KN_set_var_primal_init_values(kc, index_w, w0)
    KNITRO.KN_set_var_primal_init_values(kc, index_z, z0)
    #= KNITRO.KN_set_var_primal_init_values(kc, randn(2*dim(dat))) =#
    cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)
    KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)
    # Parse options
    for val in options
        optval = string(val.first)
        if optval == "trace" && val.second == true
            KNITRO.KN_set_newpt_callback(kc, callbackNewPoint)
            continue
        end
        if optval == "hessopt" && val.second == 5
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalHv!)
            KNITRO.KN_set_param(kc, "algorithm", 2)
        elseif optval == "hessopt" && val.second == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH!)
        end
        KNITRO.KN_set_param(kc, optval, val.second)
    end

    # Constraints definition
    # 2n linear constrains for l1 norm :
    # 0 <= z_i + w_i,
    # -z_i + w_i <= 0,      for i = 0, ..., n-1
    KNITRO.KN_add_cons(kc, 2 * n_features)

    index_cons = collect(Cint, 0:(2*n_features-1))
    KNITRO.KN_set_con_lobnds(kc, index_cons[1:n_features], zeros(n_features))
    KNITRO.KN_set_con_upbnds(kc, index_cons[n_features+1:end], zeros(n_features))

    index_vars_z = collect(Cint, vcat(n_features:(2*n_features-1), n_features:(2*n_features-1)))
    index_vars_w = collect(Cint, vcat(0:(n_features-1), 0:(n_features-1)))
    coefs_w = ones(2 * n_features)
    coefs_z = vcat(ones(n_features), -ones(n_features))

    KNITRO.KN_add_con_linear_struct(kc, index_cons, index_vars_z, coefs_z)
    KNITRO.KN_add_con_linear_struct(kc, index_cons, index_vars_w, coefs_w)

    # Linear objective
    coefs = fill(penalty.constant, n_features)
    index_obj = collect(Cint, n_features:(2*n_features-1))
    KNITRO.KN_add_obj_linear_struct(kc, index_obj, coefs)

    logit = LOT.GeneralizedLinearModel(dat, penalty, LOT.Tracer{Float64}())
    KNITRO.KN_set_cb_user_params(kc, cb, logit)

    nStatus = @time KNITRO.KN_solve(kc)
    w♯ = KNITRO.get_solution(kc)
    KNITRO.KN_free(kc)
    return w♯, logit.logger
end
