using KNITRO
using LogisticOptTools
const LOT = LogisticOptTools

function format!(tracer::LOT.Tracer; f♯=Inf)
    tracer.time .-= tracer.time[1]
    if ~isinf(f♯)
        tracer.obj .-= f♯
    end
end

function callbackEvalF(kc, cb, evalRequest, evalResult, dat)
    nfeat = LOT.nfeatures(dat) + dat.fit_intercept
    x = evalRequest.x[1:nfeat]
    evalResult.obj[1] = LOT.loss(x, dat)
    return 0
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, dat)
    nfeat = LOT.nfeatures(dat) + dat.fit_intercept
    ω = evalRequest.x[1:nfeat]
    LOT.gradient!(evalResult.objGrad, ω, dat)
    #= if isa(dat.logger, LOT.Tracer) =#
    #=     push!(dat.logger.grad, norm(evalResult.objGrad, Inf)) =#
    #=     push!(dat.logger.obj, LOT.loss(evalRequest.x[1:n], dat)) =#
    #=     push!(dat.logger.time, time()) =#
    #= end =#
    return 0
end

function callbackEvalHv!(kc, cb, evalRequest, evalResult, dat)
    nfeat = LOT.nfeatures(dat) + dat.fit_intercept
    h = evalRequest.x[1:nfeat]
    vec = evalRequest.vec[1:nfeat]
    LOT.hessvec!(evalResult.hessVec, h, vec, dat)
    return 0
end

function callbackEvalH!(kc, cb, evalRequest, evalResult, dat)
    nfeat = LOT.nfeatures(dat) + dat.fit_intercept
    x = evalRequest.x[1:nfeat]
    LOT.hessian!(evalResult.hess, x, dat)
    return 0
end

function callbackNewPoint(kc, x, lambda_, m)
    dat = m
    # Query information about the current problem.
    dFeasError = KNITRO.KN_get_abs_opt_error(kc)
    if isdefined(dat.logger)
        push!(dat.logger.grad, dFeasError)
        push!(dat.logger.time, time())
    end
    return 0
end

knfit(X, y; x0=zeros(size(X, 2)), penalty=LOT.L2Penalty(0.0), options...) = knfit(LOT.LogitData(X, y), x0, penalty, options)
function knfit(logit, x0::AbstractVector; options...) where P <: Union{LOT.L2Penalty, LOT.L1Penalty}
    p = length(x0)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, p)
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
            KNITRO.KN_set_cb_hess(kc, cb, 0, callbackEvalHv!)
            KNITRO.KN_set_param(kc, "algorithm", 2)
        elseif optval == "hessopt" && val.second == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH!)
        end
        KNITRO.KN_set_param(kc, optval, val.second)
    end

    KNITRO.KN_set_cb_user_params(kc, cb, logit)

    nStatus = @time KNITRO.KN_solve(kc)
    w_opt = KNITRO.get_solution(kc)
    KNITRO.KN_free(kc)
    return w_opt, logit.logger
end

function knfit(dat::LOT.AbstractDataset, x0, penalty::LOT.LinearizedL1Penalty; options...)
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
            KNITRO.KN_set_cb_hess(kc, cb, 0, callbackEvalHv!)
            KNITRO.KN_set_param(kc, "algorithm", 2)
        elseif optval == "hessopt" && val.second == 1
            # Define Hessian sparsity pattern
            xHess1 = Cint[]
            xHess2 = Cint[]
            for i in 1:n_features
                for j in i:n_features
                    push!(xHess1, i-1)
                    push!(xHess2, j-1)
                end
            end
            nnzh = length(xHess2)
            KNITRO.KN_set_cb_hess(kc, cb, nnzh, callbackEvalH!,
                                  hessIndexVars1=xHess1,
                                  hessIndexVars2=xHess2)
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


function knfit(dat::LOT.AbstractDataset, x0,
               penalty::LOT.L0Penalty;
               big_m=5.0, options...)
    # Formulation choices for (1 - z) w
    # * 0: qcqp
    # * 1: big-M
    # * 2: mpec
    formulation = 1
    # total number of features in dataset
    n_features = LOT.nfeatures(dat)
    # zero vector
    zero_n = zeros(n_features)
    # one vector
    one_n = ones(n_features)

    kc = KNITRO.KN_new()

    # Regularization parameters
    index_w = KNITRO.KN_add_vars(kc, n_features)
    KNITRO.KN_set_var_primal_init_values(kc, index_w, x0)
    KNITRO.KN_set_var_lobnds(kc, index_w, fill(-100.0, n_features))
    KNITRO.KN_set_var_upbnds(kc, index_w, fill(100.0, n_features))

    # Add callbacks
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
            KNITRO.KN_set_cb_hess(kc, cb, 0, callbackEvalHv!)
            KNITRO.KN_set_param(kc, "algorithm", 2)
        elseif optval == "hessopt" && val.second == 1
            # Define Hessian sparsity pattern
            xHess1 = Cint[]
            xHess2 = Cint[]
            for i in 1:n_features
                for j in i:n_features
                    push!(xHess1, i-1)
                    push!(xHess2, j-1)
                end
            end
            nnzh = length(xHess2)
            KNITRO.KN_set_cb_hess(kc, cb, nnzh, callbackEvalH!,
                                  hessIndexVars1=xHess1,
                                  hessIndexVars2=xHess2)
        else
            KNITRO.KN_set_param(kc, optval, val.second)
        end
    end

    # Add constraint \|w\|_0 <= k
    # Formulate with binary variables z:
    index_z = KNITRO.KN_add_vars(kc, n_features)
    KNITRO.KN_set_var_upbnds(kc, index_z, fill(1.0, n_features))
    KNITRO.KN_set_var_lobnds(kc, index_z, fill(0.0, n_features))
    # Add constraint sum z_i <= k
    l0_cons = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnds(kc, l0_cons, [penalty.constant])
    KNITRO.KN_add_con_linear_struct(kc,
                                    repeat(l0_cons, n_features),
                                    index_z,
                                    one_n)

    if formulation == 0
        # Flag z as binary
        KNITRO.KN_set_var_type.(Ref(kc),
                                index_z,
                                fill(KNITRO.KN_VARTYPE_BINARY, n_features))
        # Add constraint w (1 - z) = 0
        comp_cons = KNITRO.KN_add_cons(kc, n_features)
        KNITRO.KN_set_con_eqbnds(kc, comp_cons, fill(0.0, n_features))
        KNITRO.KN_add_con_linear_struct(kc, comp_cons, index_w, fill(1.0, n_features))
        KNITRO.KN_add_con_quadratic_struct(kc, comp_cons,
                                    index_w,
                                    index_z,
                                    fill(-1.0, n_features))
    elseif formulation == 1
        # Flag z as binary
        KNITRO.KN_set_var_type.(Ref(kc),
                                index_z,
                                fill(KNITRO.KN_VARTYPE_BINARY, n_features))
        # Add Big-M formulation
        comp_cons = KNITRO.KN_add_cons(kc, 2 * n_features)
        KNITRO.KN_set_con_upbnds(kc, comp_cons, fill(0.0, 2* n_features))
        # w <= M z
        up_cons = comp_cons[1:n_features]
        KNITRO.KN_add_con_linear_struct(kc, up_cons, index_w, fill(1.0, n_features))
        KNITRO.KN_add_con_linear_struct(kc, up_cons, index_z, fill(-big_m, n_features))
        # - M z <= w
        lo_cons = comp_cons[n_features+1:end]
        KNITRO.KN_add_con_linear_struct(kc, lo_cons, index_w, fill(-1.0, n_features))
        KNITRO.KN_add_con_linear_struct(kc, lo_cons, index_z, fill(-big_m, n_features))
    elseif formulation == 2
        # Flag z as binary
        KNITRO.KN_set_var_type.(Ref(kc),
                                index_z,
                                fill(KNITRO.KN_VARTYPE_BINARY, n_features))
        # x+
        x_pos = KNITRO.KN_add_vars(kc, n_features)
        KNITRO.KN_set_var_lobnds(kc, x_pos, zero_n)
        # x-
        x_neg = KNITRO.KN_add_vars(kc, n_features)
        KNITRO.KN_set_var_lobnds(kc, x_neg, zero_n)
        # |x|
        x_abs = KNITRO.KN_add_vars(kc, n_features)
        KNITRO.KN_set_var_lobnds(kc, x_abs, zero_n)
        # x = x+ - x-
        cons_xeq = KNITRO.KN_add_cons(kc, n_features)
        KNITRO.KN_set_con_eqbnds(kc, cons_xeq, zero_n)
        KNITRO.KN_add_con_linear_struct(kc,
            repeat(cons_xeq, 3),
            vcat(index_w, x_pos, x_neg),
            vcat(one_n, -one_n, one_n))
        # |x| = x+ + x-
        cons_absx = KNITRO.KN_add_cons(kc, n_features)
        KNITRO.KN_set_con_eqbnds(kc, cons_absx, zero_n)
        KNITRO.KN_add_con_linear_struct(kc,
            repeat(cons_absx, 3),
            vcat(x_abs, x_pos, x_neg),
            vcat(one_n, -one_n, -one_n))
        # Specify complementarities
        # |x| comp z
        cc_types = fill(KNITRO.KN_CCTYPE_VARVAR, 2*n_features)
        KNITRO.KN_set_compcons(kc,
                               cc_types,
                               vcat(x_abs, x_pos),
                               vcat(index_z, x_neg))
    end

    logit = LOT.LogisticRegressor(dat, penalty)
    KNITRO.KN_set_cb_user_params(kc, cb, logit)

    # Resolve!
    nStatus = @time KNITRO.KN_solve(kc)
    # Get optimal solution
    w♯ = KNITRO.get_solution(kc)
    # Free Knitro instance
    KNITRO.KN_free(kc)

    return w♯, logit.logger
end
