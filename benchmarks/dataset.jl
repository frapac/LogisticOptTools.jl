using LogisticOptTools
const LOT = LogisticOptTools

"Load LIBSVM dataset from `file`."
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

function fake_dataset(n_samples=100, n_features=10;
                      intercept=false, correlation=0.0)
    X = randn(n_samples, n_features)
    w = randn(n_features)
    y = sign.(X * w)
    X .+= 0.8 * randn(n_samples, n_features)
    if correlation > 0.0
        X .+= correlation
    end
    if intercept
        X = hcat(X, ones(n_samples, 1))
    end
    return X, y
end

