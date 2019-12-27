
@inline function expit(t::T) where T
    return one(T) / (one(T) + exp(-t))
    # Another version of expit:
    #= return 0.5 + 0.5 * tanh(0.5 * t) =#
end

# A numerically robust function to evaluate -log(1 + exp(-t))
# See: http://fa.bianp.net/blog/2019/evaluate_logistic/
function log1pexp(t::T) where T
    if t < -33.3
        return t
    elseif t <= -18.0
        return t - exp(t)
    elseif t <= 37.0
        return -log1p(exp(-t))
    else
        return -exp(-t)
    end
end


abstract type AbstractScaler end

"Scale dataset before optimization."
scale!(::AbstractScaler, X::AbstractArray{T, 2}) where T = nothing


"""
Scale each feature in dataset by standard deviation.

```math
μ_j = mean(x_j)
σ_j = std(x_j)
x_j^+ =  (x_j - μ_j) / σ_j

```

"""
struct NormalScaler <: AbstractScaler end

function scale!(::NormalScaler, X::AbstractArray{T, 2}) where T
    n, d = size(X)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)

    @inbounds for i in 1:d
        X[:, i] .= (X[:, i] .- μ[i]) ./ σ[i]
    end
end
