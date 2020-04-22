
# Numerically stable expit
function expit(t::T) where T
    if t >= 0.0
        return one(T) / (one(T) + exp(-t))
    else
        e_t = exp(t)
        return e_t / (one(T) + e_t)
    end
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

"Compute Fenchel transform of f(x) = log(1 + exp(-x))."
function logloss(λ::T) where T <: Real
    # TODO: numerically robust version
    if λ ∈ [0.0, -1.0]
        return 0.0
    elseif -1.0 < λ < 0.0
        return (1.0 + λ) * log(1.0 + λ) - λ * log(-λ)
    else
        return Inf
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

function scale!(::NormalScaler, X::Array{T, 2}) where T
    n, d = size(X)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)

    @inbounds for i in 1:d
        X[:, i] .= (X[:, i] .- μ[i]) ./ σ[i]
    end
end

"""
Format labels [y_1, ..., y_n] to ensure that y_i ∈ { -1, 1}
for all index i.
"""
function format_label!(y::AbstractVector)
    set_reference = [-1.0, 1.0]
    elts = unique(y)
    if union(elts, set_reference) != set_reference
        intersets = intersect(elts, set_reference)
        replacements = setdiff(set_reference, intersets)
        # For all elements not in reference set
        count = 1
        for elt in setdiff(elts, set_reference)
            y[y .== elt] .= replacements[count]
            count += 1
        end
    end
    return nothing
end
