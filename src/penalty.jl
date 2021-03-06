
abstract type AbstractPenalty end

"Compute inplace gradient of `penalty`."
gradient!(g, x, penalty::AbstractPenalty) = nothing

"Compute inplace Hessian of `penalty`."
hessian!(hess, x, penalty::AbstractPenalty) = nothing

"Compute inplace diagonal for Hessian of `penalty`."
diaghess!(dh, x, penalty::AbstractPenalty) = nothing

"Compute inplace Hessian vector product of `penalty`."
hessvec!(hv, x, vec, penalty::AbstractPenalty) = nothing


# L2 penalty
struct L2Penalty{T} <: AbstractPenalty
    constant::T
end

(penalty::L2Penalty{T})(x::AbstractVector{T}) where T = penalty.constant * dot(x, x)

function gradient!(g::AbstractVector{T}, x::AbstractVector{T}, penalty::L2Penalty{T}) where T
    # Sanity check
    n = length(x)
    @assert length(g) == n
    @inbounds for i in 1:n
        g[i] += T(2) * penalty.constant * x[i]
    end
    return nothing
end

function hessian!(hess::AbstractVector{T}, x::AbstractVector{T}, penalty::L2Penalty{T}) where T
    # Sanity check
    n = length(x)
    @assert length(hess) == n * (n + 1) / 2
    count = 1
    for i in 1:n
        @inbounds hess[count] += 2.0 * penalty.constant
        # Fill only terms corresponding to diagonal
        count += n - i + 1
    end
    return nothing
end

function diaghess!(dh::AbstractVector{T}, x::AbstractVector{T}, penalty::L2Penalty{T}) where T
    # Sanity check
    n = length(x)
    @assert length(dh) == n
    for i in 1:n
        @inbounds dh[i] += 2.0 * penalty.constant
    end
    return nothing
end

function hessvec!(hv::AbstractVector{T}, x::AbstractVector{T}, vec::AbstractVector{T}, penalty::L2Penalty{T}) where T
    n = length(x)
    @assert length(hv) == length(vec) == n
    @inbounds for i in 1:n
        hv[i] += T(2) * penalty.constant * vec[i]
    end
    return nothing
end


abstract type AbstractL1Penalty <: AbstractPenalty end

# Vanilla L1 penalty
struct L1Penalty{T} <: AbstractL1Penalty
    constant::T
end

function (penalty::L1Penalty{T})(x::AbstractVector{T}) where T
    return penalty.constant * sum(abs.(x))
end

function gradient!(g::AbstractVector{T}, x::AbstractVector{T}, penalty::L1Penalty{T}) where T
    @inbounds for i in 1:length(x)
        g[i] += penalty.constant * sign(x[i])
    end
    return nothing
end

# Linearized L1 penalty (require to define linearization apart)
struct LinearizedL1Penalty{T} <: AbstractL1Penalty
    constant::T
end
function (penalty::LinearizedL1Penalty{T})(x::AbstractVector{T}) where T
    return T(0)
end

# L0 penalty
# Write as:
# \| w \|_0 \leq penalty.constant
# with \|w\|_0 the l0 norm
struct L0Penalty{T} <: AbstractPenalty
    constant::T
    inner_penalty::AbstractPenalty
end
L0Penalty(c::T) where T = L0Penalty(c, L2Penalty(T(0.0)))
L0Penalty(c::T, λ::T) where T = L0Penalty(c, L2Penalty(λ))

# L0 penalty
function (penalty::L0Penalty{T})(x::AbstractVector{T}) where T
    return penalty.inner_penalty(x)
end

function gradient!(g::AbstractVector{T}, x::AbstractVector{T}, penalty::L0Penalty{T}) where T
    gradient!(g, x, penalty.inner_penalty)
end

function hess!(hess::AbstractVector{T}, x::AbstractVector{T}, penalty::L0Penalty{T}) where T
    hess!(hess, x, penalty.inner_penalty)
end

function hessvec!(hv::AbstractVector{T}, x::AbstractVector{T}, vec::AbstractVector{T}, penalty::L0Penalty{T}) where T
    hessvec!(hv, x, vec, penalty.inner_penalty)
end

