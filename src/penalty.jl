
abstract type AbstractPenalty end

"Compute inplace gradient of `penalty`."
gradient!(g, x, penalty::AbstractPenalty) = nothing

"Compute inplace Hessian of `penalty`."
hess!(hess, x, penalty::AbstractPenalty) = nothing

"Compute inplace Hessian vector product of `penalty`."
hessvec!(hv, x, vec, penalty::AbstractPenalty) = nothing


# L2 penalty
struct L2Penalty{T} <: AbstractPenalty
    constant::T
end
(penalty::L2Penalty{T})(x::Vector{T}) where T = penalty.constant * dot(x, x)

function gradient!(g::Vector{T}, x::Vector{T}, penalty::L2Penalty{T}) where T
    @inbounds for i in 1:length(x)
        g[i] += T(2) * penalty.constant * x[i]
    end
    return nothing
end

function hess!(hess::Vector{T}, x::Vector{T}, penalty::L2Penalty{T}) where T
    n = length(x)
    count = 1
    for i in 1:n
        @inbounds hess[count] += 2.0 * penalty.constant
        count += n - i + 1
    end
    return nothing
end

function hessvec!(hv::Vector{T}, x::Vector{T}, vec::Vector{T}, penalty::L2Penalty{T}) where T
    @inbounds for i in 1:length(x)
        hv[i] += T(2) * penalty.constant * vec[i]
    end
    return nothing
end


# L1 penalty
abstract type AbstractL1Penalty <: AbstractPenalty end

# Vanilla L1 penalty
struct L1Penalty{T} <: AbstractL1Penalty
    constant::T
end
function (penalty::L1Penalty{T})(x::Vector{T}) where T
    return penalty.constant * sum(abs.(x))
end
function gradient!(g::Vector{T}, x::Vector{T}, penalty::L1Penalty{T}) where T
    @inbounds for i in 1:length(x)
        g[i] += penalty.constant * sign(x[i])
    end
    return nothing
end

# Linearized L1 penalty (require to define linearization apart)
struct LinearizedL1Penalty{T} <: AbstractL1Penalty
    constant::T
end

function (penalty::LinearizedL1Penalty{T})(x::Vector{T}) where T
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
function (penalty::L0Penalty{T})(x::Vector{T}) where T
    return penalty.inner_penalty(x)
end
function gradient!(g::Vector{T}, x::Vector{T}, penalty::L0Penalty{T}) where T
    gradient!(g, x, penalty.inner_penalty)
end
function hess!(hess::Vector{T}, x::Vector{T}, penalty::L0Penalty{T}) where T
    hess!(hess, x, penalty.inner_penalty)
end
function hessvec!(hv::Vector{T}, x::Vector{T}, vec::Vector{T}, penalty::L0Penalty{T}) where T
    hessvec!(hv, x, vec, penalty.inner_penalty)
end

# TODO: Some work in progress
# See: https://francisbach.com/the-eta-trick-reloaded-multiple-kernel-learning/
struct EtaTrickL1Penalty{T} <: AbstractL1Penalty
    constant::T
end

function (penalty::EtaTrickL1Penalty{T})(x::Vector{T}) where T
    acc = zero(T)
    @inbounds for i in 1:length(x)
        acc +=0
    end
    return T(0.5) * penalty.constant * acc
end
function gradient!(g, x, penalty::EtaTrickL1Penalty)
    dimx = div(length(x), 2)
    @inbounds for i in dimx+1:2*dimx
        g[i] += penalty.constant
    end
end

