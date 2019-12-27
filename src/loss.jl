abstract type AbstractLoss end

"Logisitic loss."
struct Logit <: AbstractLoss end

function loss(::Logit, x::T, y::T) where T
    return -log1pexp(x * y)
end
