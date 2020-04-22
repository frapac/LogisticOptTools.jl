
abstract type AbstractLogger end

"Return true if `logger` is instantiated."
isdefined(::AbstractLogger) = false

"Reinitialize logger."
empty!(::AbstractLogger) = nothing

struct NoLogger <: AbstractLogger end

"""
Store evolution of objective, gradient and time across iterations.

"""
mutable struct Tracer{T <: Real} <: AbstractLogger
    obj::Vector{T}
    grad::Vector{T}
    time::Vector{T}
end
Tracer{T}() where T = Tracer(T[], T[], T[])

function empty!(tracer::Tracer{T}) where T
    tracer.obj = T[]
    tracer.grad = T[]
    tracer.time = T[]
end

isdefined(t::Tracer) = true
