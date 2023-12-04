#=
***************************************************************************************888
PLEASE NOTE:
This needs to be modified or deleted to include Jackson's RL wrappers.
If needed, make changes to the old LaserTagModule.jl file and delete this one. 
=#

module OldLaserTag

# include("LaserTag_rl.jl")

using RLAlgorithms.Spaces: Discrete, Box
using RLAlgorithms.CommonRLExtensions
using CommonRLInterface.Wrappers
using Parameters

export LaserTagWrapper, LaserTagBeliefMDP

@with_kw struct LaserTagWrapper{T<:LaserTagBeliefMDP} <: Wrappers.AbstractWrapper
    env::T = LaserTagBeliefMDP()
    steps::MVector{1,Int} = @MVector [0] #HG: Ask Jackson what @MVector even means!
    max_steps::Int = 500
end

Wrappers.wrapped_env(wrap::LaserTagWrapper) = wrap.env

function RL.observe(wrap::LaserTagWrapper)
    o = Float32.(observe(wrap.env))
    @assert !any(isnan, o)
    o[1:2] ./= wrap.env.size
    for i = 3:length(o)
        iszero(o[i]) && continue
        o[i] = max(0, 1+log10(o[i])/2)
    end
    return o
end

RL.act!(wrap::LaserTagWrapper, a::Integer) = act!(wrap.env, actions(wrap.env)[a])
function RL.act!(wrap::LaserTagWrapper, a::AbstractArray)
    wrap.steps[] += 1
    RL.act!(wrap, a[])/50
end

function RL.reset!(wrap::LaserTagWrapper)
    wrap.steps[] = 0
    reset!(wrap.env)
end

RL.actions(wrap::LaserTagWrapper) = Discrete(length(actions(wrap.env)))
RL.observations(::LaserTagWrapper) = Box(fill(-Inf32,72), fill(Inf32,72))
CommonRLExtensions.truncated(wrap::LaserTagWrapper) = wrap.steps[] >= wrap.max_steps

end
