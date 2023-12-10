using SolveBeliefMDP, RLAlgorithms
using CommonRLInterface, Plots, Statistics, BSON
using StaticArrays
using Flux

const RL = CommonRLInterface

include("mgf.jl")

CIplot(xdata, ydata; args...) = CIplot!(plot(), xdata, ydata; args...)
function CIplot!(p, xdata, ydata; Nx=500, z=1.96, k=5, c=1, label=false, plotargs...)
    dx = (maximum(xdata)-minimum(xdata))/Nx
    x = (minimum(xdata) + dx/2) .+ dx*(0:Nx-1)
    y = zeros(Nx)
    dy = zeros(Nx)
    for i in eachindex(x)
        y_tmp = ydata[(x[i]-dx*(1/2+k)) .≤ xdata .≤ (x[i]+dx*(1/2+k))]
        y[i] = mean(y_tmp)
        dy[i] = z*std(y_tmp)/sqrt(length(y_tmp))
    end
    plot!(p, x, y-dy; fillrange=y+dy, fillalpha=0.3, c, alpha=0, label=false)
    plot!(p, x, y; c, label, plotargs...)
    return p
end

function plot_LoggingWrapper(env)
    hist = get_info(env)["LoggingWrapper"]
    p1 = CIplot(hist["steps"], hist["reward"], label="Undiscounted", c=1, title="Episodic Reward", ylims=(-200,100))
    CIplot!(p1, hist["steps"], hist["discounted_reward"], label="Discounted", c=2, legend=:bottomright)
    p2 = CIplot(hist["steps"], hist["episode_length"]; title="Episode Length", xlabel="Steps", ylims=(0,300))
    plot(p1, p2; layout=(2,1))    
end

@kwdef mutable struct LaserTagWrapper{T<:LaserTagBeliefMDP} <: Wrappers.AbstractWrapper
    const env::T # = DiscreteLaserTagBeliefMDP(), = ContinuousLaserTagBeliefMDP
    const max_steps::Int = 500
    steps::Int = 0
    const reward_scale::Float64 = 1.
end

Wrappers.wrapped_env(wrap::LaserTagWrapper) = wrap.env

CommonRLExtensions.truncated(wrap::LaserTagWrapper) = wrap.steps[] >= wrap.max_steps

function RL.reset!(wrap::LaserTagWrapper)
    wrap.steps = 0
    reset!(wrap.env)
end


function RL.observations(wrap::LaserTagWrapper{<:ExactBeliefLaserTag})
    s = wrap.env.state
    o1 = Box(Float32, size(s.robot_pos))
    o2 = Box(Float32, size(s.belief_target))
    TupleSpace(o1, o2)
end
function RL.observe(wrap::LaserTagWrapper{<:ExactBeliefLaserTag})
    s = wrap.env.state
    pos = convert(AbstractArray{Float32}, s.robot_pos)
    belief = convert(AbstractArray{Float32}, s.belief_target)
    o1 = pos ./ wrap.env.size
    o2 = @. max(0, 1+log10(belief)/3)
    return (o1, o2)
end

function RL.observations(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag})
    s = wrap.env.state
    o1 = Box(Float32, 2*length(s.robot_pos))
    o2 = Box(Float32, (2, length(s.belief_target.collection.particles) ))
    TupleSpace(o1, o2)
end
function RL.observe(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag})
    s = wrap.env.state
    pos = convert(AbstractArray{Float32}, s.robot_pos)
    belief = convert(AbstractArray{Float32}, stack(s.belief_target.collection.particles))
    mu = mean(belief; dims=2)
    o1 = [vec(pos) ./ wrap.env.size; vec(mu) ./ wrap.env.size] 
    o2 = belief .- mu
    return (o1, o2)
end

RL.actions(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}) = Discrete(length(actions(wrap.env)))

const idx2act = Dict(i=>s for (i,s) in enumerate(keys(LaserTag.actiondir)))
RL.act!(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}, a::AbstractArray) = act!(wrap, a[])
function RL.act!(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}, a::Integer)
    wrap.steps += 1
    r = act!(wrap.env, idx2act[a])
    r * convert(typeof(r), wrap.reward_scale)
end

RL.actions(::LaserTagWrapper{<:ContinuousActionLaserTag}) = TupleSpace(Box(lower=[-1f0, -1f0], upper=[1f0, 1f0]), Discrete(2))
RL.act!(::LaserTagWrapper{<:ContinuousActionLaserTag}, a) = @assert false "action type error"
RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractVector, <:AbstractArray}) = act!(wrap, (a[1],a[2][]))
function RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractVector, <:Integer})
    wrap.steps += 1
    if a[2] == 1
        r = act!(wrap.env, :measure)
    else
        r = act!(wrap.env, a[1])
    end
    r * convert(typeof(r), wrap.reward_scale)
end

# RL.actions(::LaserTagWrapper{<:ContinuousActionLaserTag}) = TupleSpace(Box(lower=[-1f0], upper=[1f0]), Discrete(2))
# RL.act!(::LaserTagWrapper{<:ContinuousActionLaserTag}, a) = @assert false "action type error"
# RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractArray, <:AbstractArray}) = act!(wrap, (a[1][],a[2][]))
# function RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractFloat, <:Integer})
#     wrap.steps += 1
#     if a[2] == 1
#         r = act!(wrap.env, :measure)
#     else
#         x = cos(a[1])
#         y = sin(a[1])
#         _a = SA[x, y] ./ max(abs(x), abs(y))
#         r = act!(wrap.env, _a)
#     end
#     r * convert(typeof(r), wrap.reward_scale)
# end

symlog(x) = @. sign(x) * log(abs(x) + 1)
symexp(x) = @. sign(x) * (exp(abs(x)) - 1)







# shared = Chain(
#     x -> (view(x,1:2,:), reshape(view(x,3:72,:), (10, 7, 1, :))),
#     Parallel(
#         vcat,
#         identity,
#         Chain(
#             Conv((3,2), 1=>32, relu), # 8 x 6
#             Conv((3,2), 32=>32, relu), # 8 x 5
#             Conv((3,2), 32=>32, relu), # 7 x 4
#             Flux.flatten
#         )
#     )
# ),
# shared_out_size = 2+4*4*32,


function evaluate(test_env, ac; discount=0.997)
    reset!(test_env)
    r = 0.0
    for t in 1:1000
        s = observe(test_env)
        a = ac(s)
        r += act!(test_env, a) * discount ^ (t-1)
        terminated(test_env) && break
    end
    r
end


