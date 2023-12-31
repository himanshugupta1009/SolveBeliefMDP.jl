using SolveBeliefMDP, RLAlgorithms
using CommonRLInterface, Plots, Statistics, BSON
using StaticArrays
using Flux

const RL = CommonRLInterface

include("mgf.jl")

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
function RL.observe(v_env::VecEnv{<:LaserTagWrapper{<:ExactBeliefLaserTag}})
    sz = v_env.envs[1].env.size

    pos = stack(v_env.envs) do wrap
        state = wrap.env.state
        convert(AbstractArray{Float32}, state.robot_pos)
    end ./ sz

    belief = stack(v_env.envs) do wrap
        state = wrap.env.state
        convert(AbstractArray{Float32}, reinterpret(reshape, Int, state.belief_target))
    end
    @. belief = max(0, 1+log10(belief)/3)

    return (pos, belief)
end

function RL.observations(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag})
    s = wrap.env.state
    o1 = Box(Float32, length(s.robot_pos))
    o2 = Box(Float32, (2, length(s.belief_target.collection.particles) ))
    TupleSpace(o1, o2)
end
function RL.observe(v_env::VecEnv{<:LaserTagWrapper{<:ParticleBeliefLaserTag}})
    sz = v_env.envs[1].env.size

    pos = stack(v_env.envs) do wrap
        robot_pos = wrap.env.state.robot_pos
        convert(AbstractArray{Float32}, robot_pos)
    end ./ sz

    belief = stack(v_env.envs) do wrap
        particles = wrap.env.state.belief_target.collection.particles
        particle_arr = reinterpret(reshape, Int, particles)
        convert(AbstractArray{Float32}, particle_arr)
    end ./ sz

    return (pos, belief)
end
function RL.observe(env::LaserTagWrapper{<:ParticleBeliefLaserTag})
    sz = env.env.size

    robot_pos = env.env.state.robot_pos
    pos = convert(AbstractArray{Float32}, robot_pos) ./ sz

    particles = env.env.state.belief_target.collection.particles
    particle_arr = reinterpret(reshape, Int, particles)
    belief = convert(AbstractArray{Float32}, particle_arr) ./ sz

    return (pos, belief)
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
# RL.actions(::LaserTagWrapper{<:ContinuousActionLaserTag}) = TupleSpace(Box(lower=[-1f0], upper=[1f0]), Discrete(2))

RL.act!(::LaserTagWrapper{<:ContinuousActionLaserTag}, a) = @assert false "action type error"
RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractArray, <:AbstractArray}) = act!(wrap, (vec(a[1]),a[2][]))
function RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple{<:AbstractVector, <:Integer})
    a_c, a_d = a

    if length(a_c) == 1
        x = cos(pi*a_c[])
        y = sin(pi*a_c[])
        a_c = [x,y] ./ max(abs(x), abs(y))
    end

    wrap.steps += 1
    if a_d == 1
        r = act!(wrap.env, :measure)
    else
        r = act!(wrap.env, a_c)
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
        a = get_actionvalue(ac, Algorithms.ACInput(observation = s))[1].action
        r += act!(test_env, a) * discount ^ (t-1)
        terminated(test_env) && break
    end
    r
end


function get_mean(env, x; k=1)
    hist = get_info(env)["LoggingWrapper"]
    x_raw, y_raw = hist["steps"], hist["discounted_reward"]
    dx = x[2] - x[1]
    y = zeros(length(x))
    for i in eachindex(x)
        y_tmp = y_raw[(x[i]-dx*(1/2+k)) .≤ x_raw .≤ (x[i]+dx*(1/2+k))]
        y[i] = mean(y_tmp)
    end
    y
end

plot_seed_ci(solver_vec; args...) = plot_seed_ci!(plot(), solver_vec; args...)
function plot_seed_ci!(p, solver_vec; xmax=1_000_000, Nx=200, c=1, label=false, plotargs...)
    x = range(0, xmax, Nx)
    y_mat = stack([get_mean(solver.env, x; k=1) for solver in solver_vec])
    y = mean(y_mat; dims=2)
    dy = 1.96 * std(y_mat; dims=2) / sqrt(size(y_mat,2))
    println("$(y[end]) +/- $(dy[end])")
    plot!(p, x, y-dy; fillrange=y+dy, fillalpha=0.3, c, alpha=0, label=false)
    plot!(p, x, y; label, c, plotargs...)
    p
end

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