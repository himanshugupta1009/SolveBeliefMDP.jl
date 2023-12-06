using SolveBeliefMDP, RLAlgorithms
using CommonRLInterface, Plots, Statistics, BSON

const RL = CommonRLInterface

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
    p1 = CIplot(hist["steps"], hist["reward"], label="Undiscounted", c=1, title="Episodic Reward")
    CIplot!(p1, hist["steps"], hist["discounted_reward"], label="Discounted", c=2)
    p2 = CIplot(hist["steps"], hist["episode_length"]; title="Episode Length")
    plot(p1, p2; layout=(2,1))    
end


env = DiscreteLaserTagPFBeliefMDP()


@kwdef mutable struct LaserTagWrapper{T<:LaserTagBeliefMDP} <: Wrappers.AbstractWrapper
    const env::T # = DiscreteLaserTagBeliefMDP(), = ContinuousLaserTagBeliefMDP
    const max_steps::Int = 500
    steps::Int = 0
    reward_scale::Float64 = 1.
end

Wrappers.wrapped_env(wrap::LaserTagWrapper) = wrap.env

CommonRLExtensions.truncated(wrap::LaserTagWrapper) = wrap.steps[] >= wrap.max_steps

function RL.reset!(wrap::LaserTagWrapper)
    wrap.steps = 0
    reset!(wrap.env)
end

RL.observations(wrap::LaserTagWrapper{<:ExactBeliefLaserTag}) = Box(Float32, 2+reduce(*, wrap.env.size))
RL.observations(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag}) = TupleSpace(Box(Float32, 2), Box(2*length(wrap.env.state.belief_target.collection.particles)))

function RL.observe(wrap::LaserTagWrapper{<:ExactBeliefLaserTag})
    o = Float32.(observe(wrap.env))
    @assert !any(isnan, o)
    o[1:2] ./= wrap.env.size # normalizing
    @views @. o[3:end] = max(0, 1+log10(o[3:end])/2) # transform belief
    return o
end

RL.actions(::LaserTagWrapper{<:ContinuousActionLaserTag}) = TupleSpace(Box(lower=[-1f0, -1f0], upper=[1f0, 1f0]), Discrete(2))
RL.actions(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}) = Discrete(length(actions(wrap.env)))

const idx2act = Dict(i=>s for (i,s) in enumerate(keys(LaserTag.actiondir)))
RL.act!(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}, a::AbstractArray) = act!(wrap, a[])
function RL.act!(wrap::LaserTagWrapper{<:DiscreteActionLaserTag}, a::Integer)
    wrap.steps += 1
    r = act!(wrap.env, idx2act[a])
    r * convert(typeof(r), wrap.reward_scale)
end

function RL.act!(wrap::LaserTagWrapper{<:ContinuousActionLaserTag}, a::Tuple)
    wrap.steps += 1
    if a[2] == 1
        r = act!(wrap.env, :measure)
    else
        r = act!(wrap.env, a[1])
    end
    r * convert(typeof(r), wrap.reward_scale)
end

# # This code should take ~3 minutes to run (plus precompile time)
# discount = 0.99
# solver = PPOSolver(; 
#     env = RewNorm(; discount, env = LoggingWrapper(; discount, 
#         env = VecEnv(n_envs=8) do 
#             LaserTagWrapper(env=DiscreteLaserTagBeliefMDP())
#         end
#     )),
#     discount, 
#     n_steps = 500_000,
#     traj_len = 256,
#     batch_size = 256,
#     n_epochs = 4,
#     kl_targ = Inf32,
#     ent_coef = 0,
#     lr_decay = false,
#     ac_kwargs = (critic_dims=[256,256], actor_dims=[256,256])
# )
# ac, info_log = solve(solver)
# plot_LoggingWrapper(solver.env)


# # So Himanshu can see how to interact with RL policy.
# # Note: There are observation and action transforms that happen in the LasterTagWrapper!
# # To properly evaluate, *MUST* wrap environment in LaserTagWrapper.
# function evaluate(env, ac; discount=0.99, max_steps=500)
#     reset!(env)
#     steps = 0
#     r = 0.0
#     while !terminated(env) || steps < max_steps
#         o = observe(env)
#         a = ac(o)
#         r += act!(env, a) * discount ^ steps
#         steps += 1
#     end
#     return r
# end
# evaluate(LaserTagWrapper(env=DiscreteLaserTagBeliefMDP()), solver.ac) # ac is callable


# 0.05 is BAD
# 0.01 had best performance
# diff ent coeff for diff networks?
# 64 layer worked well enough, 2.5 mill should be good

discount = 0.99
solver = PPOSolver(; 
    env = RewNorm(; discount, env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(env=ContinuousLaserTagBeliefMDP())
        end
    )),
    discount, 
    n_steps = 5_000_000,
    traj_len = 128,
    batch_size = 128,
    n_epochs = 4,
    kl_targ = Inf32,
    clip_coef = 0.02,
    clipl2 = 0.5,
    ent_coef = (0f0, 0.01f0),
    vf_coef = 1.0,
    lr_decay = true,
    lr = 3e-4,
    ac_kwargs = (shared_dims=[], critic_dims=[256,256], actor_dims=[256,256], shared_actor_dims=[], squash=true)
)
ac, info_log = solve(solver)
plot_LoggingWrapper(solver.env)

bson("continuous_exact.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))


test_env = LaserTagWrapper(env=ContinuousLaserTagBeliefMDP())
ac = solver.ac
reset!(test_env)
target, robot_pos, belief_target = [], [], []
for t in 1:1000
    push!(target, copy(test_env.env.target))
    push!(robot_pos, copy(test_env.env.state.robot_pos))
    push!(belief_target, copy(test_env.env.state.belief_target))

    s = observe(test_env)
    a = ac(s)
    act!(test_env, a)
    terminated(test_env) && break
end

target
robot_pos
belief_target

f(x) = max(0, 1+log10(x)/2)
obs = stack([test_env.env.obstacles...])
common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10)
anim = Plots.@animate for i in eachindex(belief_target)
    b = f.(belief_target[i]')
    heatmap(b, c = Plots.cgrad(:roma, scale=:log), clim=(0,1))
    plot!([robot_pos[i][1]], [robot_pos[i][2]]; markershape=:circle, common...)
    plot!([target[i][1]], [target[i][2]]; markershape=:star5, common...)
    plot!(obs[1,:], obs[2,:]; markershape=:x, common...)
    plot!(; title = "max(0, 1+log10(b)/2), t = $i") #, action = $(actions(LaserTagBeliefMDP())[a_vec[i][]])")
end
gif(anim, "continuous_exact_animation.gif", fps = 1)



## PF


