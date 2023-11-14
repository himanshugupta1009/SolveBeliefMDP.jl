include("LaserTag.jl")

using RLAlgorithms.Spaces: Discrete, Box
using RLAlgorithms.CommonRLExtensions

using CommonRLInterface.Wrappers
using Parameters

@with_kw struct LaserTagWrapper{T<:LaserTagBeliefMDP} <: Wrappers.AbstractWrapper
    env::T = LaserTagBeliefMDP()
    steps::MVector{1,Int} = @MVector [0]
end

Wrappers.wrapped_env(wrap::LaserTagWrapper) = wrap.env

function RL.observe(wrap::LaserTagWrapper)
    o = Float32.(observe(wrap.env))
    o[1:2] ./= wrap.env.size
    return o
end

RL.act!(wrap::LaserTagWrapper, a::Integer) = act!(wrap.env, actions(wrap.env)[a])
function RL.act!(wrap::LaserTagWrapper, a::AbstractArray)
    wrap.steps[] += 1
    RL.act!(wrap, a[])
end

function RL.reset!(wrap::LaserTagWrapper)
    wrap.steps[] = 0
    reset!(wrap.env)
end

RL.actions(wrap::LaserTagWrapper) = Discrete(length(actions(wrap.env)))
RL.observations(::LaserTagWrapper) = Box(fill(-Inf32,72), fill(Inf32,72))
CommonRLExtensions.truncated(wrap::LaserTagWrapper) = wrap.steps[] >= 500

using RLAlgorithms.Algorithms: solve, PPOSolver
using RLAlgorithms.MultiEnv: VecEnv, LoggingWrapper, RewNorm, ObsNorm
using RLAlgorithms.Environments: CartPole
using RLAlgorithms.CommonRLExtensions: get_info
using Flux
using Plots
using Statistics

# Plotting confidence intervals
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

discount = 0.99

solver = PPOSolver(; 
    env = RewNorm(; discount, env=env = LoggingWrapper(; discount, env = VecEnv(()->LaserTagWrapper(); n_envs=4))), 
    discount, 
    n_steps = 500_000,
    traj_len = 128,
    batch_size = 128,
    n_epochs = 4,
    kl_targ = Inf32,
    ent_coef = 0.01
)

ac, info_log = solve(solver)

# Reward / Learning Curves
hist = get_info(solver.env)["LoggingWrapper"]
p1 = CIplot(hist["steps"], hist["reward"], label="Undiscounted", c=1, title="Episodic Reward")
CIplot!(p1, hist["steps"], hist["discounted_reward"], label="Discounted", c=2)
p2 = CIplot(hist["steps"], hist["episode_length"]; title="Episode Length")
plot(p1, p2; layout=(2,1))

# Reward stats blow up, probably not ideal
# How to fix?
solver.env.rew_stats

# Debugging / Loss Curves
plot(
    [plot(val[1], val[2]; xlabel="Steps", title=key, label=false) for (key,val) in info_log]...; 
    size=(900,900)
)

test_env = LaserTagWrapper()
reset!(test_env)
s_vec, a_vec, r_vec = [], [], []
for t in 1:1000
    s = observe(test_env)
    a = ac(s)
    r = act!(test_env, a)
    push!.((s_vec, a_vec, r_vec), (s, a, r))
    if terminated(test_env)
        println()
        break
    end
end
r_vec
sum(r_vec)

mean((s_vec |> stack)[3:end,:], dims=2)

o = reshape(s_vec[2][3:end], 10, 7)
