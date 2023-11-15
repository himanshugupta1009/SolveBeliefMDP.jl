using SolveBeliefMDP

using RLAlgorithms.Algorithms: solve, PPOSolver, ActorCritic
using RLAlgorithms.MultiEnv: VecEnv, LoggingWrapper, RewNorm, ObsNorm
using RLAlgorithms.Environments: CartPole
using RLAlgorithms.CommonRLExtensions: get_info
using Flux, Plots, Statistics, CommonRLInterface

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
env = LoggingWrapper(; discount, env = VecEnv(()->LaserTagWrapper(); n_envs=8))
solver = PPOSolver(; 
    env,
    discount, 
    n_steps = 500_000,
    traj_len = 128,
    batch_size = 64,
    n_epochs = 4,
    kl_targ = Inf32,
    ent_coef = 0,
    lr_decay = true,
    ac = ActorCritic(env; critic_dims=[256,256], actor_dims=[256,256])
)
ac, info_log = solve(solver)

# Reward / Learning Curves
hist = get_info(solver.env)["LoggingWrapper"]
p1 = CIplot(hist["steps"], hist["reward"], label="Undiscounted", c=1, title="Episodic Reward")
CIplot!(p1, hist["steps"], hist["discounted_reward"], label="Discounted", c=2)
p2 = CIplot(hist["steps"], hist["episode_length"]; title="Episode Length")
plot(p1, p2; layout=(2,1))
savefig("LearningCurve.png")

# Debugging / Loss Curves
plot(
    [plot(val[1], val[2]; xlabel="Steps", title=key, label=false) for (key,val) in info_log]...; 
    size=(900,900)
)

ac = solver.ac
test_env = LaserTagWrapper()
reset!(test_env)
target, robot_pos, belief_target, a_vec = [], [], [], []
for t in 1:1000
    push!(target, copy(test_env.env.target))
    push!(robot_pos, copy(test_env.env.state.robot_pos))
    push!(belief_target, copy(test_env.env.state.belief_target))
    push!(a_vec, copy(a))

    s = observe(test_env)
    a = ac(s)
    act!(test_env, a)
    terminated(test_env) && break
end
length(a_vec)


b = f.(belief_target[9]')

minimum(belief_target[9][belief_target[9] .> 0])

(2+log10(0.01))/3

0.0056*70

log10(1/70)

1 + log10(0.1)/2

belief_target[10]

0.5/70

belief_target[1]


f(x) = max(-1, 1+log10(x)*2/3)
obs = stack([test_env.env.obstacles...])
common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10)
anim = Plots.@animate for i in eachindex(belief_target)
    b = f.(belief_target[i]')
    heatmap(b, c = Plots.cgrad(:roma, scale=:log), clim=(-1,1))
    plot!([robot_pos[i][1]], [robot_pos[i][2]]; markershape=:circle, common...)
    plot!([target[i][1]], [target[i][2]]; markershape=:star5, common...)
    plot!(obs[1,:], obs[2,:]; markershape=:x, common...)
    plot!(; title = "t = $i, action = $(actions(LaserTagBeliefMDP())[a_vec[i][]])")
end
gif(anim, "animation.gif", fps = 1)
