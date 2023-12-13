include("new_solver.jl")

function discrete_exact_animation(ac)
    test_env = LaserTagWrapper(env=DiscreteLaserTagBeliefMDP())
    reset!(test_env)
    target, robot_pos, belief_target = [], [], []
    a_d = []
    for t in 1:500
        push!(target, copy(test_env.env.target))
        push!(robot_pos, copy(test_env.env.state.robot_pos))
        push!(belief_target, copy(test_env.env.state.belief_target))

        s = observe(test_env)
        a = ac(s)
        # push!.((a_c,a_d), a)
        push!(a_d, a)
        act!(test_env, a)
        terminated(test_env) && break
    end

    f(x) = max(0, 1+log10(x)/3)
    obs = stack([test_env.env.obstacles...])
    common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10, xticks=1:10, ticks=1:7)
    anim = Plots.@animate for i in eachindex(belief_target)
        b = f.(belief_target[i])
        heatmap(0.5 .+ (1:10), 0.5 .+ (1:7), b'; c = Plots.cgrad(:roma, scale=:log), clim=(0,1))
        plot!([0.5+robot_pos[i][1]], [0.5+robot_pos[i][2]]; markershape=:circle, common...)
        plot!([0.5+target[i][1]], [0.5+target[i][2]]; markershape=:star5, common...)
        plot!(0.5 .+ obs[1,:], 0.5 .+ obs[2,:]; markershape=:x, common...)
        a = idx2act[a_d[i][]]
        plot!(; title = "a = $a, t = $i") #, action = $(actions(LaserTagBeliefMDP())[a_vec[i][]])")
    end
    gif(anim, fps = 1)
end

function RL.observations(wrap::LaserTagWrapper{<:ExactBeliefLaserTag})
    s = wrap.env.state
    o1 = Box(Float32, size(s.robot_pos))
    # o2 = Box(Float32, size(s.belief_target))
    # TupleSpace(o1, o2)
end
function RL.observe(wrap::LaserTagWrapper{<:ExactBeliefLaserTag})
    s = wrap.env.state
    pos = convert(AbstractArray{Float32}, s.robot_pos)
    # belief = convert(AbstractArray{Float32}, s.belief_target)
    o1 = pos ./ wrap.env.size
    # o2 = @. max(0, 1+log10(belief)/3)
    # return (o1, o2)
end

# This code should take ~3 minutes to run (plus precompile time)
discount = 0.997

env = LaserTagWrapper(env=DiscreteLaserTagBeliefMDP(), reward_scale=1., max_steps=500)
function test_rand(env)
    reset!(env)
    steps = 0
    while !terminated(env)
        a = rand(1:9, (1,1))
        act!(env, a)
        steps += 1
    end
    steps
end
env = LaserTagWrapper(env=DiscreteLaserTagBeliefMDP(), reward_scale=1., max_steps=500)
mean(test_rand(env) for _ in 1:100)

env = LaserTagWrapper(env=DiscreteLaserTagBeliefMDP(size=(15, 15), n_obstacles=30), reward_scale=1., max_steps=500)
obs = stack([env.env.obstacles...])
plot(0.5 .+ obs[1,:], 0.5 .+ obs[2,:]; 
markershape=:x, label=false, seriestype=:scatter, markercolor=:black, markersize=10, 
xticks=1:16, yticks=1:16, xlims=(1,16), ylims=(1,16))
mean(test_rand(env) for _ in 1:100)

solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(env=DiscreteLaserTagBeliefMDP(size=(15, 15), n_obstacles=30), reward_scale=1., max_steps=1_000)
        end
    ),
    discount, 
    n_steps = 5_000_000,
    traj_len = 512,
    batch_size = 256,
    n_epochs = 4,
    kl_targ = 0.02,
    clipl2 = Inf32,
    ent_coef = 0.01f0, 
    lr_decay = true,
    lr = 1e-3,
    vf_coef = 1.0,
    gae_lambda = 0.95,
    burnin_steps = 0,
    ac_kwargs = (
        critic_dims = [64,64], 
        actor_dims  = [64,64], 
        critic_type = (:scalar, :categorical)[1], 
        categorical_values = range(symlog(-2/(1-discount)), symlog(100), 400),
        critic_loss_transform = symlog,
        inv_critic_loss_transform = symexp,
        # shared = Parallel(
        #     vcat,
        #     identity,
        #     Chain(
        #         x->reshape(x,10,7,:),
        #         Flux.flatten
        #     )
        # )
    )
)
ac, info_log = solve(solver)

x = range(0, 5_000_000, 500)
y = get_mean(solver.env, x)
plot(x,y; label=false)




# Notes:
#   gae = 0.95 and 0.9 are very similar, outperforms 0.99 and 0.5

p = plot(; 
    xlabel="Steps", title="Average Discounted Reward",
    xlims = (0, 1_000_000),
    ylims = (-150, 100),
    yticks = -150:25:100
)

for (c,gae) in enumerate([1.0, 0.99, 0.95, 0.9, 0.5, 0.0])
    solver_vec = PPOSolver[]
    for _ in 1:2
        solver = PPOSolver(; 
            env = LoggingWrapper(; discount, 
                env = VecEnv(n_envs=8) do 
                    LaserTagWrapper(env=DiscreteLaserTagBeliefMDP(), reward_scale=1., max_steps=500)
                end
            ),
            discount, 
            n_steps = 1_000_000,
            traj_len = 512,
            batch_size = 256,
            n_epochs = 4,
            kl_targ = 0.02,
            clipl2 = Inf32,
            ent_coef = 0.01f0, 
            lr_decay = true,
            lr = 1e-3,
            vf_coef = 1.0,
            gae_lambda = gae,
            burnin_steps = 0,
            ac_kwargs = (
                critic_dims = [64,64], 
                actor_dims  = [64,64], 
                critic_type = (:scalar, :categorical)[1], 
                categorical_values = range(symlog(-2/(1-discount)), symlog(100), 400),
                critic_loss_transform = symlog,
                inv_critic_loss_transform = symexp,
                shared = Parallel(
                    vcat,
                    identity,
                    Chain(
                        x->reshape(x,10,7,:),
                        Flux.flatten
                    )
                )
            )
        )
        push!(solver_vec, solver)
        ac, info_log = solve(solver)
    end

    plot_seed_ci!(p, solver_vec; xmax=1_000_000, label=nothing, c)
    display(p)
end

"lambda = " .* string.([1.0 0.99 0.95 0.9 0.5 0.0])
p2 = plot(p, label = "lambda = " .* string.([1.0 0.99 0.95 0.9 0.5 0.0]), legend=:bottomright)


plot(p[1][2])

plotattr(:Series)

fieldnames(typeof(p))
for i in 2:2:12
    p[1][i].plotattributes[:label] = "lambda = " * string([1.0 0.99 0.95 0.9 0.5 0.0][i ÷ 2])
end
plot(p, right_margin = 0.5Plots.cm)
savefig("gae_lambda_discrete_exact.png")





p = plot(title="Episodic Reward")
for (c,solver) in enumerate(solver_vec)
    hist = get_info(solver.env)["LoggingWrapper"]
    CIplot!(p, hist["steps"], hist["discounted_reward"]; label=false, c)
end
p



plot_LoggingWrapper(solver.env)

plot!(ylims=(50,100))

discrete_exact_animation(solver.ac)

bson("solver/discrete_exact.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))



