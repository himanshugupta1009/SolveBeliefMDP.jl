include("new_solver.jl")

function continuous_exact_animation(ac)
    test_env = LaserTagWrapper(env=ContinuousLaserTagBeliefMDP())
    reset!(test_env)
    target, robot_pos, belief_target = [], [], []
    a_c, a_d = [], []
    for t in 1:500
        push!(target, copy(test_env.env.target))
        push!(robot_pos, copy(test_env.env.state.robot_pos))
        push!(belief_target, copy(test_env.env.state.belief_target))

        s = observe(test_env)
        a = ac(s)
        push!.((a_c,a_d), a)
        act!(test_env, a)
        terminated(test_env) && break
    end

    f(x) = max(0, 1+log10(x)/3)
    obs = stack([test_env.env.obstacles...])
    common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10, xticks=1:10, ticks=1:7)
    anim = Plots.@animate for i in eachindex(belief_target)
        b = f.(belief_target[i])
        heatmap(0.5 .+ (1:10), 0.5 .+ (1:7), b'; c = Plots.cgrad(:roma, scale=:log), clim=(0,1))
        plot!([robot_pos[i][1]], [robot_pos[i][2]]; markershape=:circle, common...)
        plot!([0.5+target[i][1]], [0.5+target[i][2]]; markershape=:star5, common...)
        plot!(0.5 .+ obs[1,:], 0.5 .+ obs[2,:]; markershape=:x, common...)
        a = (a_d[i][] == 1) ? "measure" : round.(Float64.(vec(a_c[i])), digits=2)
        plot!(; title = "a = $a, t = $i") #, action = $(actions(LaserTagBeliefMDP())[a_vec[i][]])")
    end
    gif(anim, fps = 1)
end

# This code should take ~3 minutes to run (plus precompile time)
discount = 0.997

p1 = plot(; 
    xlabel="Steps", title="Average Discounted Reward",
    xlims = (0, 1_000_000),
    ylims = (-150, 100),
    yticks = -150:25:100
)
solver_vec1 = PPOSolver[]
for _ in 1:2
    solver = PPOSolver(; 
        env = LoggingWrapper(; discount, 
            env = VecEnv(n_envs=8) do 
                LaserTagWrapper(env=ContinuousLaserTagBeliefMDP(), reward_scale=1., max_steps=500)
            end
        ),
        discount, 
        n_steps = 1_000_000,
        traj_len = 512,
        batch_size = 256,
        n_epochs = 4,
        kl_targ = 0.02,
        clipl2 = Inf32,
        ent_coef = (0.01f0, 0.01f0),
        lr_decay = true,
        lr = 1e-3,
        vf_coef = 1.0,
        gae_lambda = 0.95,
        burnin_steps = 0,
        ac_kwargs = (
            critic_dims = [256,256], 
            actor_dims  = [], 
            shared_actor_dims  = [64,64], 
            critic_type = (:scalar, :categorical)[1], 
            categorical_values = range(symlog(-2/(1-discount)), symlog(100), 300),
            critic_loss_transform = symlog,
            inv_critic_loss_transform = symexp,
            shared = Parallel(
                vcat,
                identity,
                Chain(
                    x->reshape(x,10,7,:),
                    Flux.flatten
                )# CGF(init=randn(Float32,64,2)/2)
            ),
            squash = true
        )
    )
    push!(solver_vec1, solver)
    ac, info_log = solve(solver)
end
plot_seed_ci!(p1, solver_vec1; xmax=1_000_000, label=nothing, c=1)



discount = 0.997

p2 = plot(; 
    xlabel="Steps", title="Average Discounted Reward",
    xlims = (0, 1_000_000),
    ylims = (-150, 100),
    yticks = -150:25:100
)
solver_vec2 = PPOSolver[]
for _ in 1:5
    solver = PPOSolver(; 
        env = LoggingWrapper(; discount, 
            env = VecEnv(n_envs=8) do 
                LaserTagWrapper(env=ContinuousLaserTagBeliefMDP(), reward_scale=1., max_steps=1000)
            end
        ),
        discount, 
        n_steps = 5_000_000,
        traj_len = 512,
        batch_size = 256,
        n_epochs = 4,
        kl_targ = 0.02,
        clipl2 = Inf32,
        ent_coef = (0.01f0, 0.01f0),
        lr_decay = true,
        lr = 1e-3,
        vf_coef = 1.0,
        gae_lambda = 0.95,
        burnin_steps = 0,
        ac_kwargs = (
            critic_dims = [64,64], 
            actor_dims  = [], 
            shared_actor_dims  = [64,64], 
            critic_type = (:scalar, :categorical)[1], 
            categorical_values = range(symlog(-2/(1-discount)), symlog(100), 300),
            critic_loss_transform = symlog,
            inv_critic_loss_transform = symexp,
            shared = Parallel(
                vcat,
                identity,
                Chain(
                    x->reshape(x,10,7,:),
                    Flux.flatten
                )# CGF(init=randn(Float32,64,2)/2)
            ),
            squash = true
        )
    )
    push!(solver_vec2, solver)
    ac, info_log = solve(solver)
end
plot_seed_ci!(p2, solver_vec2; xmax=5_000_000, label=nothing, c)








plot_LoggingWrapper(solver.env)
savefig("solver/continuous_exact.png")

solver.ac.actor.actors[1].log_std

plot(info_log[:value_loss]...; label=false)

bson("solver/continuous_exact_10.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))


results = [evaluate(LaserTagWrapper(env=ContinuousLaserTagBeliefMDP(), reward_scale=1., max_steps=1000), solver.ac) for _ in 1:1000]
vals = mean(results)
errs = std(results)/sqrt(length(results))


data = BSON.load("solver/continuous_exact_10.bson")


continuous_exact_animation(data[:ac])


