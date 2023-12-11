include("new_solver.jl")

discount = 0.997
solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(env=ContinuousLaserTagPFBeliefMDP(), reward_scale=1., max_steps=1000)
        end
    ),
    discount, 
    n_steps = 10_000_000,
    traj_len = 512,
    batch_size = 256,
    n_epochs = 4,
    kl_targ = 0.02,
    clipl2 = Inf32,
    ent_coef = (0.01f0, 0.01f0),
    lr_decay = true,
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
            CGF(init=randn(Float32,64,2)/2)
        ),
        squash = true
    )
)
ac, info_log = solve(solver)


plot_LoggingWrapper(solver.env)
savefig("solver/data/continuous_pf.png")

solver.ac.actor.actors[1].log_std

plot(info_log[:value_loss]...; label=false)

bson("solver/continuous_pf.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))


