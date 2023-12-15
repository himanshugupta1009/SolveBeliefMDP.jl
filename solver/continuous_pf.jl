include("new_solver.jl")

solver_vec = PPOSolver[]
for _ in 1:3
    solver = PPOSolver(; 
        env = LoggingWrapper(; discount, 
            env = VecEnv(n_envs=32) do 
                LaserTagWrapper(env=ContinuousLaserTagPFBeliefMDP(num_particles=100), reward_scale=1., max_steps=2_000)
            end
        ),
        discount, 
        n_steps = 5_000_000,
        traj_len = 512,
        batch_size = 512,
        n_epochs = 4,
        kl_targ = 0.02,
        clipl2 = 2.,
        ent_coef = 0.02f0, 
        lr_decay = true,
        lr = 3e-4,
        vf_coef = 1.0,
        gae_lambda = 0.95,
        burnin_steps = 0,
        ac_kwargs = (
            critic_dims = [64,64], 
            actor_dims  = [64,64], 
            critic_type = :scalar, 
            critic_loss_transform = x -> (x .- 50) ./ 50,
            inv_critic_loss_transform = x -> x .* 50 .+ 50,
            shared = Parallel(
                vcat,
                identity,
                MGF(init=randn(Float32, 64, 2)/10)
            ),
            squash = true,
            sde = true    
        )
    )
    push!(solver_vec, solver)
    ac, info_log = solve(solver)
end
BSON.bson("solver/data/continuous_pf_final.bson", Dict(:solver_vec=>solver_vec))



plot_LoggingWrapper(solver.env)
savefig("solver/data/continuous_pf.png")

solver.ac.actor.actors[1].log_std

plot(info_log[:value_loss]...; label=false)

bson("solver/continuous_pf.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))


