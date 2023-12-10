include("new_solver.jl")

function discrete_pf_animation(ac; num_particles=100)
    test_env = LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=num_particles))
    reset!(test_env)
    target, robot_pos, particles = [], [], []
    a_d = []
    for t in 1:500
        push!(target, copy(test_env.env.target))
        push!(robot_pos, copy(test_env.env.state.robot_pos))
        push!(particles, copy(test_env.env.state.belief_target.collection.particles))

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
    anim = Plots.@animate for i in eachindex(particles)
        grid = zeros(10,7)
        for p in particles[i]
            grid[p...] += 1
        end
        grid ./= length(particles[i])
        
        heatmap(0.5 .+ (1:10), 0.5 .+ (1:7), f.(grid)'; c = Plots.cgrad(:roma, scale=:log), clim=(0,1))
        
        plot!([0.5+robot_pos[i][1]], [0.5+robot_pos[i][2]]; markershape=:circle, common...)
        plot!([0.5+target[i][1]], [0.5+target[i][2]]; markershape=:star5, common...)
        plot!(0.5 .+ obs[1,:], 0.5 .+ obs[2,:]; markershape=:x, common...)
        a = idx2act[a_d[i][]]
        plot!(; title = "a = $a, t = $i") #, action = $(actions(LaserTagBeliefMDP())[a_vec[i][]])")
    end
    gif(anim, fps = 1)
end

# This code should take ~3 minutes to run (plus precompile time)
discount = 0.997
solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=100), reward_scale=1., max_steps=500)
        end
    ),
    discount, 
    n_steps = 1_000_000,
    traj_len = 256,
    batch_size = 256,
    n_epochs = 4,
    kl_targ = 0.02,
    clipl2 = Inf32,
    ent_coef = 0f0, #(0.0f0, 0.01f0),
    lr_decay = false,
    vf_coef = 1.0,
    gae_lambda = 0.95,
    burnin_steps = 50_000,
    ac_kwargs = (
        critic_dims = [64,64], 
        actor_dims  = [64,64], 
        critic_type = (:scalar, :categorical)[1], 
        categorical_values = range(symlog(-2/(1-discount)), symlog(100), 200),
        critic_loss_transform = symlog,
        inv_critic_loss_transform = symexp,
        shared = Parallel(
            vcat,
            identity,
            CGF(init=randn(Float32,64,2)/2)
        ),
        shared_out_size = 4 + 64,
    )
)
ac, info_log = solve(solver)

plot_LoggingWrapper(solver.env)

discrete_pf_animation(solver.ac; num_particles=100)

bson("solver/discrete_exact.bson", Dict(:ac=>solver.ac, :env=>solver.env, :info=>info_log))


n_particles = [5, 10, 50, 100, 500, 1000]
vals = zeros(length(n_particles))
errs = zeros(length(n_particles))
for i in 1:length(n_particles)
    println(i)
    test_env = LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=n_particles[i]))
    results = [evaluate(test_env, solver.ac) for _ in 1:1000]
    vals[i] = mean(results)
    errs[i] = std(results)/sqrt(length(results))
end

plot(
    n_particles,vals; 
    yerror=errs, xaxis=:log, label=false, xlabel="Number of Particles", ylabel="Returns", xticks=(n_particles, string.(n_particles))
)
savefig("particles.png")



w = solver.ac.shared[2].weight
plot(w[:,1]', w[:,2]'; seriestype=:scatter, markercolor=:blue, markershape=:circle, markersize=2, label=false, xlims=(-2,2), ylims=(-2,2))

mean(w; dims=1)


using LinearAlgebra
eigvals(cov(w))

x = randn(Float32,64,2)/2
plot!(x[:,1]', x[:,2]'; seriestype=:scatter, markercolor=:red, markershape=:circle, markersize=2, label=false, xlims=(-2,2), ylims=(-2,2))


#####
## Mean and covariance
#####

function RL.observations(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag})
    s = wrap.env.state
    Box(Float32, 3*length(s.robot_pos))
end
function RL.observe(wrap::LaserTagWrapper{<:ParticleBeliefLaserTag})
    s = wrap.env.state
    pos = convert(AbstractArray{Float32}, s.robot_pos)
    belief = convert(AbstractArray{Float32}, stack(s.belief_target.collection.particles))
    mu = mean(belief; dims=2)
    sigma = std(belief; dims=2)
    o = [
        vec(pos) ./ wrap.env.size; 
        vec(mu) ./ wrap.env.size;
        vec(sigma) ./ wrap.env.size] 
    return o
end

# This code should take ~3 minutes to run (plus precompile time)
discount = 0.997
solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=100), reward_scale=1., max_steps=500)
        end
    ),
    discount, 
    n_steps = 1_000_000,
    traj_len = 256,
    batch_size = 256,
    n_epochs = 4,
    kl_targ = 0.02,
    clipl2 = Inf32,
    ent_coef = 0f0, #(0.0f0, 0.01f0),
    lr_decay = false,
    vf_coef = 1.0,
    gae_lambda = 0.95,
    burnin_steps = 50_000,
    ac_kwargs = (
        critic_dims = [64,64], 
        actor_dims  = [64,64], 
        critic_type = (:scalar, :categorical)[1], 
        categorical_values = range(symlog(-2/(1-discount)), symlog(100), 200),
        critic_loss_transform = symlog,
        inv_critic_loss_transform = symexp,
    )
)
ac, info_log = solve(solver)

plot_LoggingWrapper(solver.env)

discrete_pf_animation(solver.ac; num_particles=100)

n_particles = [5, 10, 50, 100, 500, 1000]
vals = zeros(length(n_particles))
errs = zeros(length(n_particles))
for i in 1:length(n_particles)
    println(i)
    test_env = LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=n_particles[i]))
    results = [evaluate(test_env, solver.ac) for _ in 1:1000]
    vals[i] = mean(results)
    errs[i] = std(results)/sqrt(length(results))
end
plot(
    n_particles,vals; 
    yerror=errs, xaxis=:log, label=false, xlabel="Number of Particles", ylabel="Returns", xticks=(n_particles, string.(n_particles))
)
savefig("particles.png")
