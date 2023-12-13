include("new_solver.jl")

function discrete_pf_animation(ac, test_env = LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=num_particles)); num_particles=100)
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
    common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10, xticks=1:10, yticks=1:7)
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
        plot!(; title = "a = $a, t = $i") 
    end
    gif(anim, fps = 1)
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


solver_vec = PPOSolver[]
discount = 0.997
for fun in [MGF, CGF], mgf_scale in [0.01f0, 0.1f0, 1f0]
    solver = PPOSolver(; 
        env = LoggingWrapper(; discount, 
            env = VecEnv(n_envs=8) do 
                LaserTagWrapper(
                    env=DiscreteLaserTagPFBeliefMDP(num_particles=100), 
                    reward_scale=1., max_steps=500
                )
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
        lr = 10e-4,
        vf_coef = 0.5,
        gae_lambda = 0.95,
        burnin_steps = 0,
        ac_kwargs = (
            critic_dims = [64,64], 
            actor_dims  = [64,64], 
            critic_type = :scalar, 
            critic_loss_transform = symlog,
            inv_critic_loss_transform = symexp,
            shared = Parallel(
                    vcat,
                    identity,
                    fun(init=mgf_scale*randn(Float32, 64, 2))
                ),
        )
    )
    push!(solver_vec, solver)
    ac, info_log = solve(solver)

    bson("solver/data/mgf_cgf.bson", Dict(:solver_vec=>solver_vec))
end

p = plot()
for (solver, label) in zip(solver_vec, ["MGF, sigma=0.01", "MGF, sigma=0.1", "MGF, sigma=1.0", "CGF, sigma=0.01", "CGF, sigma=0.1", "CGF, sigma=1.0"])
    x = range(0, 5_000_000, 500)
    y = get_mean(solver.env, x)
    plot!(p, x,y; label)
end
plot(p; ylims=(-150,100), yticks=-150:25:100, xlims=(0,5e6), title="MGF and CGF over Varrying Initial Covariance", right_margin = 0.5Plots.cm)
savefig("mgf_cgf_5m.png")

for (solver, label) in zip(solver_vec, ["MGF, 0.01", "MGF, 0.1", "MGF, 1.0", "CGF, 0.01", "CGF, 0.1", "CGF, 1.0"])
    w = solver.ac.shared[2].weight
    sqrt.(eigvals(cov(w))) |> display
end

for (solver, label) in zip(solver_vec, ["MGF, 0.01", "MGF, 0.1", "MGF, 1.0", "CGF, 0.01", "CGF, 0.1", "CGF, 1.0"])
    w = solver.ac.shared[2].weight
    mean(w; dims=1) |> display
end

mapreduce(solver->sqrt.(eigvals(cov(solver.ac.shared[2].weight))), hcat, solver_vec)


using LinearAlgebra
eigvals

plot(x,y)

w = solver_vec[1].ac.shared[2].weight
plot(w[:,1], w[:,2]; seriestype=:scatter, markercolor=:blue, markershape=:circle, markersize=2, label="weights")
xy = sqrt(cov(w)) *( [x'; y'] ) .+ mean(w; dims=1)'
plot!(xy[1,:], xy[2,:]; label="95% confidence")

p = plot()
for (solver, label, c) in zip(solver_vec, ["MGF, sigma=0.01", "MGF, sigma=0.1", "MGF, sigma=1.0", "CGF, sigma=0.01", "CGF, sigma=0.1", "CGF, sigma=1.0"], [1,2,3,4,5,6])
    x = cos.(range(0,2pi,100)) * sqrt(5.991)
    y = sin.(range(0,2pi,100)) * sqrt(5.991)
    w = solver.ac.shared[2].weight
    plot!(p, w[:,1], w[:,2]; seriestype=:scatter, markercolor=c, markershape=:circle, markersize=2, label=label, c)
    xy = sqrt(cov(w)) *( [x'; y'] ) .+ mean(w; dims=1)'
    plot!(p, xy[1,:], xy[2,:]; label=false, c)
end
plot(p; legend=:outerright, title="Weights After 5x10^6 Steps")
savefig("mgf_cgf_weight.png")


 
w = ac.shared[2].weight

plot(xlims=(-2,2), ylims=(-2,2))
plot!(w[:,1]', w[:,2]'; seriestype=:scatter, markercolor=:blue, markershape=:circle, markersize=2, label=false,)
plot!(init[:,1]', init[:,2]'; seriestype=:scatter, markercolor=:red, markershape=:circle, markersize=2, label=false,)




ac = solver_vec[1].ac

n_particles = [5, 10, 50, 100, 500, 1000]
vals = zeros(length(n_particles))
errs = zeros(length(n_particles))
for i in 1:length(n_particles)
    println(i)
    test_env = LaserTagWrapper(env=DiscreteLaserTagPFBeliefMDP(num_particles=n_particles[i]))
    results = [evaluate(test_env, ac) for _ in 1:1000]
    vals[i] = mean(results)
    errs[i] = std(results)/sqrt(length(results))
end

plot(
    n_particles,vals; 
    title="Test Time Accuracy of Agent Trained on 100 Particles",
    yerror=errs, xaxis=:log, label=false, xlabel="Number of Particles at Test time", ylabel="Returns", 
    xticks=(n_particles, string.(n_particles)), xlims=(3,2_000),
    ylims=(-50,100), yticks=-50:25:100,
    right_margin = 0.5Plots.cm
)
savefig("mgf_particles_test_comparison.png")



## Test runtime
discount = 0.997
solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(
                env=DiscreteLaserTagPFBeliefMDP(num_particles=100), 
                reward_scale=1., max_steps=500
            )
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
    lr = 10e-4,
    vf_coef = 0.5,
    gae_lambda = 0.95,
    burnin_steps = 0,
    ac_kwargs = (
        critic_dims = [64,64], 
        actor_dims  = [64,64], 
        critic_type = :scalar, 
        critic_loss_transform = symlog,
        inv_critic_loss_transform = symexp,
        shared = (x) -> x[1],

    )
)
ac, info_log = solve(solver)

x = range(0, 5_000_000, 500)
y = get_mean(solver.env, x)
plot(x,y; 
    label=false, ylims=(-150,100), yticks=-150:25:100, xlims=(0,5e6), right_margin = 0.5Plots.cm,
    title="Returns with No Target Belief", xlabel="Steps"
    )
savefig("solver/figures/no_belief.png")


bson("solver/data/no_belief.bson", Dict(:solver=>solver, :ac=>ac, :info_log=>info_log))

# train with This


env = LoggingWrapper(; discount, 
    env = VecEnv(n_envs=8) do 
        LaserTagWrapper(
            env=DiscreteLaserTagPFBeliefMDP(num_particles=100), 
            reward_scale=1., max_steps=500
        )
    end
)

test_env = LaserTagWrapper(
    env=DiscreteLaserTagPFBeliefMDP(num_particles=100), 
    reward_scale=1., max_steps=500
)

ac_old = solver.ac
ac_new = ActorCritic(env; 
    critic_dims = [64,64], 
    actor_dims  = [64,64], 
    critic_type = :scalar, 
    critic_loss_transform = symlog,
    inv_critic_loss_transform = symexp,
    shared = Parallel(
            vcat,
            identity,
            MGF(init=0.1f0*randn(Float32, 64, 2))
        ),
)

reset!(env)
opt = Flux.setup(solver.opt_0, ac_new)
buffer = Algorithms.Buffer(env, 512)

results = Float64[]

for _ in 1:length(results)
    Algorithms.fill_buffer!(env, buffer, ac_old)

    for idxs in Iterators.partition(randperm(length(buffer)), 256)
        function temp_fun(x)
            y = reshape(x, size(x)[1:end-2]..., :) # flatten
            copy(selectdim(y, ndims(y), idxs)) # copy important!        
        end
        temp_fun(x::Tuple) = temp_fun.(x)

        ac_input = Algorithms.ACInput(observation = temp_fun(buffer.s))

        old_actor_out, old_critic_out = get_actionvalue(ac_old, Algorithms.ACInput(; observation = temp_fun(buffer.s)))
        target_action = Flux.onehotbatch(old_actor_out.action, 1:9)
        critic_targets = old_critic_out.critic_out


        grads = Flux.gradient(ac_new) do ac_new
            shared_out = Algorithms.get_shared(ac_new.shared, ac_input.observation)

            actor_out = ac_new.actor.actor(shared_out)
            probs = softmax(actor_out; dims=1)
            actor_loss = Flux.logitcrossentropy(actor_out, target_action)

            actor_out, critic_out = get_actionvalue(ac_new, ac_input)
            value_loss = 0.5f0 * Algorithms.get_criticloss(ac_new.critic, critic_out, critic_targets)   
            return value_loss + actor_loss
        end

        Flux.update!(opt, ac_new, grads[1])
    end

    push!(results, mean(evaluate(test_env, ac_new) for _ in 1:100))
end

plot(results)
# 200 epochs is good

ac = deepcopy(ac_new)

solver = PPOSolver(; 
    env = LoggingWrapper(; discount, 
        env = VecEnv(n_envs=8) do 
            LaserTagWrapper(
                env=DiscreteLaserTagPFBeliefMDP(num_particles=100), 
                reward_scale=1., max_steps=500
            )
        end
    ),
    discount, 
    n_steps = 500_000,
    traj_len = 512,
    batch_size = 256,
    n_epochs = 4,
    kl_targ = 0.02,
    clipl2 = Inf32,
    ent_coef = 0.01f0,
    lr_decay = false,
    lr = 4e-4,
    vf_coef = 0.5,
    gae_lambda = 0.95,
    burnin_steps = 0,
    ac = ac
)
ac, info_log = solve(solver)


x = range(0, 500_000, 200)
y = get_mean(solver.env, x)
plot(x,y)

results[end]