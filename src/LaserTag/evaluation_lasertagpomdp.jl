include("LaserTagModule.jl")
using .LaserTag

using StaticArrays
using POMDPs
using ProgressMeter
using Random
using Statistics

#=
Case 1 (Discrete and Exact) - QMDP evaluation
=#
using QMDP
import POMDPSimulators:RolloutSimulator, stepthrough

d = DiscreteLaserTagPOMDP();
solver = QMDPSolver(max_iterations=20,belres=1e-3,verbose=true)
qmdp_policy = solve(solver, d);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 100
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
exact_up = updater(qmdp_policy); #Same as up = DiscreteUpdater(d)
qmdp_pomdp = d;
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, qmdp_pomdp, qmdp_policy, exact_up)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )

#=
Case 2 (Discrete and Particle Filter) - QMDP evaluation
=#
using QMDP
import POMDPSimulators:RolloutSimulator,stepthrough
import BeliefUpdaters:DiscreteUpdater

d = DiscreteLaserTagPOMDP();
solver = QMDPSolver(max_iterations=20,belres=1e-3,verbose=true)
qmdp_policy = solve(solver, d);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 100
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
qmdp_pomdp = d;
pf_up = BootstrapFilter(qmdp_pomdp, 100, MersenneTwister(abs(rand(Int8))));
function evaluate(pomdp,policy,updater,M)
    rsum = 0.0
    for (b,s,a,o,r) in stepthrough(pomdp, policy, updater, "b,s,a,o,r", max_steps=M)
       # println("belief is :", b.particles);println("action: ",a, " observation: ", o)
       rsum+=r
       if( all( i->isterminal(pomdp,i), b.particles) )
           println("HG")
           break
       end
    end
    return rsum
end
@showprogress for i in 1:n_episodes
    returns[i] = evaluate(qmdp_pomdp, qmdp_policy, pf_up, max_steps)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )

#=
Case 3 (Continuous and Exact) and
Case 4 (Continuous and Particle Filter)
are not possible to solve with QMDP
=#


#=
Case 1 (Discrete and Exact) - ARDESPOT evaluation
64.96516429394714 (+/-) 5.566928834587658
=#

using ARDESPOT
using ParticleFilters
import POMDPSimulators:RolloutSimulator,HistoryRecorder
import BeliefUpdaters:DiscreteUpdater

d = DiscreteLaserTagPOMDP();
lower_discrete = DefaultPolicyLB(solve(QMDPSolver(max_iterations=30,belres=1e-3,verbose=true), d));
function upper_discrete(m, b)
    dist = minimum(sum(abs, s.robot-s.target) for s in particles(b))
    closing_steps = div(dist, 2)
    if closing_steps > 1
        return -sum(1*discount(m)^t for t in 0:closing_steps-2) + 110.0*discount(m)^(closing_steps-1)
    else
        return 110.0
    end
end
solver_discrete = DESPOTSolver(
    bounds = IndependentBounds(lower_discrete, upper_discrete, check_terminal=true, consistency_fix_thresh=0.1),
    K = 100,
    T_max = 0.5,
    # max_trials=20,
    tree_in_info=true,
    default_action = :measure
);
despot_planner_discrete = solve(solver_discrete, d);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = d;
exact_up = DiscreteUpdater(despot_pomdp);
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, despot_planner_discrete, exact_up)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )



#=
Case 2 (Discrete and Particle Filter) - ARDESPOT evaluation
71.41204348860734 (+/-) 4.830166994411795
=#

using ARDESPOT
using ParticleFilters
import POMDPSimulators:RolloutSimulator,HistoryRecorder
import BeliefUpdaters:DiscreteUpdater

d = DiscreteLaserTagPOMDP();
lower_discrete = DefaultPolicyLB(solve(QMDPSolver(max_iterations=30,belres=1e-3,verbose=true), d));
function upper_discrete(m, b)
    dist = minimum(sum(abs, s.robot-s.target) for s in particles(b))
    closing_steps = div(dist, 2)
    if closing_steps > 1
        return -sum(1*discount(m)^t for t in 0:closing_steps-2) + 110.0*discount(m)^(closing_steps-1)
    else
        return 110.0
    end
end
solver_discrete = DESPOTSolver(
    bounds = IndependentBounds(lower_discrete, upper_discrete, check_terminal=true, consistency_fix_thresh=0.1),
    K = 100,
    T_max = 0.5,
    # max_trials=20,
    tree_in_info=true,
    default_action = :measure
);
despot_planner_discrete = solve(solver_discrete, d);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = d;
pf_up = BootstrapFilter(despot_pomdp, 100, MersenneTwister(abs(rand(Int8))));
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, despot_planner_discrete, pf_up)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )
#=
hr = HistoryRecorder(max_steps=100)
h = simulate(hr, despot_pomdp, despot_planner_discrete, pf_up);
Using the code below, you can even plot the robot trajectories from the history.
=#

#=
Case 3 (Continuous and Exact) - ARDESPOT evaluation
=#

using ARDESPOT
using ParticleFilters
import POMDPSimulators:RolloutSimulator,HistoryRecorder
import BeliefUpdaters:DiscreteUpdater

c = ContinuousLaserTagPOMDP();
lower_continuous = DefaultPolicyLB(RandomPolicy(c));
function upper_continuous(m, b::ScenarioBelief)
    dist = minimum(sum(abs, s.robot-s.target) for s in particles(b))
    closing_steps = div(dist, 2)
    if closing_steps > 1
        return -sum(1*discount(m)^t for t in 0:closing_steps-2) + 110.0*discount(m)^(closing_steps-1)
    else
        return 110.0
    end
end
solver_continuous = DESPOTSolver(
    bounds = IndependentBounds(lower_continuous, upper_continuous, check_terminal=true, consistency_fix_thresh=0.1),
    K = 100,
    T_max = 0.5,
    # max_trials = 20,
    tree_in_info=true,
    default_action=:measure
)
despot_planner_continuous = solve(solver_continuous, c);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 2
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = c;
exact_up = DiscreteUpdater(despot_pomdp);

@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, despot_planner_continuous, exact_up)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )

#=
for (b,s,a,o,r) in stepthrough(despot_pomdp, despot_planner_continuous, exact_up, "b,s,a,o,r", max_steps=5)
   # println("belief is :", b.particles);println("action: ",a, " observation: ", o)
   rsum+=r
end
return rsum

hr = HistoryRecorder(max_steps=100)
h = simulate(hr, despot_pomdp, despot_planner_continuous, exact_up);
Using the code below, you can even plot the robot trajectories from the history.
=#

#=
Case 4 (Continuous and Particle Filter) - ARDESPOT evaluation
=#

using ARDESPOT
using ParticleFilters
import POMDPSimulators:RolloutSimulator,HistoryRecorder
import BeliefUpdaters:DiscreteUpdater

c = ContinuousLaserTagPOMDP();
lower_continuous = DefaultPolicyLB(RandomPolicy(c));
function upper_continuous(m, b::ScenarioBelief)
    dist = minimum(sum(abs, s.robot-s.target) for s in particles(b))
    closing_steps = div(dist, 2)
    if closing_steps > 1
        return -sum(1*discount(m)^t for t in 0:closing_steps-2) + 110.0*discount(m)^(closing_steps-1)
    else
        return 110.0
    end
end
solver_continuous = DESPOTSolver(
    bounds = IndependentBounds(lower_continuous, upper_continuous, check_terminal=true, consistency_fix_thresh=0.1),
    K = 100,
    T_max = 0.5,
    # max_trials = 20,
    tree_in_info=true,
    default_action=:measure
)
despot_planner_continuous = solve(solver_continuous, c);
sim_rng = MersenneTwister( abs(rand(Int8)) )
# sim_rng = MersenneTwister(221)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = c;
pf_up = BootstrapFilter(despot_pomdp, 100, MersenneTwister(abs(rand(Int8))));

@showprogress for i in 1:n_episodes
    println("Episode Num: ",i)
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, despot_planner_continuous, pf_up)
end
println( mean(returns), " (+/-) ", std(returns)/sqrt(n_episodes) )

#=
hr = HistoryRecorder(max_steps=100)
h = simulate(hr, despot_pomdp, despot_planner_continuous, pf_up);
Using the code below, you can even plot the robot trajectories from the history.
=#

#=

grid_b = h.state.belief_target
T_states = LTState[]
T_states_p = Float64[]
for i in 1:d.size[1]
    for j in 1:d.size[2]
        sts = LTState( d.robot_init, SVector(i,j) )
        push!(T_states,sts)
        push!(T_states_p,grid_b[i,j])
    end
end
b = SparseCat(T_states,T_states_p)


hr = HistoryRecorder(max_steps=100)
h = simulate(hr, d, planner_discrete, up);

robot_states = SVector[]
target_states = SVector[]
robot_actions = []

#For DiscreteUpdater
belief_states = Matrix{Float64}[]
for e in 1:length(h.hist)
    s = h.hist[e].s
    b = h.hist[e].b
    push!(robot_states,s.robot)
    push!(target_states,s.target)
    push!(robot_actions,h.hist[e].a)
    matr = zeros(d.size[1],d.size[2])
    for i in 1:length(b.b)
        bs = b.state_list[i]
        if(bs.robot == s.robot)
            prob = b.b[i]
            matr[bs.target...] = prob
        end
    end
    push!(belief_states,matr)
end

#For BootstrapFilter Updater

function get_matrix_belief(pomdp, particles)
    b = zeros(pomdp.size...)
    total_particles = length(particles)
    for p in particles
        b[p.target...] += 1
    end
    b = b/total_particles
    return b
end

belief_states = []
for e in 1:length(h.hist)
    s = h.hist[e].s
    push!(robot_states,s.robot)
    push!(target_states,s.target)
    push!(robot_actions,h.hist[e].a)
    matrix_belief = get_matrix_belief(c,h.hist[e].b.particles)
    push!(belief_states,matrix_belief)
end


=#
