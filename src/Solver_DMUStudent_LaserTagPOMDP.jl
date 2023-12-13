#=
Code to Run Zach's version of LaserTagPOMDP from DMUStudent.HW6
Use both QMDP and ARDESPOT for it
=#

using POMDPs
using ARDESPOT
import POMDPSimulators:RolloutSimulator
import BeliefUpdaters:DiscreteUpdater
using ParticleFilters
using ProgressMeter
using Random
using Statistics
using DMUStudent.HW6
using QMDP

ogd = LaserTagPOMDP();
POMDPs.discount(m::LaserTagPOMDP) = 0.997

#=
********************************************************************************************************************************
QMDP Policy Evaluation
********************************************************************************************************************************
=#

solver = QMDPSolver(max_iterations=20,belres=1e-3,verbose=true)
policy = solve(solver, ogd);
sim_rng = MersenneTwister(21)
n_episodes = 1000
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
up = updater(policy);
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, ogd, policy, up)
end
mean(returns)
std(returns)/sqrt(n_episodes)
println( mean(returns) , " +- ", std(returns)/sqrt(n_episodes) )


#=
********************************************************************************************************************************
ARDESPOT Policy Evaluation
********************************************************************************************************************************
=#
lower = DefaultPolicyLB(solve(QMDPSolver(max_iterations=20,belres=1e-3,verbose=true), ogd));
function upper(m, b)
    dist = minimum(sum(abs, s.robot-s.target) for s in particles(b))
    closing_steps = div(dist, 2)
    if closing_steps > 1
        return -sum(1*discount(m)^t for t in 0:closing_steps-2) + 100.0*discount(m)^(closing_steps-1)
    else
        return 100.0
    end
end

solver = DESPOTSolver(
    bounds = IndependentBounds(lower, upper, check_terminal=true, consistency_fix_thresh=0.1),
    K = 50,
    max_trials = 20,
    default_action = :measure
);

planner = solve(solver, ogd);
sim_rng = MersenneTwister(2)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = ogd;
up = DiscreteUpdater(despot_pomdp);
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, ogd, planner, up)
end
mean(returns)
sqrt(var(returns))/sqrt(n_episodes)
println( mean(returns) , " +- ", std(returns)/sqrt(n_episodes) )
