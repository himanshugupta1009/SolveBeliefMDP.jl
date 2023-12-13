# using POMDPs
# import POMDPTools:RandomPolicy
# using Random
# include("LaserTag.jl")

struct LTState{S,T}
    robot::S
    target::T
end

Base.convert(::Type{SVector{4, Int}}, s::LTState) = SA[s.robot..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractVector}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractArray}, s::LTState) = convert(SVector{4, Int}, s)

struct LaserTagPOMDP{S} <: POMDP{LTState, Symbol, SVector{5,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    robot_init::S
    obsindices::Array{Union{Nothing,Int}, 5}
end

function lasertag_observations(size)
    os = SVector{5,Int}[]
    for same_grid in 0:1, left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(same_grid, left, right, up, down))
    end
    return os
end


#=
***************************************************************************************
LaserTag POMDp with discrete robot state space
=#
function DiscreteLaserTagPOMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    #Initialize Robot
    robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while robot_init in obstacles
        robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    obsindices = Array{Union{Nothing,Int}}(nothing, 2, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(SVector(size), obstacles, blocked, robot_init, obsindices)
end

POMDPs.states(m::LaserTagPOMDP{SVector{2, Int64}}) = vec(collect(LTState(SVector(c[1],c[2]), SVector(c[3], c[4])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])))
POMDPs.actions(m::LaserTagPOMDP{SVector{2, Int64}}) = (:left, :right, :up, :down, :left_up, :right_up, :left_down, :right_down, :measure)
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :left_up=>5, :right_up=>6, :left_down=>7, :right_down=>8, :measure=>9)
POMDPs.observations(m::LaserTagPOMDP{SVector{2, Int64}}) = lasertag_observations(m.size)
POMDPs.discount(m::LaserTagPOMDP{SVector{2, Int64}}) = 0.997
POMDPs.stateindex(m::LaserTagPOMDP{SVector{2, Int64}}, s) = LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]))[s.robot..., s.target...]
POMDPs.actionindex(m::LaserTagPOMDP{SVector{2, Int64}}, a) = actionind[a]
POMDPs.obsindex(m::LaserTagPOMDP{SVector{2, Int64}}, o) = m.obsindices[(o.+1)...]::Int
POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Int64}}, s) = s.robot == s.target


function POMDPs.initialstate(m::LaserTagPOMDP)
    states = LTState[]
    for x in 1:m.size[1],y in 1:m.size[2]
        target_state = SVector(x,y)
        if(!(target_state in m.obstacles))
            push!(states,LTState(m.robot_init,target_state))
        end
    end
    return Uniform(states)
end

function POMDPs.transition(m::LaserTagPOMDP, s, a)

    oldrobot = s.robot
    newrobot = move_robot(m, oldrobot, a)

    if isterminal(m, s)
        @assert s.robot == s.target
        # return a new terminal state where the robot has moved
        # this maintains the property that the robot always moves the same, regardless of the target state
        return SparseCat([LTState(newrobot, newrobot)], [1.0])
    end

    oldtarget = s.target
    target_T = target_transition_likelihood(m,oldrobot,newrobot,oldtarget)

    states = LTState[]
    probs = Float64[]
    for i in eachindex(target_T.vals)
        newtarget = target_T.vals[i]
        push!(states,LTState(newrobot,newtarget))
        push!(probs,target_T.probs[i])
    end

    return SparseCat(states, probs)
end

function POMDPs.observation(m::LaserTagPOMDP, a, sp)
    R = sp.robot
    T = sp.target
    O_likelihood = observation_likelihood(m,a,R,T)
    return O_likelihood
end

function POMDPs.reward(m::LaserTagPOMDP, s, a, sp)
    if isterminal(m, s)
        return 0.0
    elseif sp.robot in sp.target
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.reward(m::LaserTagPOMDP, s, a)
    r = 0.0
    td = transition(m, s, a)
    #weighted_iterator is a function provided by POMDPTools.jl
    for (sp, w) in weighted_iterator(td)
        r += w*reward(m, s, a, sp)
    end
    return r
end

function POMDPs.gen(m::LaserTagPOMDP{SVector{2, Int64}},s::LTState,a::Symbol,rng::AbstractRNG)

    curr_robot = s.robot
    curr_target = s.target

    #Move Robot
    new_robot = move_robot(m,curr_robot,a)

    #Move Target
    T_target = target_transition_likelihood(m,curr_robot,new_robot,curr_target)
    new_target = rand(rng, T_target)

    #Get Observation
    Z_target = observation_likelihood(m,a,new_robot,new_target)
    O = rand(rng,Z_target)

    new_state = LTState(new_robot,new_target)
    r = reward(m,s,a,new_state)

    return (sp = new_state,o=O,r=r)
end

#=
include("LaserTagModule.jl")
using .LaserTag

using StaticArrays
using POMDPs
d = DiscreteLaserTagPOMDP();
s = LTState( SVector(5,5), SVector(1,1) )
transition(d,s,:right)
observation(d,:right,s)
b = initialstate(d)
reward(d,s,:right)

using QMDP
solver = QMDPSolver(max_iterations=20,belres=1e-3,verbose=true)
policy = solve(solver, d);
a = action(policy,b)

import POMDPSimulators:RolloutSimulator
using ProgressMeter
using Random
using Statistics
sim_rng = MersenneTwister(21)
n_episodes = 100
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
up = updater(policy);
# up = BootstrapFilter(d, 100, MersenneTwister(abs(rand(Int8))));
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, d, policy, up)
end
mean(returns)
std(returns)/sqrt(n_episodes)


using ARDESPOT
using ParticleFilters
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

planner_discrete = solve(solver_discrete, d);
a = action(planner_discrete,b)

import POMDPSimulators:RolloutSimulator,HistoryRecorder
import BeliefUpdaters:DiscreteUpdater
using ProgressMeter
using Random
using Statistics
sim_rng = MersenneTwister(2)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = d;
up = DiscreteUpdater(despot_pomdp);
# up = BootstrapFilter(despot_pomdp, 100, MersenneTwister(abs(rand(Int8))));
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, planner_discrete, up)
end
mean(returns)
sqrt(var(returns))/sqrt(n_episodes)


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
belief_states = Matrix{Float64}[]

#For DiscreteUpdater
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
for e in 1:length(h.hist)
    s = h.hist[e].s
    push!(robot_states,s.robot)
    push!(target_states,s.target)
    push!(robot_actions,h.hist[e].a)
    push!(belief_states,h.hist[e].b.particles)
end


=#


#=
***************************************************************************************
LaserTag POMDp with continuous robot state space
=#
function ContinuousLaserTagPOMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    #Initialize Robot
    robot_init = SVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    while robot_init in obstacles
        robot_init = SVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    end

    obsindices = Array{Union{Nothing,Int}}(nothing, 2, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(SVector(size), obstacles, blocked, robot_init, obsindices)
end

function get_actions(m)
    return ( :right, :right_up, :up, :left_up, :left, :left_down, :down, :right_down, :measure )
    # return SVector( (1.0,0.0), (1.0,1.0), (0.0,1.0), (-1.0,1.0), (-1.0,0.0), (-1.0,-1.0), (0.0,-1.0), (1.0,-1.0), (-10.0,-10.0) )
end
POMDPs.actions(m::LaserTagPOMDP{SVector{2, Float64}}) = get_actions(m)
POMDPs.discount(m::LaserTagPOMDP{SVector{2, Float64}}) = 0.95
POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Float64}}, s) = s.robot in s.target

function POMDPs.gen(m::LaserTagPOMDP{SVector{2, Float64}},s::LTState,a::Symbol,rng::AbstractRNG)

    curr_robot = s.robot
    curr_target = s.target

    #Move Robot
    new_robot = move_robot(m,curr_robot,a)

    #Move Target
    T_target = target_transition_likelihood(m,curr_robot,new_robot,curr_target)
    new_target = rand(rng, T_target)

    #Get Observation
    Z_target = observation_likelihood(m,a,new_robot,new_target)
    O = rand(rng,Z_target)

    new_state = LTState(new_robot,new_target)
    r = reward(m,s,a,new_state)

    return (sp = new_state,o=O,r=r)
end


#=
include("LaserTagModule.jl")
using .LaserTag

using StaticArrays
using POMDPs
using ARDESPOT

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
planner_continuous = solve(solver_continuous, c);
b = initialstate(c)
a = action(planner_continuous,b)

import POMDPSimulators:RolloutSimulator
import BeliefUpdaters:DiscreteUpdater
using ParticleFilters
using ProgressMeter
using Random
using Statistics
sim_rng = MersenneTwister(2)
n_episodes = 10
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
despot_pomdp = c;
up = DiscreteUpdater(despot_pomdp);
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, despot_pomdp, planner_discrete, up)
end
mean(returns)
sqrt(var(returns))/sqrt(n_episodes)


function lower_continuous(pomdp, b::ScenarioBelief)
    return 0.0
    # return DefaultPolicyLB(RandomPolicy(pomdp, rng=MersenneTwister(14)))
end
function upper_continuous(pomdp::LaserTagPOMDP{SVector{2, Float64}}, b::ScenarioBelief)
    return 100.0
end


=#
