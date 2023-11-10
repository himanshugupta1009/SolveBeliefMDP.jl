using CommonRLInterface
using StaticArrays
const RL = CommonRLInterface
using POMDPTools:Uniform,SparseCat
using Random

struct BeliefMDPState{T}
    robot_pos::MVector{2, Int}
    belief_target::T
end

struct LaserTagBeliefMDP <: AbstractEnv
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    obsindices::Array{Union{Nothing,Int}, 4}
    target::MVector{2, Int}
    state::BeliefMDPState
end

function initialbelief(size, obstacles)
    num_valid_states = prod(size) - length(obstacles)
    b = ones(size...)
    for obs in obstacles
        b[obs...] = 0.0
    end
    return (1/num_valid_states)*b
end

function LaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    obsindices = Array{Union{Nothing,Int}}(nothing, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while t in obstacles
        t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    r = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while r in obstacles
        r = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    b = initialbelief(size,obstacles)
    initial_state = BeliefMDPState(r,b)

    return LaserTagBeliefMDP(size, obstacles, blocked, obsindices, t, initial_state)
end

#1
function RL.reset!(env::LaserTagBeliefMDP)

    t = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    while t in env.obstacles
        t = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    end
    r = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    while r in env.obstacles
        r = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    end

    #Reset Robot and target's initial positions
    env.target[1],env.target[1] = t[1],t[2]
    env.state.robot_pos[1],env.state.robot_pos[2] = r[1],r[2]

    #Reset belief over target to uniform belief
    b = initialbelief(env.size,env.obstacles)
    for i in 1:env.size[1]
        for j in 1:env.size[2]
            env.state.belief_target[i,j] = b[i,j]
        end
    end
end

#2
RL.actions(env::LaserTagBeliefMDP) = (:left, :right, :up, :down, :measure)
const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0))
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :measure=>5)

#3
function RL.observe(env::LaserTagBeliefMDP)
    s = env.state
    #This is done because vcat concatenates using columns and not rows, while we want to concatenate row
    b_transpose = s.belief_target'
    return vcat(s.robot_pos, b_transpose...)
end


#4
function RL.act!(env::LaserTagBeliefMDP, a)
    #=
    Sample Obsevation
    Update Belief
    =#
    S = env.state

    #Move Robot
    new_robot_pos = bounce(env, S.robot_pos, actiondir[a])
    #Move Target
    new_target_pos_dist = target_transition_likelihood(env,new_robot_pos)
    new_target_pos = rand(new_target_pos_dist)
    #Sample Observation
    observation_dist = observation_likelihood(env, a, new_robot_pos, new_target_pos)
    observation = rand(observation_dist)
    #Update Belief
    bp = update_belief(env,S.belief_target,a,observation,new_robot_pos)
    #Calculate reward
    r = reward(env,S.belief_target,a,bp)

    #Modify environment state
    env.state.robot_pos[1],env.state.robot_pos[2] = new_robot_pos[1],new_robot_pos[2]
    env.target[1],env.target[2] = new_target_pos[1],new_target_pos[2]
    for i in 1:env.size[1]
        for j in 1:env.size[2]
            env.state.belief_target[i,j] = bp[i,j]
        end
    end

    #Return the reward
    return r
end


function bounce(m::LaserTagBeliefMDP, pos, change)
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

function lasertag_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1
        for right in 0:size[1]-left-1
            for up in 0:size[2]-1
                for down in 0:size[2]-up-1
                    push!(os, SVector(left, right, up, down))
                end
            end
        end
    end
    return os
end

function target_transition_likelihood(m::LaserTagBeliefMDP, newrobot, oldtarget)
    # newrobot = bounce(m, s.robot, actiondir[a])

    # if isterminal(m, s)
    #     @assert s.robot == s.target
    #     # return a new terminal state where the robot has moved
    #     # this maintains the property that the robot always moves the same, regardless of the target and wanderer states
    #     return SparseCat([LTState(newrobot, newrobot, s.wanderer)], [1.0])
    # end

    targets = [oldtarget]
    targetprobs = Float64[0.0]
    if sum(abs, newrobot - oldtarget) > 2 # move randomly
        for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
            newtarget = bounce(m, oldtarget, change)
            if newtarget == oldtarget
                targetprobs[1] += 0.25
            else
                push!(targets, newtarget)
                push!(targetprobs, 0.25)
            end
        end
    else # move away
        away = sign.(oldtarget - m.state.robot_pos)
        if sum(abs, away) == 2 # diagonal
            away = away - SVector(0, away[2]) # preference to move in x direction
        end
        newtarget = bounce(m, oldtarget, away)
        targets[1] = newtarget
        targetprobs[1] = 1.0
    end

    target_states = SVector{2,Int}[]
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        push!(target_states, t)
        push!(probs, tp)
    end

    return SparseCat(target_states, probs)
end

function observation_likelihood(m::LaserTagBeliefMDP, a, robot_pos, target_pos)
    left = robot_pos[1]-1
    right = m.size[1]-robot_pos[1]
    up = m.size[2]-robot_pos[2]
    down = robot_pos[2]-1
    ranges = SVector(left, right, up, down)
    # println(ranges)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, robot_pos, obstacle)
    end
    ranges = laserbounce(ranges, robot_pos, target_pos)
    os = SVector(ranges, SVector(0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9)
    end
    return SparseCat(os, probs)
end

function laserbounce(ranges, robot, obstacle)
    left, right, up, down = ranges
    diff = obstacle - robot
    if diff[1] == 0
        if diff[2] > 0
            up = min(up, diff[2]-1)
        elseif diff[2] < 0
            down = min(down, -diff[2]-1)
        end
    elseif diff[2] == 0
        if diff[1] > 0
            right = min(right, diff[1]-1)
        elseif diff[1] < 0
            left = min(left, -diff[1]-1)
        end
    end
    return SVector(left, right, up, down)
end

function update_belief(m::LaserTagBeliefMDP,b,a,o,robot_pos)
    bp = zeros(m.size...)

    for i in 1:m.size[1]
        for j in 1:m.size[2]
            b_s = b[i,j]
            T = target_transition_likelihood(m,robot_pos,SVector(i,j))
            for k in 1:length(T.vals)
                bp[T.vals[k]...] += b_s*T.probs[k]
            end
        end
    end

    for i in 1:m.size[1]
        for j in 1:m.size[2]
            O = observation_likelihood(m,a,robot_pos,SVector(i,j))
            index = findfirst(x -> x==o,O.vals)
            if(isnothing(index))
                bp[i,j] = 0.0
            else
                bp[i,j] = bp[i,j]*O.probs[index]
            end
        end
    end

    return bp/sum(bp)
end

function reward(env::LaserTagBeliefMDP,b,a,bp)
    if(env.target == env.state.robot_pos)
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end


#5
function RL.terminated(env::LaserTagBeliefMDP)
    return env.target == env.state.robot_pos
end
