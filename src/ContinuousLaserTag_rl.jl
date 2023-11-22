using CommonRLInterface
using StaticArrays
const RL = CommonRLInterface
import POMDPTools:Uniform,SparseCat
using Random
using RLAlgorithms
import LazySets:LineSegment,intersection

struct BeliefMDPState{T}
    robot_pos::MVector{2, Float64}
    belief_target::T
end

struct ContinuousLaserTagBeliefMDP <: AbstractEnv
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

function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Set{SVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) in o
end

function ContinuousLaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(29))
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

    r = MVector(1+rand()*size[1], 1+rand()*size[2])
    while r in obstacles || r in t
        r = MVector(1+rand()*size[1], 1+rand()*size[2])
    end

    b = initialbelief(size,obstacles)
    initial_state = BeliefMDPState(r,b)

    return ContinuousLaserTagBeliefMDP(size, obstacles, blocked, obsindices, t, initial_state)
end

#1
function RL.reset!(env::ContinuousLaserTagBeliefMDP)

    t = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    while t in env.obstacles
        t = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    end
    r = SVector(1+rand()*env.size[1], 1+rand()*env.size[2])
    while r in env.obstacles || r in t
        r = SVector(1+rand()*env.size[1], 1+rand()*env.size[2])
    end

    #Reset Robot and target's initial positions
    env.target[1],env.target[1] = t[1],t[2]
    env.state.robot_pos[1],env.state.robot_pos[2] = r[1],r[2]

    #Reset belief over target to uniform belief
    b = initialbelief(env.size,env.obstacles)
    for i in 1:env.size[1],j in 1:env.size[2]
        env.state.belief_target[i,j] = b[i,j]
    end
end


#2
RL.actions(env::ContinuousLaserTagBeliefMDP) = (vx_lower=0.0,vx_upper=1.0,vy_lower=0.0,vy_upper=1.0)
#=
function RL.actions(wrap::ContinuousLaserTagWrapper)
    a = RL.actions(wrap.env) = (vx_lower=0.0,vx_upper=1.0,vy_lower=0.0,vy_upper=1.0)
    return RLAlgorithms.Box{Float32}(SA[a[:vx_lower],a[:vy_lower]], SA[a[:vx_upper],a[:vy_upper]])
    # Box([0, 0], [1, 1])
end
RL.actions(wrap::ContinuousLaserTagWrapper) = RLAlgorithms.Box{Float32}(SA[actions(wrap.env)[:lower_bound]], SA[actions(wrap.env)[:upper_bound]])
RL.actions(wrap::ContinuousLaserTagWrapper) = RLAlgorithms.Box{Float32}(SA[0f0], SA[2f0*pi])
=#

#3
function RL.observe(env::ContinuousLaserTagBeliefMDP)
    s = env.state
    #This is done because vcat concatenates using columns and not rows, while we want to concatenate row
    b_transpose = s.belief_target'
    return vcat(s.robot_pos, b_transpose...)
end


#4
function RL.act!(env::ContinuousLaserTagBeliefMDP, a)
    #=
    Sample Obsevation
    Update Belief
    =#
    S = env.state

    #Move Robot
    new_robot_pos = move_robot(env, S.robot_pos, a)
    #Move Target
    new_target_pos_dist = target_transition_likelihood(env,new_robot_pos,env.target)
    new_target_pos = rand(new_target_pos_dist)
    #Sample Observation
    observation_dist = observation_likelihood(env, a, new_robot_pos, new_target_pos)
    observation = rand(observation_dist)
    #Update Belief
    bp = update_belief(env,S.belief_target,a,observation,new_robot_pos)

    #Modify environment state
    env.state.robot_pos[1],env.state.robot_pos[2] = new_robot_pos[1],new_robot_pos[2]
    env.target[1],env.target[2] = new_target_pos[1],new_target_pos[2]
    for i in 1:env.size[1],j in 1:env.size[2]
        env.state.belief_target[i,j] = bp[i,j]
    end

    #Calculate reward
    r = reward(env,S.belief_target,a,bp)

    #Return the reward
    return r
end

function check_collision(m::ContinuousLaserTagBeliefMDP,old_pos,new_pos)

    l = LineSegment(old_pos,new_pos)
    delta_op = ( SVector(0,1),SVector(1,0) )
    delta_corner = ( SVector(-1,0),SVector(0,-1) )

    for o in m.obstacles
        for delta in delta_op
            obs_boundary = LineSegment(o,o+delta)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
        corner_point = o+SVector(1,1)
        for delta in delta_corner
            obs_boundary = LineSegment(corner_point,corner_point+delta)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
    end
    return false
end

function move_robot(m::ContinuousLaserTagBeliefMDP, pos, a)
    change = SVector(a)
    #The dot operator in clamp below specifies that apply the clamp operation to each entry of that SVector with corresponding lower and upper bounds
    #new_pos = clamp.(pos + change, SVector(1.0,1.0), SVector(1.0,1.0)+m.size)
    new_pos = pos + change
    if( new_pos[1] >= 1.0+m.size[1] || new_pos[1] < 1.0 ||
        new_pos[2] >= 1.0+m.size[2] || new_pos[2] < 1.0  ||
        check_collision(m,pos,new_pos) )
        return pos
    else
        return new_pos
    end
end

function bounce(m::ContinuousLaserTagBeliefMDP, pos, change)
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

function lasertag_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(left, right, up, down))
    end
    return os
end

function target_transition_likelihood(m::ContinuousLaserTagBeliefMDP, robot_pos, oldtarget)
    # newrobot = bounce(m, s.robot, actiondir[a])

    # if isterminal(m, s)
    #     @assert s.robot == s.target
    #     # return a new terminal state where the robot has moved
    #     # this maintains the property that the robot always moves the same, regardless of the target and wanderer states
    #     return SparseCat([LTState(newrobot, newrobot, s.wanderer)], [1.0])
    # end

    targets = [oldtarget]
    targetprobs = Float64[0.0]
    newrobot = SVector( Int(floor(robot_pos[1])),Int(floor(robot_pos[2])) )
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
        c_old_robot_pos = m.state.robot_pos #Continuous
        d_old_robot_pos = SVector( Int(floor(c_old_robot_pos[1])),Int(floor(c_old_robot_pos[2])) ) #Discrete
        away = sign.(oldtarget - d_old_robot_pos) #Shouldn't this be the new robot position and not the old one?
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

function observation_likelihood(m::ContinuousLaserTagBeliefMDP, a, continuous_robot_pos, target_pos)

    robot_pos = SVector( Int(floor(continuous_robot_pos[1])),Int(floor(continuous_robot_pos[2])) )
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

function update_belief(m::ContinuousLaserTagBeliefMDP,b,a,o,robot_pos)
    bp = zeros(m.size...)

    for i in 1:m.size[1]
        for j in 1:m.size[2]
            b_s = b[i,j]
            T = target_transition_likelihood(m,robot_pos,SVector(i,j))
            for k in 1:length(T.vals)
                @assert !isnan(T.probs[k])
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

function reward(env::ContinuousLaserTagBeliefMDP,b,a,bp)
    if(env.state.robot_pos in env.target)
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end


#5
function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Union{SVector{2, Int},MVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) == o
end

function RL.terminated(env::ContinuousLaserTagBeliefMDP)
    return env.state.robot_pos in env.target
end
