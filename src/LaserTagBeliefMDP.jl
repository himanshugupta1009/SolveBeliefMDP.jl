using CommonRLInterface
using StaticArrays
using POMDPTools:Uniform,SparseCat
using Random
import LazySets:LineSegment,intersection
include("BeliefMDP.jl")
const RL = CommonRLInterface

struct BeliefMDPState{S,T}
    robot_pos::S
    belief_target::T
end

struct LaserTagBeliefMDP{S} <: BeliefMDP
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    obsindices::Array{Union{Nothing,Int}, 4}
    target::MVector{2, Int}
    state::S
end

function Base.in(s::Union{MVector{2,Int},SVector{2,Int}}, o::Union{SVector{2, Int},MVector{2, Int}})
    return s[1]==o[1] && s[2]==o[2]
end

function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Union{SVector{2, Int},MVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) == o
end

function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Set{SVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) in o
end

#1: Define RL.reset!
function RL.reset!(env::LaserTagBeliefMDP)

    t = sample_pos(env,env.target)
    while t in env.obstacles
        t = sample_pos(env,env.target)
    end
    r = sample_pos(env,env.state.robot_pos)
    while r in env.obstacles || r in t
        r = sample_pos(env,env.state.robot_pos)
    end

    #Reset Robot and target's initial positions
    env.target[1],env.target[1] = t[1],t[2]
    env.state.robot_pos[1],env.state.robot_pos[2] = r[1],r[2]

    #Reset belief over target to uniform belief
    b = resetbelief(env)
    set_belief!(env,b)
end

function sample_pos(env::LaserTagBeliefMDP,r::MVector{2,Int64})
    pos = SVector(rand(1:env.size[1]), rand(1:env.size[2]))
    return pos
end

function sample_pos(env::LaserTagBeliefMDP,r::MVector{2,Float64})
    pos = SVector(1+rand()*env.size[1], 1+rand()*env.size[2])
    return pos
end


#2: Define RL.actions
RL.actions(env::LaserTagBeliefMDP{<:BeliefMDPState{MVector{2, Int},<:Any}}) = (:left, :right, :up, :down, :measure)
const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0))
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :measure=>5)

RL.actions(env::LaserTagBeliefMDP{<:BeliefMDPState{MVector{2, Float64},<:Any}}) = (vx_lower=0.0,vx_upper=1.0,vy_lower=0.0,vy_upper=1.0)
#=
function RL.actions(wrap::ContinuousLaserTagWrapper)
    a = RL.actions(wrap.env) = (vx_lower=0.0,vx_upper=1.0,vy_lower=0.0,vy_upper=1.0)
    return RLAlgorithms.Box{Float32}(SA[a[:vx_lower],a[:vy_lower]], SA[a[:vx_upper],a[:vy_upper]])
    # Box([0, 0], [1, 1])
end
RL.actions(wrap::ContinuousLaserTagWrapper) = RLAlgorithms.Box{Float32}(SA[actions(wrap.env)[:lower_bound]], SA[actions(wrap.env)[:upper_bound]])
RL.actions(wrap::ContinuousLaserTagWrapper) = RLAlgorithms.Box{Float32}(SA[0f0], SA[2f0*pi])
=#


#3: Define RL.observe
function RL.observe(env::LaserTagBeliefMDP)
    s = env.state
    formatted_b = change_belief_format(s.belief_target)
    return vcat(s.robot_pos, formatted_b)
end

function change_belief_format(b::MMatrix)
    #Transpose is taken because vcat concatenates using columns and not rows, while we want to concatenate rows
    b_transpose = b'
    return SVector(b...)
end

function change_belief_format(b::PFBelief)
    particles = b.collection.particles
    return SVector(vcat(particles...))
end

#4: Define RL.act!
function RL.act!(env::LaserTagBeliefMDP, a)
    #=
    Move Robot and Target; Sample Obsevation; Update Belief
    =#
    S = env.state

    #Move Robot
    new_robot_pos = move_robot(env, S.robot_pos, a)
    #Move Target
    new_target_pos = get_new_target_pos(env,new_robot_pos)
    #Sample Observation
    o = get_observation(env,new_robot_pos,new_target_pos,a)
    #Update Belief
    # return(env,S.belief_target,a,o,new_robot_pos)
    bp = update_belief(env,S.belief_target,a,o,new_robot_pos)

    #Modify environment state
    env.state.robot_pos[1],env.state.robot_pos[2] = new_robot_pos[1],new_robot_pos[2]
    env.target[1],env.target[2] = new_target_pos[1],new_target_pos[2]
    set_belief!(env,bp)

    #Calculate reward
    r = reward(env,S.belief_target,a,bp)
    #Return the reward
    return r
end

function move_robot(m::LaserTagBeliefMDP, pos::Union{MVector{2,Int},SVector{2,Int}},a)
    return bounce(m, pos, actiondir[a])
end

function bounce(m::LaserTagBeliefMDP, pos, change)
    #The dot operator in clamp below specifies that apply the clamp operation to each entry of that SVector with corresponding lower and upper bounds
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

function check_collision(m::LaserTagBeliefMDP,old_pos,new_pos)
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

function move_robot(m::LaserTagBeliefMDP, pos::Union{MVector{2,Float64},SVector{2,Float64}}, a)
    change = SVector(a)
    new_pos = pos + change
    if( new_pos[1] >= 1.0+m.size[1] || new_pos[1] < 1.0 ||
        new_pos[2] >= 1.0+m.size[2] || new_pos[2] < 1.0  ||
        check_collision(m,pos,new_pos) )
        return pos
    else
        return new_pos
    end
end

function target_transition_likelihood(m::LaserTagBeliefMDP, robot_pos, oldtarget)

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
        old_robot_pos = SVector( Int(floor(m.state.robot_pos[1])),Int(floor(m.state.robot_pos[2])) )
        away = sign.(oldtarget - old_robot_pos)
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

function get_new_target_pos(env,new_robot_pos)
    new_target_pos_dist = target_transition_likelihood(env,new_robot_pos, env.target)
    new_target_pos = rand(new_target_pos_dist)
    return new_target_pos
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

function observation_likelihood(m::LaserTagBeliefMDP, a, newrobot, target_pos)
    robot_pos = SVector( Int(floor(newrobot[1])),Int(floor(newrobot[2])) )
    left = robot_pos[1]-1
    right = m.size[1]-robot_pos[1]
    up = m.size[2]-robot_pos[2]
    down = robot_pos[2]-1
    ranges = SVector(left, right, up, down)
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

function get_observation(env,new_robot_pos,new_target_pos,a)
    observation_dist = observation_likelihood(env, a, new_robot_pos, new_target_pos)
    observation = rand(observation_dist)
    return observation
end

function reward(env::LaserTagBeliefMDP,b,a,bp)
    if(env.state.robot_pos in env.target)
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end


#5: Define RL.terminated
function RL.terminated(env::LaserTagBeliefMDP)
    return env.state.robot_pos in env.target
end
