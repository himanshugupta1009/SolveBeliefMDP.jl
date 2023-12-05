# using Random
# import LazySets:LineSegment,intersection
# include("../BeliefMDP.jl")
# include("LaserTag.jl")
const RL = CommonRLInterface

struct BeliefMDPState{S,T}
    robot_pos::S
    belief_target::T
end

struct LaserTagBeliefMDP{S} <: BeliefMDP
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    target::MVector{2, Int}
    state::S
end

# Types
const DiscreteActionLaserTag   = LaserTagBeliefMDP{<:BeliefMDPState{MVector{2, Int},<:Any}}
const ContinuousActionLaserTag = LaserTagBeliefMDP{<:BeliefMDPState{MVector{2, Float64},<:Any}}

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
    env.target[1],env.target[2] = t[1],t[2]
    env.state.robot_pos[1],env.state.robot_pos[2] = r[1],r[2]

    #Reset belief over target to uniform belief
    b = resetbelief(env)
    set_belief!(env,b)

    #This empty return statement is put to prevent the reset! function from printing anything on the REPL
    return
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
RL.actions(env::DiscreteActionLaserTag) = keys(actiondir)
RL.actions(env::ContinuousActionLaserTag) = Box(lower=[-1f0, -1f0], upper=[1f0, 1f0])
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


#4: Define RL.act!
function RL.act!(env::LaserTagBeliefMDP, a)
    #=
    Move Robot and Target; Sample Obsevation; Update Belief
    =#
    S = env.state
    old_robot_pos = S.robot_pos

    #Move Robot
    new_robot_pos = move_robot(env, S.robot_pos, a)
    #Move Target
    new_target_pos = get_new_target_pos(env,old_robot_pos,new_robot_pos)
    #Sample Observation
    o = get_observation(env,new_robot_pos,new_target_pos,a)
    #Get Updated Belief
    bp = update_belief(env,S.belief_target,a,o,new_robot_pos)

    #Modify environment state
    env.state.robot_pos[1],env.state.robot_pos[2] = new_robot_pos[1],new_robot_pos[2]
    env.target[1],env.target[2] = new_target_pos[1],new_target_pos[2]
    set_belief!(env,bp)

    #Calculate reward
    r = get_reward(env,S.belief_target,a,bp)
    #Return the reward
    return r
end

function get_new_target_pos(env,old_robot_pos,new_robot_pos)
    new_target_pos_dist = target_transition_likelihood(env,old_robot_pos,new_robot_pos,env.target)
    new_target_pos = rand(new_target_pos_dist)
    return new_target_pos
end

function get_observation(env,new_robot_pos,new_target_pos,a)
    observation_dist = observation_likelihood(env, a, new_robot_pos, new_target_pos)
    observation = rand(observation_dist)
    return observation
end

function get_reward(env::LaserTagBeliefMDP,b,a,bp)
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
