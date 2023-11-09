using CommonRLInterface
using StaticArrays
const RL = CommonRLInterface

struct BeliefMDPState{T}
    robot_pos::MVector{2, Int}
    belief_target::T
end

struct LaserTagBeliefMDP <: AbstractEnv
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    obsindices::Array{Union{Nothing,Int}, 4}
    target::SVector{2, Int}
    state::BeliefMDPState
end

function LaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end
    robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))

    obsindices = Array{Union{Nothing,Int}}(nothing, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(size, obstacles, blocked, robot_init, obsindices)
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


function RL.reset!(env::LaserTagBeliefMDP)

end


RL.actions(env::LaserTagBeliefMDP) = (:left, :right, :up, :down, :measure)


function RL.observe(env::LaserTagBeliefMDP)
    s = env.state
    return vcat(s.robot_pos, s.belief_target)
end


function RL.terminated(env::LaserTagBeliefMDP)
    return env.target == env.state.robot_pos
end


function RL.act!(env::LaserTagBeliefMDP, a)
    #=
    Move Robot
    Move Target
    Sample Obsevation
    Update Belief
    =#
    S = env.state
    SP = 22
    new_pos = move_robot()
    new_belief = update_belief()
    env.state = BeliefMDPState()
    env.target =
    return reward(env,S,a,SP)
end
