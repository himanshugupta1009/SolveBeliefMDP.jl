# include("LaserTagBeliefMDP.jl")

const ExactBeliefLaserTag = LaserTagBeliefMDP{<:BeliefMDPState{<:Any,<:MMatrix}}

function initialbelief(size, obstacles)
    num_valid_states = prod(size) - length(obstacles)
    b = ones(size...)
    for obs in obstacles
        b[obs...] = 0.0
    end
    return (1/num_valid_states)*b
end

function resetbelief(env::LaserTagBeliefMDP{<:BeliefMDPState{<:Any,<:MMatrix}})
    b = initialbelief(env.size,env.obstacles)
    return MMatrix{env.size[1],env.size[2],Float64}(b)
end


#Define Constructors
function DiscreteLaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))

    #Generate Obstacles
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    #Initialize target
    t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while t in obstacles
        t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    #Initialize Robot
    r = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while r in obstacles
        r = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    #Initialize Belief
    b = initialbelief(size,obstacles)
    b = MMatrix{size[1],size[2],Float64}(b)
    initial_state = BeliefMDPState(r,b)

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, t, initial_state)
end

function ContinuousLaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(29))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while t in obstacles
        t = MVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    r = MVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    while r in obstacles || r in t
        r = MVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    end

    b = initialbelief(size,obstacles)
    b = MMatrix{size[1],size[2],Float64}(b)
    initial_state = BeliefMDPState(r,b)

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, t, initial_state)
end

function update_belief(m::LaserTagBeliefMDP,b::MMatrix,a,o,newrobot)
    bp = MMatrix{m.size[1],m.size[2]}(zeros(m.size...))

    if(o[1] == 1) #Robot sees the Target in its grid
        pos = SVector( Int(floor(newrobot[1])),Int(floor(newrobot[2])) )
        bp[pos...] = 1.0 #Collapse the belief state
    else
        oldrobot = m.state.robot_pos

        for i in 1:m.size[1], j in 1:m.size[2]
            b_s = b[i,j]
            T = target_transition_likelihood(m,oldrobot,newrobot,SVector(i,j))
            @assert all(isfinite, T.probs)
            for k in 1:length(T.vals)
                bp[T.vals[k]...] += b_s*T.probs[k]
            end
        end

        for i in 1:m.size[1], j in 1:m.size[2]
            O = observation_likelihood(m,a,newrobot,SVector(i,j))
            index = findfirst(x -> x==o,O.vals)
            if(isnothing(index))
                bp[i,j] = 0.0
            else
                bp[i,j] = bp[i,j]*O.probs[index]
            end
        end
    end
    return bp ./ sum(bp)
end

change_belief_format(b::MMatrix) = SVector(b)


function set_belief!(env::LaserTagBeliefMDP,new_b::MMatrix)
    env.state.belief_target .= new_b
    nothing
end


#=
include("LaserTagModule.jl")
using .LaserTag

using StaticArrays
using CommonRLInterface
using Random
const RL = CommonRLInterface

d = DiscreteLaserTagBeliefMDP();
d.state.robot_pos
d.state.belief_target
d.target

RL.observe(d)
RL.terminated(d)
RL.actions(d)

rng = MersenneTwister(19)
for i in 1:100
    a = rand(rng, RL.actions(d))
    RL.act!(d,a)
end

RL.reset!(d)
d.state.robot_pos
d.state.belief_target
d.target

c = ContinuousLaserTagBeliefMDP();
c.state.robot_pos
c.state.belief_target
c.target

RL.observe(c)
RL.terminated(c)
RL.actions(c)

rng = MersenneTwister(19)
for i in 1:100
    a = SVector(rand(rng),rand(rng))
    RL.act!(c,a)
end

RL.reset!(c)
c.state.robot_pos
c.state.belief_target
c.target
=#
