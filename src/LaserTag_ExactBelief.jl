include("LaserTagBeliefMDP.jl")


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
function lasertag_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(left, right, up, down))
    end
    return os
end

function DiscreteLaserTagBeliefMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))

    #Generate Obstacles
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

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, obsindices, t, initial_state)
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

    r = MVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    while r in obstacles || r in t
        r = MVector(1+rand(rng)*size[1], 1+rand(rng)*size[2])
    end

    b = initialbelief(size,obstacles)
    b = MMatrix{size[1],size[2],Float64}(b)
    initial_state = BeliefMDPState(r,b)

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, obsindices, t, initial_state)
end

function update_belief(m::LaserTagBeliefMDP,b::MMatrix,a,o,robot_pos)
    # bp = zeros(m.size...)
    bp = MMatrix{m.size[1],m.size[2]}(zeros(m.size...))
    for i in 1:m.size[1], j in 1:m.size[2]
        b_s = b[i,j]
        T = target_transition_likelihood(m,robot_pos,SVector(i,j))
        for k in 1:length(T.vals)
            @assert !isnan(T.probs[k])
            bp[T.vals[k]...] += b_s*T.probs[k]
        end
    end

    for i in 1:m.size[1], j in 1:m.size[2]
        O = observation_likelihood(m,a,robot_pos,SVector(i,j))
        index = findfirst(x -> x==o,O.vals)
        if(isnothing(index))
            bp[i,j] = 0.0
        else
            bp[i,j] = bp[i,j]*O.probs[index]
        end
    end

    return bp/sum(bp)
end

function set_belief!(env::LaserTagBeliefMDP,new_b::MMatrix)
    for i in 1:env.size[1], j in 1:env.size[2]
        env.state.belief_target[i,j] = new_b[i,j]
    end
end
