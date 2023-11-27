include("LaserTagBeliefMDP.jl")
using ParticleFilters

function particle_propogation(x,u,rng)
    m,robot_pos,a = u
    target_pos = SVector(x)
    T = target_transition_likelihood(m,robot_pos,target_pos)
    new_target_pos = rand(rng,T)
    return new_target_pos
end

function measurement_likelihood(x_previous,u,x,y)
    m,robot_pos,a = u
    target_pos = SVector(x)
    O = observation_likelihood(m, a, robot_pos, target_pos)
    index = findfirst(x -> x==y,O.vals)
    if(isnothing(index))
        return 0.0
    else
        return O.probs[index]
    end
end

struct PFBelief{S,T}
    updater::S
    collection::T
end


function initialbelief(size,obstacles,::Val{M}) where M
    valid_states = Set{SVector{2,Int}}()
    for i in 1:size[1], j in 1:size[2]
        if !(SVector(i,j) in obstacles)
            push!(valid_states,SVector(i,j))
        end
    end
    dist = Uniform(valid_states)
    b = MVector{M,SVector{2,Int}}(undef)
    for i in 1:M
        p = rand(dist)
        b[i] = p
    end
    return ParticleCollection(b)
end

function resetbelief(env::LaserTagBeliefMDP{<:BeliefMDPState{<:Any, <:PFBelief}})
    N = n_particles(env.state.belief_target.collection)
    return initialbelief(env.size,env.obstacles,Val(N));
end


#Define Constructors
function lasertag_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(left, right, up, down))
    end
    return os
end

function DiscreteLaserTagPFBeliefMDP(;size=(10, 7), n_obstacles=9, num_particles=100, rng::AbstractRNG=Random.MersenneTwister(20))

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
    b = initialbelief(size,obstacles,Val(num_particles))
    model = ParticleFilterModel{SVector{2,Int}}(particle_propogation, measurement_likelihood)
    pf_updater = BootstrapFilter(model,num_particles)
    pfb = PFBelief(pf_updater,b)

    initial_state = BeliefMDPState(r,pfb)

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, t, initial_state)
end

function ContinuousLaserTagPFBeliefMDP(;size=(10, 7), n_obstacles=9, num_particles=100, rng::AbstractRNG=Random.MersenneTwister(29))
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

    r = MVector(1+rand()*size[1], 1+rand()*size[2])
    while r in obstacles || r in t
        r = MVector(1+rand()*size[1], 1+rand()*size[2])
    end

    b = initialbelief(size,obstacles,Val(num_particles))
    model = ParticleFilterModel{SVector{2,Int}}(particle_propogation, measurement_likelihood)
    pf_updater = BootstrapFilter(model,num_particles)
    pfb = PFBelief(pf_updater,b)

    initial_state = BeliefMDPState(r,pfb)

    return LaserTagBeliefMDP(SVector(size), obstacles, blocked, t, initial_state)
end

function update_belief(m::LaserTagBeliefMDP,b::PFBelief,a,o,robot_pos)
    u = (m,robot_pos,a)
    pf = b.updater
    particles = b.collection
    new_particles = update(pf,particles,u,o)
    return new_particles
end

function change_belief_format(b::PFBelief)
    particles = b.collection.particles
    return SVector(vcat(particles...))
end

function set_belief!(env::LaserTagBeliefMDP,new_b::ParticleCollection)
    new_particles = new_b.particles
    env.state.belief_target.collection.particles = new_particles
end


#=
d = DiscreteLaserTagPFBeliefMDP();
d.state.robot_pos
d.state.belief_target
d.target

RL.observe(d)
RL.terminated(d)
RL.actions(d)

rng = MersenneTwister(19)
for i in 1:100
    a = rand(rng, actions(d))
    RL.act!(d,a)
end

RL.reset!(d)
d.state.robot_pos
d.state.belief_target
d.target

c = ContinuousLaserTagPFBeliefMDP();
c.state.robot_pos
c.state.belief_target
c.target

RL.observe(c)
RL.terminated(c)
RL.actions(c)

rng = MersenneTwister(19)
for i in 1:100
    a = ( rand(rng), rand(rng) )
    RL.act!(c,a)
end

RL.reset!(c)
c.state.robot_pos
c.state.belief_target
c.target
=#
