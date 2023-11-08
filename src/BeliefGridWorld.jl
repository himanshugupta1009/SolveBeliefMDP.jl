using CommonRLInterface
using StaticArrays
using POMDPTools:Uniform

const RL = CommonRLInterface

struct BeliefGridWorldState{T}
    prob::T
end

struct BeliefGridWorld <: AbstractEnv
    size::SVector{2, Int}
    rewards::Dict{SVector{2, Int}, Float64}
    state::BeliefGridWorldState
end

function initialbelief(size)
    b = Uniform(SVector(x, y) for x in 1:size[1], y in 1:size[2])
    return BeliefGridWorldState(b)
end

function BeliefGridWorld()
    rewards = Dict(SA[9,3]=> 10.0,
                   SA[8,8]=>  3.0,
                   SA[4,3]=>-10.0,
                   SA[4,6]=> -5.0)
    size = SA[10, 10]
    b0 = initialbelief(size)
    return BeliefGridWorld(size, rewards, b0)
end

RL.reset!(env::BeliefGridWorld) = (env.state = SA[rand(1:env.size[1]), rand(1:env.size[2])])
RL.actions(env::BeliefGridWorld) = (SA[1,0], SA[-1,0], SA[0,1], SA[0,-1])
RL.observe(env::BeliefGridWorld) = env.state
RL.terminated(env::BeliefGridWorld) = haskey(env.rewards, env.state)

function RL.act!(env::BeliefGridWorld, a)
    if rand() < 0.4 # 40% chance of going in a random direction (=30% chance of going in a wrong direction)
        a = rand(actions(env))
    end

    env.state = clamp.(env.state + a, SA[1,1], env.size)

    return get(env.rewards, env.state, 0.0)
end
