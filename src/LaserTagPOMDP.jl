using POMDPs
using StaticArrays
using Random
import POMDPTools:SparseCat
include("LaserTagProblem.jl")

struct LTState{S,T}
    robot::S
    target::T
end

Base.convert(::Type{SVector{4, Int}}, s::LTState) = SA[s.robot..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractVector}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractArray}, s::LTState) = convert(SVector{4, Int}, s)

struct LaserTagPOMDP{S} <: POMDP{LTState, Symbol, SVector{4,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    robot_init::S
    obsindices::Array{Union{Nothing,Int}, 4}
end

function lasertag_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(left, right, up, down))
    end
    return os
end

function DiscreteLaserTagPOMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
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

    LaserTagPOMDP(SVector(size), obstacles, blocked, robot_init, obsindices)
end

POMDPs.actions(m::LaserTagPOMDP{SVector{2, Int64}}) = (:left, :right, :up, :down, :measure)
POMDPs.states(m::LaserTagPOMDP{SVector{2, Int64}}) = vec(collect(LTState(SVector(c[1],c[2]), SVector(c[3], c[4])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])))
POMDPs.observations(m::LaserTagPOMDP{SVector{2, Int64}}) = lasertag_observations(m.size)
POMDPs.discount(m::LaserTagPOMDP{SVector{2, Int64}}) = 0.95
POMDPs.stateindex(m::LaserTagPOMDP{SVector{2, Int64}}, s) = LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]))[s.robot..., s.target...]
POMDPs.actionindex(m::LaserTagPOMDP{SVector{2, Int64}}, a) = actionind[a]
POMDPs.obsindex(m::LaserTagPOMDP{SVector{2, Int64}}, o) = m.obsindices[(o.+1)...]::Int
POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Int64}}, s) = s.robot == s.target
POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Float64}}, s) = s.robot in s.target

const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0))
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :measure=>5)

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
    target_T = target_transition_likelihood(m,newrobot,oldtarget)

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

# This function needs to be changed. 1/70 to 1/61
function POMDPs.initialstate(m::LaserTagPOMDP{SVector{2, Int64}})
    return Uniform(LTState(m.robot_init, SVector(x, y)) for x in 1:m.size[1], y in 1:m.size[2])
end


function POMDPs.reward(m::LaserTagPOMDP, s, a, sp)
    if isterminal(m, s)
        return 0.0
    elseif sp.robot == sp.target
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.reward(m, s, a)
    r = 0.0
    td = transition(m, s, a)
    for (sp, w) in weighted_iterator(td)
        r += w*reward(m, s, a, sp)
    end
    return r
end
