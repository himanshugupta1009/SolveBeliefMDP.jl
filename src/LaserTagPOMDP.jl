using POMDPs
using StaticArrays
using Random
using POMDPTools

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

const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0))
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :measure=>5)

function bounce(m::LaserTagPOMDP, pos, change)
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

function move_robot(m::LaserTagPOMDP,pos::Union{MVector{2,Int},SVector{2,Int}},a)
    return bounce(m, pos, actiondir[a])
end

POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Int64}}, s) = s.robot == s.target
POMDPs.isterminal(m::LaserTagPOMDP{SVector{2, Float64}}, s) = s.robot in s.target

function POMDPs.transition(m::LaserTagPOMDP, s, a)
    newrobot = move_robot(m, s.robot, a)

    if isterminal(m, s)
        @assert s.robot == s.target
        # return a new terminal state where the robot has moved
        # this maintains the property that the robot always moves the same, regardless of the target state
        return SparseCat([LTState(newrobot, newrobot)], [1.0])
    end

    targets = [s.target]
    targetprobs = Float64[0.0]
    if sum(abs, newrobot - s.target) > 2 # move randomly
        for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
            newtarget = bounce(m, s.target, change)
            if newtarget == s.target
                targetprobs[1] += 0.25
            else
                push!(targets, newtarget)
                push!(targetprobs, 0.25)
            end
        end
    else # move away
        away = sign.(s.target - s.robot)
        if sum(abs, away) == 2 # diagonal
            away = away - SVector(0, away[2]) # preference to move in x direction
        end
        newtarget = bounce(m, s.target, away)
        targets[1] = newtarget
        targetprobs[1] = 1.0
    end

    states = LTState[]
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        push!(states, LTState(newrobot, t))
        push!(probs, tp)
    end

    return SparseCat(states, probs)
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

function POMDPs.observation(m::LaserTagPOMDP, a, sp)
    left = sp.robot[1]-1
    right = m.size[1]-sp.robot[1]
    up = m.size[2]-sp.robot[2]
    down = sp.robot[2]-1
    ranges = SVector(left, right, up, down)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, sp.robot, obstacle)
    end
    ranges = laserbounce(ranges, sp.robot, sp.target)
    os = SVector(ranges, SVector(0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9)
    end
    return SparseCat(os, probs)
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
