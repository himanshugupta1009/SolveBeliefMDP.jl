include("LaserTagBeliefMDP.jl")

struct LTState
    robot::SVector{2, Int}
    target::SVector{2, Int}
    wanderer::SVector{2, Int}
end

struct DiscreteLaserTagPOMDP <: POMDP{LTState, Symbol, SVector{4,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    robot_init::SVector{2, Int}
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
    
    LaserTagPOMDP(size, obstacles, blocked, robot_init, obsindices)
end
