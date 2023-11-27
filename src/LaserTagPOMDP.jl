include("LaserTagBeliefMDP.jl")

struct LaserTagPOMDP <: POMDP{LTState, Symbol, SVector{4,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    robot_init::SVector{2, Int}
    obsindices::Array{Union{Nothing,Int}, 4}
end
