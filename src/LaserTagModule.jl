module LaserTag

using StaticArrays
using CommonRLInterface
using ParticleFilters
using Random
using POMDPs

import POMDPTools:Uniform,SparseCat,weighted_iterator,action_info
import LazySets:LineSegment,intersection
import POMDPTools:RandomPolicy

include("BeliefMDP.jl")
include("LaserTag/LaserTag.jl")
include("LaserTag/LaserTagBeliefMDP.jl")
include("LaserTag/LaserTag_ExactBelief.jl")
include("LaserTag/LaserTag_PFBelief.jl")
include("LaserTag/LaserTagPOMDP.jl")

export DiscreteLaserTagBeliefMDP, ContinuousLaserTagBeliefMDP
export DiscreteLaserTagPFBeliefMDP, ContinuousLaserTagPFBeliefMDP
export LTState, DiscreteLaserTagPOMDP, ContinuousLaserTagPOMDP

end
