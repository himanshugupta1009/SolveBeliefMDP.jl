# using StaticArrays
# import POMDPTools:Uniform,SparseCat,weighted_iterator,action_info
# import LazySets:LineSegment,intersection

function Base.in(s::Union{MVector{2,Int},SVector{2,Int}}, o::Union{SVector{2, Int},MVector{2, Int}})
    return s[1]==o[1] && s[2]==o[2]
end

function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Union{SVector{2, Int},MVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) == o
end

function Base.in(s::Union{MVector{2,Float64},SVector{2,Float64}}, o::Set{SVector{2, Int}})
    grid_x = Int(floor(s[1]))
    grid_y = Int(floor(s[2]))
    return SVector(grid_x,grid_y) in o
end

const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0),
                        :left_up=>SVector(-1,1), :right_up=>SVector(1,1), :left_down=>SVector(-1, -1), :right_down=>SVector(1,-1)
                            )

function bounce(m, pos, change)
    #The dot operator in clamp below specifies that apply the clamp operation to each entry of that SVector with corresponding lower and upper bounds
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

function move_robot(m,pos::Union{MVector{2,Int},SVector{2,Int}},a::Symbol)
    return bounce(m, pos, actiondir[a])
end

function check_collision(m,old_pos,new_pos)
    l = LineSegment(old_pos,new_pos)
    delta_op = ( SVector(0,1),SVector(1,0) )
    delta_corner = ( SVector(-1,0),SVector(0,-1) )

    for o in m.obstacles
        for delta in delta_op
            obs_boundary = LineSegment(o,o+delta)
            # println(l,obs_boundary)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
        corner_point = o+SVector(1,1)
        for delta in delta_corner
            obs_boundary = LineSegment(corner_point,corner_point+delta)
            # println(l,obs_boundary)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
    end
    return false
end

function move_robot(m, pos::Union{MVector{2,Float64},SVector{2,Float64}}, a::SVector)
    change = SVector(a)
    new_pos = pos + change
    if( new_pos[1] >= 1.0+m.size[1] || new_pos[1] < 1.0 ||
        new_pos[2] >= 1.0+m.size[2] || new_pos[2] < 1.0  ||
        check_collision(m,pos,new_pos) )
        return pos
    else
        return new_pos
    end
end

function move_robot(m, pos::Union{MVector{2,Float64},SVector{2,Float64}}, a::Symbol)
    if(a==:measure)
        return pos
    else
        change = actiondir[a]
        new_pos = pos + change
        if( new_pos[1] >= 1.0+m.size[1] || new_pos[1] < 1.0 ||
            new_pos[2] >= 1.0+m.size[2] || new_pos[2] < 1.0  ||
            check_collision(m,pos,new_pos) )
            return pos
        else
            return new_pos
        end
    end
end

function target_transition_likelihood(m,oldrobot_pos,newrobot_pos,oldtarget)

    targets = [oldtarget]
    targetprobs = Float64[0.0]
    newrobot = SVector( Int(floor(newrobot_pos[1])),Int(floor(newrobot_pos[2])) )
    if sum(abs, newrobot - oldtarget) > 2 # move randomly
        for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
            newtarget = bounce(m, oldtarget, change)
            if newtarget == oldtarget
                targetprobs[1] += 0.25
            else
                push!(targets, newtarget)
                push!(targetprobs, 0.25)
            end
        end
    else # move away
        oldrobot = SVector( Int(floor(oldrobot_pos[1])),Int(floor(oldrobot_pos[2])) )
        away = sign.(oldtarget - oldrobot)
        if sum(abs, away) == 2 # diagonal
            away = away - SVector(0, away[2]) # preference to move in x direction
        end
        newtarget = bounce(m, oldtarget, away)
        targets[1] = newtarget
        targetprobs[1] = 1.0
    end

    target_states = SVector{2,Int}[]
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        push!(target_states, t)
        push!(probs, tp)
    end

    return SparseCat(target_states, probs)
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

function observation_likelihood(m, a, newrobot, target_pos)
    robot_pos = SVector( Int(floor(newrobot[1])),Int(floor(newrobot[2])) )
    left = robot_pos[1]-1
    right = m.size[1]-robot_pos[1]
    up = m.size[2]-robot_pos[2]
    down = robot_pos[2]-1
    ranges = SVector(left, right, up, down)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, robot_pos, obstacle)
    end
    ranges = laserbounce(ranges, robot_pos, target_pos)
    os = SVector(ranges, SVector(0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9)
    end
    return SparseCat(os, probs)
end
