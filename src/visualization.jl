using Plots

square(x,y,r) = Shape(x .+ [0,r,r,0], y .+ [0,0,r,r])

# function plot_lasertag(m::ContinuousLaserTagBeliefMDP)
#
#     n_x,n_y = m.size
#     p_size = 1000
#     p = plot(
#         legend=false,
#         gridlinewidth=2.0,
#         # gridstyle=:dash,
#         # axis=([],true),
#         axis=true,
#         gridalpha=1.0,
#         xticks=[1:1:2*(n_x+1)...],
#         yticks=[1:1:2*(n_y+1)...],
#         size=(p_size,p_size)
#         )
#
#     for o in m.obstacles
#         plot!( square(o[1],o[2],1.0), opacity=0.5, color=:brown)
#     end
#     display(p)
# end


function plot_lasertag(m::LaserTagPOMDP, robot, target, belief)

    n_x,n_y = m.size
    p_size = 1000
    p = plot(
        legend=false,
        gridlinewidth=2.0,
        # gridstyle=:dash,
        # axis=([],true),
        axis=true,
        gridalpha=1.0,
        xlims=(1,16),
        ylims=(1,16),
        xticks=[1:1:2*(n_x+5)...],
        yticks=[1:1:2*(n_y+5)...],
        size=(p_size,p_size)
        )

    heatmap!(0.5 .+ (1:n_x), 0.5 .+ (1:n_y), belief', c = cgrad(:roma,scale=:log))
    for o in m.obstacles
        plot!( square(o[1],o[2],1.0), opacity=0.5, color=:black)
    end

    plot!( [0.5+robot[1]],[0.5+robot[2]], opacity=0.9, color=:black,seriestype=:scatter, markercolor=:black,
                        markersize=p_size/35, markershape=:circle)

    plot!( [0.5+target[1]],[0.5+target[2]], opacity=0.5, color=:black,seriestype=:scatter, markercolor=:black,
                        markersize=p_size/35, markershape=:star5)

    # display(p)
end


#=
anim = @animate for k ∈ 1:length(h.hist)
    r = robot_states[k]
    t = target_states[k]
    b = belief_states[k]
    plot_lasertag(d,r,t,b)
end
gif(anim, "DESPOT_policy.gif", fps = 2)

=#
