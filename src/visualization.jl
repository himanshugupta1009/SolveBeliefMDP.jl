using Plots

square(x,y,r) = Shape(x .+ [0,r,r,0], y .+ [0,0,r,r])

function plot_lasertag(m::ContinuousLaserTagBeliefMDP)

    n_x,n_y = m.size
    p_size = 1000
    p = plot(
        legend=false,
        gridlinewidth=2.0,
        # gridstyle=:dash,
        # axis=([],true),
        axis=true,
        gridalpha=1.0,
        xticks=[1:1:2*(n_x+1)...],
        yticks=[1:1:2*(n_y+1)...],
        size=(p_size,p_size)
        )

    for o in m.obstacles
        plot!( square(o[1],o[2],1.0), opacity=0.5, color=:brown)
    end
    display(p)
end
