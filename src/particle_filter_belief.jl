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

model = ParticleFilterModel{SVector{2,Int}}(particle_propogation, measurement_likelihood)
pf = BootstrapFilter(model,100)

function update_particle_filter_belief_target(m,pf,b,a,o,robot_pos)
    u = (m,robot_pos,a)
    b_new = update(pf, b, u, o)
    return b_new
end
