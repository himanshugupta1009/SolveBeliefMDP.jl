struct MGF{T<:AbstractArray}
    weight::T
end
Flux.@functor MGF

struct CGF{T<:AbstractArray}
    weight::T
end
Flux.@functor CGF

MGF(; dimx=2, nv=100, init=Flux.orthogonal(nv, dimx)) = MGF{Matrix{Float32}}(init)
CGF(; dimx=2, nv=100, init=Flux.orthogonal(nv, dimx)) = CGF{Matrix{Float32}}(init)

(f::MGF)(x) = f(x, ones(eltype(x),size(x,2))/size(x,2))
(f::MGF)((x, p)::Tuple{<:AbstractArray, <:AbstractArray}) = f(x, p)
function (f::MGF)(x::AbstractArray{<:Real, 2}, p::AbstractArray{<:Real, 1})
    exp_wx = exp.(f.weight * x)
    mgf = exp_wx * p
end
function (f::MGF)(x::AbstractArray{<:Real, 3}, p::AbstractArray)
    exp_wx = exp.(batched_mul(f.weight, x))
    mgf = batched_vec(exp_wx, p)
end
function (f::MGF)(x::AbstractArray{<:Real, 3})
    exp_wx = exp.(batched_mul(f.weight, x))
    mgf = dropdims(sum(exp_wx; dims=2); dims=2) ./ size(x,2)
end

(f::CGF)(x) = f(x, ones(eltype(x),size(x,2))/size(x,2))
(f::CGF)((x, p)::Tuple{<:AbstractArray, <:AbstractArray}) = f(x, p)
function (f::CGF)(x::AbstractArray{<:Real, 2}, p::AbstractArray{<:Real, 1})
    exp_wx = exp.(f.weight * x)
    mgf = exp_wx * p
    return log.(mgf) 
end
function (f::CGF)(x::AbstractArray{<:Real, 3}, p::AbstractArray)
    exp_wx = exp.(batched_mul(f.weight, x))
    mgf = batched_vec(exp_wx, p)
    return log.(mgf)
end
function (f::CGF)(x::AbstractArray{<:Real, 3})
    exp_wx = exp.(batched_mul(f.weight, x))
    mgf = dropdims(sum(exp_wx; dims=2); dims=2) ./ size(x,2)
    return log.(mgf)
end

using ChainRulesCore
function ChainRulesCore.rrule(f::CGF, x::AbstractArray{<:Real, 3})
    w = f.weight

    exp_wx = exp.(batched_mul(w, x))
    mgf = dropdims(sum(exp_wx; dims=2); dims=2) ./ size(x,2)
    cgf = log.(mgf) 

    function CGF_pullback(Ȳ)
        ỹ = Ȳ ./ mgf

        ỹ_3d = reshape(ỹ, (size(ỹ,1),1,size(ỹ,2)))
        p_3d = fill(1f0, (1,size(x,2),size(x,3)))

        temp1 = exp_wx .* batched_mul(ỹ_3d, p_3d)

        W̄_batch = batched_mul(temp1, batched_transpose(x))
        W̄ = dropdims(sum(W̄_batch; dims=3); dims=3)

        return Tangent{CGF}(; weight=W̄), ZeroTangent()
    end

    return cgf, CGF_pullback
end