module Setup

using Random

@inline wrap_coord(x::Float64) = x - floor(x)


@inline minimum_displacement(Δx::Float64) = Δx - round(Δx)


@inline function d²(xi::Float64, yi::Float64, xj::Float64, yj::Float64)
    Δx = xi - xj
    Δx -= round(Δx)
    Δy = yi - yj
    Δy -= round(Δy)
    return Δx*Δx + Δy*Δy
end

function a_from_rho_star(N::Integer, rho_star::Float64)
    rho_star > 0 || throw(ArgumentError("ρ⋆ must be positive."))
    return √(rho_star / N)
end


function a_from_η(N::Integer, η::Float64)
    η > 0 || throw(ArgumentError("η must be positive."))
    return √(4η / (π * N))
end


@inline function hardcore_overlap_pair(xi::Float64, yi::Float64, xj::Float64, yj::Float64, a::Float64)
    return d²(xi, yi, xj, yj) < a*a
end


function hardcore_check_trial(R_trial, R, a::Float64; 
    exclude_index::Union{Nothing, Int} = nothing, last_index::Int = size(R, 1),
)
    x = R_trial[1]
    y = R_trial[2]

    for j in 1:last_index
        (exclude_index !== nothing && j == exclude_index) && continue

        hardcore_overlap_pair(x, y, R[j, 1], R[j, 2], a) && return true
    end

    return false
end


function initial_condition(
    N::Integer, a::Float64;
    rng::AbstractRNG = Random.default_rng(), max_attempts::Int = 10_000,
)
    (N > 0 && iseven(N)) || throw(ArgumentError("N must be positive and even."))
    a > 0 || throw(ArgumentError("a must be positive."))
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive."))

    q = fill(1.0, N)
    q[(N ÷ 2 + 1):end] .= -1.0
    shuffle!(rng, q)

    R = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        placed = false

        for _ in 1:max_attempts
            x, y = rand(rng), rand(rng)
            
            overlap = hardcore_check_trial((x, y), R, a; last_index=i - 1)
            if !overlap
                R[i, 1], R[i, 2] = x, y
                placed = true
                break
            end
        end

        placed || throw(ArgumentError(
                "Could not place particle $i after $max_attempts attempts. Try a smaller hardcore diameter."
            ))

    end

    return q, R
end

end