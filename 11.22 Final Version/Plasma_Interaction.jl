module Plasma_Interaction

using SpecialFunctions: expint
using Base.MathConstants: eulergamma
using Interpolations: cubic_spline_interpolation

const γ_E = Float64(eulergamma)
const LOG2 = log(2.0)
# Majorana image table: total |ΔβU2| ≤ η per configuration (11.7_final_code_review/README.md).
const MAJORANA_ETA = 5e-5
const MAJORANA_GRID = 257
const MAJORANA_RCUT = 13.0


function ewald_self_energy1(q, α::Float64)
    return -0.25 * (γ_E + log(α^2)) * sum(abs2, q)
end


function ewald_real_energy1(R, q, α::Float64, n_cut::Integer)
    N = length(q)
    U = 0.0
    α² = α^2
    @inbounds for nx in -n_cut:n_cut
        for ny in -n_cut:n_cut
            for i in 1:N
                for j in 1:N

                    if nx == 0 && ny == 0 && i == j
                        continue
                    end

                    Δx = R[i, 1] - R[j, 1] + nx
                    Δy = R[i, 2] - R[j, 2] + ny
                    r² = Δx*Δx + Δy*Δy

                    U += q[i] * q[j] * expint(α² * r²)
                end
            end
        end
    end

    return 0.25 * U
end


function ewald_recip_energy1(R, q, α::Float64, m_cut::Integer)
    N = length(q)
    U = 0.0

    for mx in -m_cut:m_cut
        for my in -m_cut:m_cut

            if mx == 0 && my == 0
                continue
            end

            m² = mx^2 + my^2

            weight = exp(-π^2 * m² / α^2) / m²

            ρ = 0.0 + 0.0im

            for j in 1:N
                phase = mx * R[j, 1] + my * R[j, 2]
                ρ += q[j] * cis(2π * phase)
            end

            U += weight * abs2(ρ)
        end
    end

    return U / (4π)
end


function ewald_polar_energy1(R, q)
    N = length(q)
    Px = 0.0
    Py = 0.0

    for i in 1:N
        Px += q[i] * R[i, 1]
        Py += q[i] * R[i, 2]
    end

    return (π / 2) * (Px^2 + Py^2)
end


function ewald_energy1(R, q, α::Float64, n_cut::Integer, m_cut::Integer)
    return (
        ewald_self_energy1(q, α) +
        ewald_real_energy1(R, q, α, n_cut) +
        ewald_recip_energy1(R, q, α, m_cut) +
        ewald_polar_energy1(R, q)
    )
end


@inline function logcosh_stable(x)
    ax = abs(x)
    return ax + log1p(exp(-2 * ax)) - LOG2
end


function majorana_image_table(N, E0_over_T, ξF_over_ξ, a::Float64)
    ξF = ξF_over_ξ * a
    h = 0.5 / (MAJORANA_GRID - 1)
    pad = 16
    xs = range(-pad * h, 0.5 + pad * h; length=MAJORANA_GRID + 2pad)
    n_max = ceil(Int, MAJORANA_RCUT * ξF + 1)
    G = zeros(length(xs), length(xs))
    for (k, x) in enumerate(xs), (l, y) in enumerate(xs), nx in -n_max:n_max, ny in -n_max:n_max
        (nx == 0 && ny == 0) && continue
        G[k, l] -= logcosh_stable(E0_over_T * exp(-sqrt((x + nx)^2 + (y + ny)^2) / ξF))
    end
    inner = (pad + 1):(pad + MAJORANA_GRID)
    maximum(abs, view(G, inner, inner)) * N * (N - 1) / 2 <= MAJORANA_ETA && return nothing
    return cubic_spline_interpolation((xs, xs), G)
end


@inline majorana_image(::Nothing, Δx, Δy) = 0.0
@inline majorana_image(gtab, Δx, Δy) = gtab(abs(Δx), abs(Δy))


# Self-image term: each particle with its own images, (N/2)·g(0).
majorana_self_energy(gtab, N) = 0.5 * N * majorana_image(gtab, 0.0, 0.0)


function energy2_pair(Ri, Rj, E0_over_T, ξF_over_ξ, a::Float64, gtab)
    ξF = ξF_over_ξ * a

    Δx = Ri[1] - Rj[1]
    Δx -= round(Δx)

    Δy = Ri[2] - Rj[2]
    Δy -= round(Δy)

    r = sqrt(Δx^2 + Δy^2)

    x = E0_over_T * exp(-r / ξF)

    return -logcosh_stable(x) + majorana_image(gtab, Δx, Δy)
end


function energy2(R, E0_over_T, ξF_over_ξ, a::Float64, gtab)
    U2 = 0.0
    N = size(R, 1)

    for i in 1:(N - 1)
        for j in (i + 1):N
            U2 += energy2_pair(
                R[i, :],
                R[j, :],
                E0_over_T,
                ξF_over_ξ,
                a,
                gtab,
            )
        end
    end

    return U2 + majorana_self_energy(gtab, N)
end


# Since Monte Carlo only relies on energy differences, we can compute the change in energy to reduce computation workload. 


function ewald_real_energy1_diff(R, q, i, R_new_i, α::Float64, n_cut::Integer)
    qi = q[i]

    x_old = R[i, 1]
    y_old = R[i, 2]

    x_new = R_new_i[1]
    y_new = R_new_i[2]

    α² = α^2
    delta_sum = 0.0

    for nx in -n_cut:n_cut
        for ny in -n_cut:n_cut
            for j in eachindex(q)
                if j == i
                    continue
                end

                Δx_new = x_new - R[j, 1] + nx
                Δy_new = y_new - R[j, 2] + ny
                r²_new = Δx_new^2 + Δy_new^2

                Δx_old = x_old - R[j, 1] + nx
                Δy_old = y_old - R[j, 2] + ny
                r²_old = Δx_old^2 + Δy_old^2

                delta_sum += q[j] * (
                    expint(α² * r²_new) -
                    expint(α² * r²_old)
                )
            end
        end
    end

    return 0.5 * qi * delta_sum
end


function make_m_vectors(m_cut::Integer)
    mvecs = Tuple{Int, Int}[]

    for mx in -m_cut:m_cut
        for my in -m_cut:m_cut
            if mx == 0 && my == 0
                continue
            end

            push!(mvecs, (mx, my))
        end
    end

    return mvecs
end


function ewald_recip_weight1(α::Float64, mvecs)
    weight = Vector{Float64}(undef, length(mvecs))

    for k in eachindex(mvecs)
        mx, my = mvecs[k]
        m² = mx^2 + my^2

        weight[k] = exp(-π^2 * m² / α^2) / m²
    end

    return weight
end


function ewald_ρ1(R, q, mvecs)
    ρ = zeros(ComplexF64, length(mvecs))

    for k in eachindex(mvecs)
        mx, my = mvecs[k]

        ρ_k = 0.0 + 0.0im

        for j in eachindex(q)
            phase = mx * R[j, 1] + my * R[j, 2]
            ρ_k += q[j] * cis(2π * phase)
        end

        ρ[k] = ρ_k
    end

    return ρ
end


function ewald_recip_state1(R, q, α::Float64, m_cut::Integer)
    mvecs = make_m_vectors(m_cut)
    weight = ewald_recip_weight1(α, mvecs)
    ρ = ewald_ρ1(R, q, mvecs)

    return mvecs, weight, ρ
end


function ewald_recip_energy1_diff(R, q, i, R_new_i, ρ, mvecs, weight)
    qi = q[i]

    x_old = R[i, 1]
    y_old = R[i, 2]

    x_new = R_new_i[1]
    y_new = R_new_i[2]

    Δρ = similar(ρ)
    ΔU_recip = 0.0

    for k in eachindex(mvecs)
        mx, my = mvecs[k]

        phase_old = mx * x_old + my * y_old
        phase_new = mx * x_new + my * y_new

        dρ = qi * (
            cis(2π * phase_new) -
            cis(2π * phase_old)
        )

        Δρ[k] = dρ

        ΔU_recip += weight[k] * (
            abs2(ρ[k] + dρ) -
            abs2(ρ[k])
        )
    end

    return ΔU_recip / (4π), Δρ
end


function ewald_polar_state1(R, q)
    Px = 0.0
    Py = 0.0

    for i in eachindex(q)
        Px += q[i] * R[i, 1]
        Py += q[i] * R[i, 2]
    end

    return [Px, Py]
end


function ewald_polar_energy1_diff(R, q, i, R_new_i, P)
    qi = q[i]

    ΔPx = qi * (R_new_i[1] - R[i, 1])
    ΔPy = qi * (R_new_i[2] - R[i, 2])

    ΔU_polar = (π / 2) * (
        (P[1] + ΔPx)^2 +
        (P[2] + ΔPy)^2 -
        P[1]^2 -
        P[2]^2
    )

    ΔP = (ΔPx, ΔPy)

    return ΔU_polar, ΔP
end


function ewald_energy1_diff(R, q, i, R_new_i, α::Float64, n_cut::Integer, ρ, mvecs, weight,P)
    ΔU_real = ewald_real_energy1_diff(R, q, i, R_new_i, α, n_cut)

    ΔU_recip, Δρ = ewald_recip_energy1_diff(R, q, i, R_new_i, ρ, mvecs, weight)

    ΔU_polar, ΔP = ewald_polar_energy1_diff(R, q, i, R_new_i, P)

    ΔU1 = ΔU_real + ΔU_recip + ΔU_polar

    return ΔU1, Δρ, ΔP
end


function energy2_diff(R, i, R_new_i, E0_over_T, ξF_over_ξ, a::Float64, gtab)
    R_old_i = R[i, :]

    ΔU2 = 0.0

    for j in axes(R, 1)
        if j == i
            continue
        end

        U_old = energy2_pair(
            R_old_i,
            R[j, :],
            E0_over_T,
            ξF_over_ξ,
            a,
            gtab,
        )

        U_new = energy2_pair(
            R_new_i,
            R[j, :],
            E0_over_T,
            ξF_over_ξ,
            a,
            gtab,
        )

        ΔU2 += U_new - U_old
    end

    return ΔU2
end


function energy_total_diff(
    R,
    q,
    i,
    R_new_i,
    kappa,
    α::Float64,
    n_cut::Integer,
    ρ,
    mvecs,
    weight,
    P,
    E0_over_T,
    ξF_over_ξ,
    a::Float64,
    gtab;
    include_log::Bool = true,
    include_cosh::Bool = false,
)
    ΔβU = 0.0

    ΔU1 = 0.0
    ΔU2 = 0.0
    Δρ = nothing
    ΔP = nothing

    if include_log
        ΔU1, Δρ, ΔP = ewald_energy1_diff(
            R,
            q,
            i,
            R_new_i,
            α,
            n_cut,
            ρ,
            mvecs,
            weight,
            P,
        )

        ΔβU += kappa * ΔU1
    end

    if include_cosh
        ΔU2 = energy2_diff(
            R,
            i,
            R_new_i,
            E0_over_T,
            ξF_over_ξ,
            a,
            gtab,
        )

        ΔβU += ΔU2
    end

    return ΔβU, ΔU1, ΔU2, Δρ, ΔP
end


function ewald_real_energy1_cluster_diff(R, q, members, new_pos, α, n_cut)
    member_index = zeros(Int, length(q))
    member_index[members] .= eachindex(members)
    ΔU = 0.0
    α² = α^2
    # T013 增量公式：折回改变截断镜像和，成员内部有序对保留 1/4 系数。
    for (k, i) in enumerate(members), j in eachindex(q)
        i == j && continue
        l = member_index[j]
        xj = l == 0 ? R[j, 1] : new_pos[l, 1]
        yj = l == 0 ? R[j, 2] : new_pos[l, 2]
        coefficient = l == 0 ? 0.5 : 0.25
        for nx in -n_cut:n_cut, ny in -n_cut:n_cut
            dx_new = new_pos[k, 1] - xj + nx
            dy_new = new_pos[k, 2] - yj + ny
            dx_old = R[i, 1] - R[j, 1] + nx
            dy_old = R[i, 2] - R[j, 2] + ny
            ΔU += coefficient * q[i] * q[j] * (
                expint(α² * (dx_new^2 + dy_new^2)) -
                expint(α² * (dx_old^2 + dy_old^2))
            )
        end
    end
    return ΔU
end


function ewald_recip_energy1_cluster_diff(R, q, members, new_pos, ρ, mvecs, weight)
    Δρ = similar(ρ)
    ΔU = 0.0
    for k in eachindex(mvecs)
        mx, my = mvecs[k]
        dρ = 0.0 + 0.0im
        for (l, i) in enumerate(members)
            phase_new = mx * new_pos[l, 1] + my * new_pos[l, 2]
            phase_old = mx * R[i, 1] + my * R[i, 2]
            dρ += q[i] * (cis(2π * phase_new) - cis(2π * phase_old))
        end
        Δρ[k] = dρ
        ΔU += weight[k] * (abs2(ρ[k] + dρ) - abs2(ρ[k]))
    end
    return ΔU / (4π), Δρ
end


function ewald_polar_energy1_cluster_diff(R, q, members, new_pos, P)
    ΔPx = 0.0
    ΔPy = 0.0
    for (k, i) in enumerate(members)
        ΔPx += q[i] * (new_pos[k, 1] - R[i, 1])
        ΔPy += q[i] * (new_pos[k, 2] - R[i, 2])
    end
    ΔU = (π / 2) * ((P[1] + ΔPx)^2 + (P[2] + ΔPy)^2 - P[1]^2 - P[2]^2)
    return ΔU, (ΔPx, ΔPy)
end


function energy2_cluster_diff(R, members, new_pos, E0_over_T, ξF_over_ξ, a, gtab)
    included = falses(size(R, 1))
    included[members] .= true
    ΔU2 = 0.0
    for (k, i) in enumerate(members), j in axes(R, 1)
        included[j] && continue
        ΔU2 += energy2_pair(new_pos[k, :], R[j, :], E0_over_T, ξF_over_ξ, a, gtab) -
            energy2_pair(R[i, :], R[j, :], E0_over_T, ξF_over_ξ, a, gtab)
    end
    return ΔU2
end


function energy_total_cluster_diff(
    R, q, members, new_pos, kappa, α::Float64, n_cut::Integer,
    ρ, mvecs, weight, P, E0_over_T, ξF_over_ξ, a::Float64, gtab;
    include_log::Bool = true, include_cosh::Bool = false,
)
    ΔβU = 0.0
    ΔU1 = 0.0
    ΔU2 = 0.0
    Δρ = nothing
    ΔP = nothing
    if include_log
        ΔU_real = ewald_real_energy1_cluster_diff(R, q, members, new_pos, α, n_cut)
        ΔU_recip, Δρ = ewald_recip_energy1_cluster_diff(R, q, members, new_pos, ρ, mvecs, weight)
        ΔU_polar, ΔP = ewald_polar_energy1_cluster_diff(R, q, members, new_pos, P)
        ΔU1 = ΔU_real + ΔU_recip + ΔU_polar
        ΔβU += kappa * ΔU1
    end
    if include_cosh
        ΔU2 = energy2_cluster_diff(R, members, new_pos, E0_over_T, ξF_over_ξ, a, gtab)
        ΔβU += ΔU2
    end
    return ΔβU, ΔU1, ΔU2, Δρ, ΔP
end

end