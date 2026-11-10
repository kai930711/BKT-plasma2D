module Monte_Carlo

include("Setup.jl")
include("Plasma_Interaction.jl")

using Random
using Statistics

const ST = Setup
const PI = Plasma_Interaction


function interaction_flags(interaction::Symbol)
    return interaction in (:log, :both), interaction in (:majorana, :both)
end


@inline function trial_position(R, i::Integer, step_size::Float64, rng::AbstractRNG)
    x_new = ST.wrap_coord(R[i, 1] + (2.0 * rand(rng) - 1.0) * step_size)
    y_new = ST.wrap_coord(R[i, 2] + (2.0 * rand(rng) - 1.0) * step_size)

    return (x_new, y_new)
end


function build_cluster(R, seed, rc)
    members = [seed]
    included = falses(size(R, 1))
    included[seed] = true
    head = 1
    while head <= length(members)
        i = members[head]
        for j in axes(R, 1)
            if !included[j] && ST.d²(R[i, 1], R[i, 2], R[j, 1], R[j, 2]) < rc^2
                included[j] = true
                push!(members, j)
            end
        end
        head += 1
    end
    return members
end


function translate_cluster(R, members, step, rng)
    dx = (2 * rand(rng) - 1) * step
    dy = (2 * rand(rng) - 1) * step
    new_pos = Matrix{Float64}(undef, length(members), 2)
    for (k, i) in enumerate(members)
        new_pos[k, 1] = ST.wrap_coord(R[i, 1] + dx)
        new_pos[k, 2] = ST.wrap_coord(R[i, 2] + dy)
    end
    return new_pos
end


function cluster_contact(R, members, new_pos, rc)
    included = falses(size(R, 1))
    included[members] .= true
    for (k, i) in enumerate(members), j in axes(R, 1)
        if !included[j] && ST.d²(new_pos[k, 1], new_pos[k, 2], R[j, 1], R[j, 2]) < rc^2
            return true
        end
    end
    return false
end


function cluster_move(R, q, kappa, α, n_cut, ρ, mvecs, weight, P,
    E0_over_T, ξF_over_ξ, a, gtab, rc, step, lambda, rng;
    include_log::Bool = true, include_cosh::Bool = false,
)
    members = build_cluster(R, rand(rng, axes(R, 1)), rc)
    new_pos = translate_cluster(R, members, step, rng)
    if cluster_contact(R, members, new_pos, rc)
        return false, true, length(members), 0.0, 0.0, 0.0, nothing, nothing
    end
    ΔβU, ΔU1, ΔU2, Δρ, ΔP = PI.energy_total_cluster_diff(
        R, q, members, new_pos, kappa, α, n_cut, ρ, mvecs, weight, P,
        E0_over_T, ξF_over_ξ, a, gtab; include_log=include_log, include_cosh=include_cosh,
    )
    ΔS = lambda * ΔβU
    accepted = ΔS <= 0.0 || rand(rng) < exp(-ΔS)
    if accepted
        R[members, :] = new_pos
    end
    return accepted, false, length(members), ΔβU, ΔU1, ΔU2, Δρ, ΔP
end


function run_mc_one_state(
    N::Integer,
    kappa::Float64,
    a::Float64,
    α::Float64,
    n_cut::Integer,
    m_cut::Integer,
    step_size::Float64,
    n_burnin::Integer,
    n_sweeps::Integer,
    sample_every::Integer;
    snapshot_every::Integer = sample_every,
    E0_over_T::Float64 = 0.0,
    ξF_over_ξ::Float64 = 5.0,
    interaction::Symbol = :log,
    seed::Integer = 0,
    lambda::Float64 = 1.0,
    cluster_moves_per_sweep::Integer = 0,
    cluster_cutoff::Float64 = 1.5,
    cluster_step::Float64 = 0.5,
)
    include_log, include_cosh = interaction_flags(interaction)
    gtab = include_cosh ? PI.majorana_image_table(N, E0_over_T, ξF_over_ξ, a) : nothing

    rng = MersenneTwister(seed)

    q, R = ST.initial_condition(N, a; rng=rng)

    U1_current = 0.0
    U2_current = 0.0
    beta_U_current = 0.0

    ρ = nothing
    mvecs = nothing
    weight = nothing
    P = nothing

    if include_log
        U1_current = PI.ewald_energy1(
            R,
            q,
            α,
            n_cut,
            m_cut,
        )

        mvecs, weight, ρ = PI.ewald_recip_state1(
            R,
            q,
            α,
            m_cut,
        )

        P = PI.ewald_polar_state1(R, q)

        beta_U_current += kappa * U1_current
    end

    if include_cosh
        U2_current = PI.energy2(
            R,
            E0_over_T,
            ξF_over_ξ,
            a,
            gtab,
        )

        beta_U_current += U2_current
    end

    beta_U_trace = Float64[]
    U1_trace = Float64[]
    U2_trace = Float64[]
    snapshots = Matrix{Float64}[]

    accept = 0
    trials = 0
    n_cluster_trials = 0
    n_cluster_accept = 0
    n_cluster_contact = 0
    cluster_size_sum = 0

    total_sweeps = n_burnin + n_sweeps

    for sweep in 1:total_sweeps
        for _ in 1:N
            i = rand(rng, 1:N)

            R_new_i = trial_position(R, i, step_size, rng)

            trials += 1

            if ST.hardcore_check_trial(R_new_i, R, a; exclude_index=i)
                continue
            end

            ΔβU, ΔU1, ΔU2, Δρ, ΔP = PI.energy_total_diff(
                R,
                q,
                i,
                R_new_i,
                kappa,
                α,
                n_cut,
                ρ,
                mvecs,
                weight,
                P,
                E0_over_T,
                ξF_over_ξ,
                a,
                gtab;
                include_log=include_log,
                include_cosh=include_cosh,
            )

            # Scale acceptance only; energy traces retain the unscaled 10.5 unit-box convention.
            ΔS = lambda * ΔβU
            if ΔS <= 0.0 || rand(rng) < exp(-ΔS)
                R[i, 1] = R_new_i[1]
                R[i, 2] = R_new_i[2]

                if include_log
                    U1_current += ΔU1
                    ρ .+= Δρ
                    P[1] += ΔP[1]
                    P[2] += ΔP[2]
                end

                if include_cosh
                    U2_current += ΔU2
                end

                beta_U_current += ΔβU
                accept += 1
            end
        end

        for _ in 1:cluster_moves_per_sweep
            accepted, contact, cluster_size, ΔβU, ΔU1, ΔU2, Δρ, ΔP = cluster_move(
                R, q, kappa, α, n_cut, ρ, mvecs, weight, P,
                E0_over_T, ξF_over_ξ, a, gtab, cluster_cutoff * a, cluster_step, lambda, rng;
                include_log=include_log, include_cosh=include_cosh,
            )
            n_cluster_trials += 1
            n_cluster_contact += contact
            cluster_size_sum += cluster_size
            if accepted
                if include_log
                    U1_current += ΔU1
                    ρ .+= Δρ
                    P[1] += ΔP[1]
                    P[2] += ΔP[2]
                end
                if include_cosh
                    U2_current += ΔU2
                end
                beta_U_current += ΔβU
                n_cluster_accept += 1
            end
        end

        if sweep > n_burnin
            production_sweep = sweep - n_burnin - 1

            if production_sweep % sample_every == 0
                push!(beta_U_trace, beta_U_current)
                push!(U1_trace, U1_current)
                push!(U2_trace, U2_current)
            end
            if production_sweep % snapshot_every == 0
                push!(snapshots, copy(R))
            end
        end
    end

    return (
        q = q,
        final_R = copy(R),
        snapshots = snapshots,
        beta_U_trace = beta_U_trace,
        U1_trace = U1_trace,
        U2_trace = U2_trace,
        acceptance_rate = accept / trials,
        n_accept = accept,
        n_trials = trials,
        n_cluster_trials = n_cluster_trials,
        n_cluster_accept = n_cluster_accept,
        n_cluster_contact = n_cluster_contact,
        cluster_size_sum = cluster_size_sum,
        n_samples = length(U1_trace),
        interaction = interaction,
        lambda = lambda,
        include_log = include_log,
        include_cosh = include_cosh,
        U2_self = PI.majorana_self_energy(gtab, N),
        image_table = gtab !== nothing,
    )
end

end
