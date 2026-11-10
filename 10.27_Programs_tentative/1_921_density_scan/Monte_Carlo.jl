module Monte_Carlo

include("Setup.jl")
include("Plasma_Interaction.jl")

using Random
using Statistics

const ST = Setup
const PI = Plasma_Interaction


function interaction_flags(interaction::Symbol)
    if interaction == :log
        return true, false
    elseif interaction == :majorana
        return false, true
    elseif interaction == :both
        return true, true
    else
        throw(ArgumentError("Choose at least one term as potential."))
    end
end


@inline function trial_position(R, i::Integer, step_size::Float64, rng::AbstractRNG)
    x_new = ST.wrap_coord(R[i, 1] + (2.0 * rand(rng) - 1.0) * step_size)
    y_new = ST.wrap_coord(R[i, 2] + (2.0 * rand(rng) - 1.0) * step_size)

    return (x_new, y_new)
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
    E0_over_T::Float64 = 0.0,
    ξF_over_ξ::Float64 = 1.0,
    interaction::Symbol = :log,
    seed::Integer = 0,
)
    include_log, include_cosh = interaction_flags(interaction)

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
        )

        beta_U_current += U2_current
    end

    beta_U_trace = Float64[]
    U1_trace = Float64[]
    U2_trace = Float64[]
    snapshots = Matrix{Float64}[]

    accept = 0
    trials = 0

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
                a;
                include_log=include_log,
                include_cosh=include_cosh,
            )

            if isfinite(ΔβU) && (ΔβU <= 0.0 || rand(rng) < exp(-ΔβU))
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

        if sweep > n_burnin
            production_sweep = sweep - n_burnin - 1

            if production_sweep % sample_every == 0
                push!(beta_U_trace, beta_U_current)
                push!(U1_trace, U1_current)
                push!(U2_trace, U2_current)
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
        n_samples = length(snapshots),
        interaction = interaction,
        include_log = include_log,
        include_cosh = include_cosh,
    )
end

end