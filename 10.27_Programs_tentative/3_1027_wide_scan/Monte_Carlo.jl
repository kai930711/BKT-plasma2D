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


# Compact, exactly neutral checkerboard droplet in the box centre (a "cold"/condensed start).
# Lattice spacing 1.15a keeps every pair outside the hard core. No RNG is consumed.
function cold_condition(N::Integer, a::Float64)
    iseven(N) || throw(ArgumentError("N must be even for a neutral start."))
    s = 1.15 * a
    L = ceil(Int, sqrt(N)) + 3
    c = (L - 1) / 2
    sites = [(i, j) for i in 0:L-1 for j in 0:L-1]
    sort!(sites; by = p -> ((p[1] - c)^2 + (p[2] - c)^2, p[1], p[2]))
    even_sites = [p for p in sites if iseven(p[1] + p[2])][1:N÷2]
    odd_sites = [p for p in sites if isodd(p[1] + p[2])][1:N÷2]
    chosen = vcat(even_sites, odd_sites)
    q = vcat(fill(1.0, N÷2), fill(-1.0, N÷2))
    R = Matrix{Float64}(undef, N, 2)
    for (n, p) in enumerate(chosen)
        R[n, 1] = ST.wrap_coord(0.5 + (p[1] - c) * s)
        R[n, 2] = ST.wrap_coord(0.5 + (p[2] - c) * s)
    end
    return q, R
end


# Defaults (init=:hot, adapt_step=false) reproduce the original run_mc_one_state exactly.
# adapt_step only changes the step size during burn-in; production uses the frozen value.
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
    init::Symbol = :hot,
    adapt_step::Bool = false,
    target_accept::Float64 = 0.35,
    adapt_every::Integer = 50,
    step_min::Float64 = 1e-4,
    step_max::Float64 = 0.25,
)
    include_log, include_cosh = interaction_flags(interaction)

    rng = MersenneTwister(seed)

    if init == :hot
        q, R = ST.initial_condition(N, a; rng=rng)
    elseif init == :cold
        q, R = cold_condition(N, a)
    else
        throw(ArgumentError("init must be :hot or :cold"))
    end

    step = step_size
    accept_mark = 0
    trials_mark = 0
    accept_burnin = 0
    trials_burnin = 0

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

            R_new_i = trial_position(R, i, step, rng)

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

        if adapt_step && sweep <= n_burnin && sweep % adapt_every == 0
            window_accept = (accept - accept_mark) / max(trials - trials_mark, 1)
            if window_accept > target_accept + 0.05
                step = min(step * 1.15, step_max)
            elseif window_accept < target_accept - 0.05
                step = max(step * 0.85, step_min)
            end
            accept_mark = accept
            trials_mark = trials
        end

        if sweep == n_burnin
            accept_burnin = accept
            trials_burnin = trials
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
        init = init,
        step_size_start = step_size,
        step_size_final = step,
        n_accept_prod = accept - accept_burnin,
        n_trials_prod = trials - trials_burnin,
        interaction = interaction,
        include_log = include_log,
        include_cosh = include_cosh,
    )
end

end