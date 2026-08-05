include("Monte_Carlo.jl")
include("one_over_epsilon0.jl")
include("Free_Energy_Observables.jl")

using Statistics
using Printf

const MC = Monte_Carlo
const EPS = OneOverEpsilon0
const FEO = FreeEnergyObservables


const FIELDNAMES = [
    "N",
    "rho_star",
    "T_star",
    "kappa",
    "E0_star",
    "E0_over_T",
    "a",
    "xiF_over_xi",
    "α",
    "n_cut",
    "m_cut",
    "step_size",
    "n_burnin",
    "n_sweeps",
    "sample_every",
    "n_blocks",
    "n_samples",
    "seed",
    "mean_abs_Q2_first_shell",
    "ϵ0_inv",
    "ϵ0_inv_block_std",
    "universal_line_4T",
    "diff_to_4T",
    "acceptance_rate",
    "U1_mean",
    "U1_block_std",
    "U2_mean",
    "U2_block_std",
    "beta_U_mean",
    "beta_U_block_std",
    "majorana_response_mean",
    "majorana_response_block_std",
    "d_betaF_d_kappa_mean",
    "d_betaF_d_kappa_block_std",
    "d_betaF_d_E0_mean",
    "d_betaF_d_E0_block_std",
]


function csv_value(x)
    if x === nothing
        return ""
    else
        return string(x)
    end
end


function write_csv_row(io, row::Dict{String, Any})
    values = [csv_value(row[name]) for name in FIELDNAMES]
    println(io, join(values, ","))
    flush(io)
end


function run_one_state(
    N::Integer,
    rho_star::Float64,
    T_star::Float64,
    α::Float64,
    n_cut::Integer,
    m_cut::Integer,
    step_size::Float64,
    n_burnin::Integer,
    n_sweeps::Integer,
    sample_every::Integer,
    seed::Integer,
    ;
    E0_star::Float64 = 1.0,
    n_blocks::Integer = 5,
)
    kappa = 1.0 / T_star
    E0_over_T = E0_star / T_star
    a = MC.ST.a_from_rho_star(N, rho_star)
    ξF_over_ξ = 10.0

    result = MC.run_mc_one_state(
        N,
        kappa,
        a,
        α,
        n_cut,
        m_cut,
        step_size,
        n_burnin,
        n_sweeps,
        sample_every;
        E0_over_T = E0_over_T,
        ξF_over_ξ = ξF_over_ξ,
        interaction = :both,
        seed = seed,
    )

    eps_result = EPS.ϵ0_inv_first_shell(
        result.snapshots,
        result.q,
        kappa,
        n_blocks = n_blocks,
    )

    free_energy_result = FEO.free_energy_derivative_observables(
        result.snapshots,
        result.U1_trace,
        result.U2_trace,
        result.beta_U_trace,
        kappa,
        E0_star,
        ξF_over_ξ,
        a,
        n_blocks = n_blocks,
    )

    mean_abs_Q2 = eps_result.mean_abs_Q2_first_shell
    ϵ0_inv = eps_result.ϵ0_inv

    return Dict{String, Any}(
        "N" => N,
        "rho_star" => rho_star,
        "T_star" => T_star,
        "kappa" => kappa,
        "E0_star" => E0_star,
        "E0_over_T" => E0_over_T,
        "a" => a,
        "xiF_over_xi" => ξF_over_ξ,
        "α" => α,
        "n_cut" => n_cut,
        "m_cut" => m_cut,
        "step_size" => step_size,
        "n_burnin" => n_burnin,
        "n_sweeps" => n_sweeps,
        "sample_every" => sample_every,
        "n_blocks" => eps_result.n_blocks,
        "n_samples" => result.n_samples,
        "seed" => seed,
        "mean_abs_Q2_first_shell" => mean_abs_Q2,
        "ϵ0_inv" => ϵ0_inv,
        "ϵ0_inv_block_std" => eps_result.ϵ0_inv_block_std,
        "universal_line_4T" => 4.0 * T_star,
        "diff_to_4T" => ϵ0_inv - 4.0 * T_star,
        "acceptance_rate" => result.acceptance_rate,
        "U1_mean" => free_energy_result.U1_mean,
        "U1_block_std" => free_energy_result.U1_block_std,
        "U2_mean" => free_energy_result.U2_mean,
        "U2_block_std" => free_energy_result.U2_block_std,
        "beta_U_mean" => free_energy_result.beta_U_mean,
        "beta_U_block_std" => free_energy_result.beta_U_block_std,
        "majorana_response_mean" => free_energy_result.majorana_response_mean,
        "majorana_response_block_std" =>
            free_energy_result.majorana_response_block_std,
        "d_betaF_d_kappa_mean" => free_energy_result.d_betaF_d_kappa_mean,
        "d_betaF_d_kappa_block_std" =>
            free_energy_result.d_betaF_d_kappa_block_std,
        "d_betaF_d_E0_mean" => free_energy_result.d_betaF_d_E0_mean,
        "d_betaF_d_E0_block_std" =>
            free_energy_result.d_betaF_d_E0_block_std,
    )
end


function e0_output_tag(E0_star::Real)
    return replace(@sprintf("%.1f", E0_star), "." => "p")
end


function main()
    line_specs = [
        (N = 2, rho_star = 0.001, seed_base = 100000, label = "N=2, rho_star=0.001"),
    ]

    T_star_values = collect(range(0.05, 0.5; length=51))
    E0_star_values = collect(0.0:0.1:1.0)

    α = sqrt(π)
    n_cut = 1
    m_cut = 6

    step_size = 0.05

    n_burnin = 2000
    n_sweeps = 10000
    sample_every = 10
    n_blocks = 5

    output_dir = "Data"
    mkpath(output_dir)

    total_lines = length(line_specs)
    points_per_line = length(T_star_values)
    total_E0_values = length(E0_star_values)
    total_points = total_lines * total_E0_values * points_per_line

    global_point = 0

    for (E0_index, E0_star) in enumerate(E0_star_values)
        E0_tag = e0_output_tag(E0_star)
        output_csv = joinpath(output_dir, "op_dielectric_E0_$(E0_tag).csv")

        open(output_csv, "w") do io
            println(io, join(FIELDNAMES, ","))
            flush(io)

            for (line_index, line) in enumerate(line_specs)
                println("============================================================")
                @printf(
                    "Starting E0* %d/%d: %.6g | line %d/%d: %s\n",
                    E0_index,
                    total_E0_values,
                    E0_star,
                    line_index,
                    total_lines,
                    line.label,
                )
                flush(stdout)

                for (T_index, T_star) in enumerate(T_star_values)
                    global_point += 1
                    seed =
                        line.seed_base +
                        (E0_index - 1) * points_per_line +
                        (T_index - 1)

                    @printf(
                        "[E0* %d/%d | line %d/%d | point %d/%d | global %d/%d] Running E0*=%.6g, N=%d, rho_star=%.6g, T_star=%.6g, kappa=%.6g, seed=%d\n",
                        E0_index,
                        total_E0_values,
                        line_index,
                        total_lines,
                        T_index,
                        points_per_line,
                        global_point,
                        total_points,
                        E0_star,
                        line.N,
                        line.rho_star,
                        T_star,
                        1.0 / T_star,
                        seed,
                    )
                    flush(stdout)

                    row = run_one_state(
                        line.N,
                        line.rho_star,
                        T_star,
                        α,
                        n_cut,
                        m_cut,
                        step_size,
                        n_burnin,
                        n_sweeps,
                        sample_every,
                        seed,
                        E0_star = E0_star,
                        n_blocks = n_blocks,
                    )

                    write_csv_row(io, row)

                    @printf(
                        "[done E0* %d/%d | line %d/%d | point %d/%d | global %d/%d] 1/ϵ0=%.6f ± %.6f, d(βF)/dE0=%.6f ± %.6f, accept=%.4f, samples=%d\n",
                        E0_index,
                        total_E0_values,
                        line_index,
                        total_lines,
                        T_index,
                        points_per_line,
                        global_point,
                        total_points,
                        row["ϵ0_inv"],
                        row["ϵ0_inv_block_std"],
                        row["d_betaF_d_E0_mean"],
                        row["d_betaF_d_E0_block_std"],
                        row["acceptance_rate"],
                        row["n_samples"],
                    )
                    flush(stdout)
                end
            end
        end

        println("Saved CSV to $(output_csv)")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
