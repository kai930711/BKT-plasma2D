include("Monte_Carlo.jl")
include("one_over_epsilon0.jl")

using Statistics
using Printf

const MC = Monte_Carlo
const EPS = OneOverEpsilon0


const FIELDNAMES = [
    "N",
    "rho_star",
    "T_star",
    "kappa",
    "a",
    "α",
    "n_cut",
    "m_cut",
    "step_size",
    "n_burnin",
    "n_sweeps",
    "sample_every",
    "n_samples",
    "seed",
    "mean_abs_Q2_first_shell",
    "ϵ0_inv",
    "universal_line_4T",
    "diff_to_4T",
    "acceptance_rate",
    "U1_mean",
    "beta_U_mean",
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
)
    kappa = 1.0 / T_star
    a = MC.ST.a_from_rho_star(N, rho_star)

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
        E0_over_T = 0.0,
        ξF_over_ξ = 1.0,
        interaction = :log,
        seed = seed,
    )

    eps_result = EPS.ϵ0_inv_first_shell(
        result.snapshots,
        result.q,
        kappa,
    )

    mean_abs_Q2 = eps_result.mean_abs_Q2_first_shell
    ϵ0_inv = eps_result.ϵ0_inv

    return Dict{String, Any}(
        "N" => N,
        "rho_star" => rho_star,
        "T_star" => T_star,
        "kappa" => kappa,
        "a" => a,
        "α" => α,
        "n_cut" => n_cut,
        "m_cut" => m_cut,
        "step_size" => step_size,
        "n_burnin" => n_burnin,
        "n_sweeps" => n_sweeps,
        "sample_every" => sample_every,
        "n_samples" => result.n_samples,
        "seed" => seed,
        "mean_abs_Q2_first_shell" => mean_abs_Q2,
        "ϵ0_inv" => ϵ0_inv,
        "universal_line_4T" => 4.0 * T_star,
        "diff_to_4T" => ϵ0_inv - 4.0 * T_star,
        "acceptance_rate" => result.acceptance_rate,
        "U1_mean" => mean(result.U1_trace),
        "beta_U_mean" => mean(result.beta_U_trace),
    )
end


function main()
    line_specs = [
        (N = 50,  rho_star = 0.001, seed_base = 100000, label = "N=50, rho_star=0.001"),
        (N = 50,  rho_star = 0.005, seed_base = 101000, label = "N=50, rho_star=0.005"),
        (N = 50,  rho_star = 0.01,  seed_base = 102000, label = "N=50, rho_star=0.01"),
        (N = 200, rho_star = 0.01,  seed_base = 200000, label = "N=200, rho_star=0.01"),
    ]

    T_star_values = [
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.80,
        1.00,
    ]

    α = sqrt(π)
    n_cut = 1
    m_cut = 6

    step_size = 0.05

    n_burnin = 2000
    n_sweeps = 10000
    sample_every = 10

    output_csv = "op_dielectric_data_julia.csv"

    total_lines = length(line_specs)
    points_per_line = length(T_star_values)
    total_points = total_lines * points_per_line

    open(output_csv, "w") do io
        println(io, join(FIELDNAMES, ","))
        flush(io)

        global_point = 0

        for (line_index, line) in enumerate(line_specs)
            println("============================================================")
            @printf(
                "Starting line %d/%d: %s\n",
                line_index,
                total_lines,
                line.label,
            )
            flush(stdout)

            for (T_index, T_star) in enumerate(T_star_values)
                global_point += 1
                seed = line.seed_base + (T_index - 1)

                @printf(
                    "[line %d/%d | point %d/%d | global %d/%d] Running N=%d, rho_star=%.6g, T_star=%.6g, kappa=%.6g, seed=%d\n",
                    line_index,
                    total_lines,
                    T_index,
                    points_per_line,
                    global_point,
                    total_points,
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
                )

                write_csv_row(io, row)

                @printf(
                    "[done line %d/%d | point %d/%d | global %d/%d] 1/ϵ0=%.6f, 4T=%.6f, diff=%.6f, accept=%.4f, samples=%d\n",
                    line_index,
                    total_lines,
                    T_index,
                    points_per_line,
                    global_point,
                    total_points,
                    row["ϵ0_inv"],
                    row["universal_line_4T"],
                    row["diff_to_4T"],
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

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end