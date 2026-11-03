module Free_Energy_Reproduction

include("Monte_Carlo_Lambda.jl")
include("Free_Energy.jl")

const MC = Monte_Carlo_Lambda
const RHO_STAR = [0.055, 0.06, 0.065, 0.07, 0.075, 0.085, 0.095, 0.105, 0.11, 0.115,
    0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22,
    0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.36, 0.37,
    0.38, 0.39, 0.41, 0.42, 0.43, 0.44]
const DENSITIES = [(density_id=21 + i, rho_star=rho) for (i, rho) in enumerate(RHO_STAR)]
const SETTINGS = (N=50, T_star=0.05, alpha=sqrt(π), n_cut=1, m_cut=6,
    step_size=0.05, lambda_points=10, n_burnin_low=5000, n_burnin_high=100000,
    n_sweeps=100000, sample_every=10, snapshot_every=1000, n_blocks=5,
    seed_base=23_100_000, replicates=[2, 3], densities=DENSITIES)

function run_jobs(c, out)
    mkpath(out)
    rule = Free_Energy.gauss_legendre_rule(c.lambda_points)
    jobs = NamedTuple[]
    for high_density in (true, false), rep in c.replicates, d in c.densities
        (d.rho_star >= 0.05) == high_density || continue
        for k in eachindex(rule.lambdas)
            node_id = d.density_id * 10 + k
            seed = c.seed_base + (rep - 1) * 1000 + d.density_id * 10 + k
            push!(jobs, (replicate_id=rep, node_id=node_id, density_id=d.density_id,
                rho_star=d.rho_star, lambda_id=k, lambda=rule.lambdas[k],
                weight=rule.weights[k], seed=seed))
        end
    end
    prefix(s) = joinpath(out, "rep$(s.replicate_id)_node$(s.node_id)")
    pending = filter(s -> !isfile(prefix(s) * ".csv"), jobs)
    println("Pending nodes: $(length(pending)); skipped: $(length(jobs) - length(pending))")
    completed = Threads.Atomic{Int}(0)
    for high_density in (true, false)
        phase = filter(s -> (s.rho_star >= 0.05) == high_density, pending)
        Threads.@threads :dynamic for index in eachindex(phase)
            s = phase[index]
            started = time()
            burnin = s.lambda_id >= 5 ? c.n_burnin_high : c.n_burnin_low
            result = MC.run_mc_one_state(c.N, 1 / c.T_star, sqrt(s.rho_star / c.N),
                c.alpha, c.n_cut, c.m_cut, c.step_size, burnin, c.n_sweeps, c.sample_every;
                snapshot_every=c.snapshot_every, E0_over_T=0.0, ξF_over_ξ=10.0,
                interaction=:log, seed=s.seed, lambda=s.lambda)
            stem = prefix(s)
            open(stem * "_energy.f64", "w") do io
                write(io, result.U1_trace)
                write(io, result.U2_trace)
            end
            open(stem * "_coords.f64", "w") do io
                for R in result.snapshots
                    write(io, R)
                end
            end
            write(stem * "_charges.f64", result.q)
            row = merge(s, (n_burnin=burnin, n_sweeps=c.n_sweeps,
                n_samples=result.n_samples, n_accept=result.n_accept, n_trials=result.n_trials,
                elapsed_seconds=time() - started))
            open(stem * ".csv.tmp", "w") do io
                println(io, join(string.(keys(row)), ","))
                println(io, join(values(row), ","))
            end
            mv(stem * ".csv.tmp", stem * ".csv"; force=true)
            Threads.atomic_add!(completed, 1)
            println("COMPLETE rep$(s.replicate_id) node$(s.node_id) elapsed=$(row.elapsed_seconds)s")
            flush(stdout)
        end
    end
    return completed[]
end

function main()
    run_jobs(SETTINGS, joinpath(@__DIR__, "Data"))
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    Free_Energy_Reproduction.main()
end
