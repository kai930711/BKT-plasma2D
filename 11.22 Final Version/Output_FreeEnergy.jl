# Lambda-integration free-energy runs (11.7 final code). Run from this directory with:
# julia --project=. Output_FreeEnergy.jl --write-jobs --N 50 --T 0.05 --E0 0 --densities 0.05,0.1 --replicates 2
# julia --project=. Output_FreeEnergy.jl --job J --output DIR
# Options: --jobs-file PATH (default FreeEnergy_jobs.csv); smoke tests only: --burnin-low --burnin-high --sweeps.
module Output_FreeEnergy

include("Monte_Carlo.jl")
include("Free_Energy.jl")
const MC = Monte_Carlo
const SETTINGS = (alpha=4.0, n_cut=1, m_cut=5, step_size=0.05, xiF_over_xi=5.0, cluster_fraction=0.02)
# Seconds per sweep at N=50 from the T016 local test, scaled as N^2.
seconds_per_sweep(N) = 0.00124775627085 * (N / 50)^2
# Sweep lengths (11.7_final_code_review/sweep_length.md): burn-in by lambda node, production, sample interval.
burnin(k, low=50_000, high=1_000_000) = k <= 4 ? low : high
const PRODUCTION = 1_000_000
const SAMPLE_EVERY = 100
# Seed lane 10; lanes 0–3 are 10.5, 4–6 are 11.3 and 7–9 are the 11.7 dielectric scans.
seed(rep, node_id) = 1_050_000_000 + 10_000_000_000 * 10 + 10_000 * (rep - 1) + node_id
const JOB_HEADER = "job,N,T_star,E0_star,replicate_id,density_id,rho_star,lambda_ids,estimated_hours"

function pack_jobs(N, T, E0, densities, replicates)
    jobs = NamedTuple[]
    add!(rep, d, rho, ids, hours) = push!(jobs, (job=length(jobs), N=N, T_star=T, E0_star=E0,
        replicate_id=rep, density_id=d, rho_star=rho, lambda_ids=join(ids, ';'), estimated_hours=hours))
    for rep in 1:replicates, (i, rho) in enumerate(densities)
        ids, hours = Int[], 0.0
        for k in 1:10
            cost = (burnin(k) + PRODUCTION) * seconds_per_sweep(N) * 1.05 / 3600
            if !isempty(ids) && hours + cost > 3
                add!(rep, i - 1, rho, ids, hours)
                ids, hours = Int[], 0.0
            end
            push!(ids, k)
            hours += cost
        end
        add!(rep, i - 1, rho, ids, hours)
    end
    return jobs
end

function write_jobs(path, jobs)
    open(path, "w") do io
        println(io, JOB_HEADER)
        for row in jobs
            println(io, join(values(row), ','))
        end
    end
end

function job_nodes(path, job)
    f = Dict(zip(split(JOB_HEADER, ','), split(readlines(path)[job+2], ',')))
    rule = Free_Energy.gauss_legendre_rule(10)
    rep, d = parse(Int, f["replicate_id"]), parse(Int, f["density_id"])
    return [(replicate_id=rep, node_id=d*10+k, density_id=d, N=parse(Int, f["N"]),
        T_star=parse(Float64, f["T_star"]), E0_star=parse(Float64, f["E0_star"]),
        rho_star=parse(Float64, f["rho_star"]), lambda_id=k, lambda=rule.lambdas[k],
        weight=rule.weights[k], seed=seed(rep, d*10+k)) for k in parse.(Int, split(f["lambda_ids"], ';'))]
end

function write_output(s, result, out, burn, sweeps, started)
    stem = joinpath(out, "rep$(s.replicate_id)_node$(s.node_id)")
    open(stem * "_energy.f64", "w") do io
        write(io, result.U1_trace)
        write(io, result.U2_trace)
    end
    row = merge(s, (n_burnin=burn, n_sweeps=sweeps, sample_every=SAMPLE_EVERY, n_samples=result.n_samples,
        n_accept=result.n_accept, n_trials=result.n_trials, elapsed_seconds=time()-started,
        n_cluster_trials=result.n_cluster_trials, n_cluster_accept=result.n_cluster_accept,
        n_cluster_contact=result.n_cluster_contact, cluster_size_sum=result.cluster_size_sum,
        image_table=result.image_table, U2_self=result.U2_self))
    open(stem * ".csv.tmp", "w") do io
        println(io, join(string.(keys(row)), ','))
        println(io, join(values(row), ','))
    end
    mv(stem * ".csv.tmp", stem * ".csv"; force=true)
    return row
end

function run_node(s, out; burn=burnin(s.lambda_id), sweeps=PRODUCTION)
    c = SETTINGS
    println("START rep=$(s.replicate_id) node=$(s.node_id) N=$(s.N) T*=$(s.T_star) E0*=$(s.E0_star) " *
        "rho*=$(s.rho_star) lambda=$(s.lambda) seed=$(s.seed) burn-in=$burn")
    flush(stdout)
    started = time()
    result = MC.run_mc_one_state(s.N, 1/s.T_star, sqrt(s.rho_star/s.N), c.alpha, c.n_cut, c.m_cut,
        c.step_size, burn, sweeps, SAMPLE_EVERY; snapshot_every=sweeps, E0_over_T=s.E0_star/s.T_star,
        ξF_over_ξ=c.xiF_over_xi, interaction=iszero(s.E0_star) ? :log : :both, seed=s.seed,
        lambda=s.lambda, cluster_moves_per_sweep=round(Int, c.cluster_fraction*s.N))
    row = write_output(s, result, out, burn, sweeps, started)
    println("COMPLETE rep=$(s.replicate_id) node=$(s.node_id) elapsed=$(row.elapsed_seconds)s " *
        "acceptance_rate=$(result.acceptance_rate) n_cluster_accept=$(row.n_cluster_accept)/$(row.n_cluster_trials) " *
        "image_table=$(row.image_table)")
    flush(stdout)
    return row
end

function run_job(path, job, out, low, high, sweeps)
    mkpath(out)
    for s in job_nodes(path, job)
        isfile(joinpath(out, "rep$(s.replicate_id)_node$(s.node_id).csv")) && continue
        run_node(s, out; burn=burnin(s.lambda_id, low, high), sweeps=sweeps)
    end
end

function main(args=ARGS)
    o = Dict("--output"=>"./results", "--jobs-file"=>joinpath(@__DIR__, "FreeEnergy_jobs.csv"),
        "--burnin-low"=>"50000", "--burnin-high"=>"1000000", "--sweeps"=>string(PRODUCTION), "--replicates"=>"2")
    write = false; i = 1
    while i <= length(args)
        if args[i] == "--write-jobs"
            write = true; i += 1
        else
            o[args[i]] = args[i+1]; i += 2
        end
    end
    if write
        write_jobs(o["--jobs-file"], pack_jobs(parse(Int, o["--N"]), parse(Float64, o["--T"]),
            parse(Float64, o["--E0"]), parse.(Float64, split(o["--densities"], ',')), parse(Int, o["--replicates"])))
    end
    if haskey(o, "--job")
        run_job(o["--jobs-file"], parse(Int, o["--job"]), o["--output"], parse(Int, o["--burnin-low"]),
            parse(Int, o["--burnin-high"]), parse(Int, o["--sweeps"]))
    end
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    Output_FreeEnergy.main()
end
