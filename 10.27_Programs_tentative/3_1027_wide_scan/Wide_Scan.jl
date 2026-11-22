# Wide (T*, E0*) scan at fixed N and rho*, with hot/cold starts and an adaptive step size.
# Adapted from 9.21/Task_Animation_Dielectric.jl. The on-disk schema is unchanged, so read_scan.py
# works as-is; run_extra.csv additionally records the tuned step size and production-only acceptance.
#   julia --project=. Wide_Scan.jl --dry-run
#   julia --project=. Wide_Scan.jl --replicate-id 1 --init hot --start-id 1 --end-id 30
include("Monte_Carlo.jl")
using Printf, Dates, SHA, TOML
const MC = Monte_Carlo

const DEFAULT_RHO = "0.05"
const DEFAULT_T_RANGE = "0.05:0.025:0.60"     # start:step:stop
const DEFAULT_E0_RANGE = "0:0.05:1.0"
const SETTINGS = (alpha=sqrt(π), n_cut=1, m_cut=6, step_size=0.05,
    n_burnin=5000, n_sweeps=10000, sample_every=10, xiF_over_xi=10.0,
    adapt_step=true, target_accept=0.35, init=:hot)

function parse_range(spec::AbstractString)
    parts = parse.(Float64, split(spec, ":"))
    length(parts) == 3 || error("range must be start:step:stop, got $spec")
    a, s, b = parts
    (s > 0 && b >= a) || error("invalid range $spec")
    n = floor(Int, (b - a) / s + 1e-9)
    return Tuple(round(a + k * s; digits=6) for k in 0:n)
end

function state_grid(; N::Int, rhos, Ts, E0s, replicate_id::Int, seed_base::Int)
    1 <= replicate_id <= 1000000 || error("replicate-id must be 1..1000000")
    seed_base >= 0 || error("seed-base must be nonnegative")
    states = NamedTuple[]
    for rho in rhos, T in Ts, E0 in E0s
        id = length(states) + 1
        seed = seed_base + 10000 * (replicate_id - 1) + id
        push!(states, (state_id=id, run_id=@sprintf("s%04d_r%04d", id, replicate_id),
            replicate_id=replicate_id, N=N, rho_star=rho, T_star=T, E0_star=E0, seed=seed))
    end
    return states
end

function csvrow(io, values)
    println(io, join(("\"" * replace(string(v), "\"" => "\"\"") * "\"" for v in values), ","))
end

function source_hashes()
    files = ("Wide_Scan.jl", "Monte_Carlo.jl", "Setup.jl", "Plasma_Interaction.jl", "Project.toml", "Manifest.toml")
    return Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__, f)))) for f in files)
end

init_label(c) = c.init == :hot ? "random_nonoverlap" : "cold_checkerboard_droplet"

function write_plan(out, states, c, ngrid)
    hashes = source_hashes()
    version = bytes2hex(sha256(join((f * ":" * hashes[f] for f in sort(collect(keys(hashes)))), "\n")))
    endian = ENDIAN_BOM == 0x04030201 ? "little" : "big"
    metadata = Dict(
        "schema_version" => 1, "created_at_utc" => string(now(UTC)),
        "julia_version" => string(VERSION), "code_version_sha256" => version,
        "source_sha256" => hashes, "n_selected_states" => length(states), "n_grid_states" => ngrid,
        "scan_kind" => "wide_scan", "initial_condition_kind" => init_label(c),
        "adapt_step" => c.adapt_step, "target_accept" => c.target_accept,
        "coordinate_dtype" => "Float64", "byte_order" => endian,
        "coordinates_file" => "coordinates.f64", "charges_file" => "charges.f64",
        "coordinate_storage" => "For each frame: x[1:N], then y[1:N]; frames in sampling order. No header.",
        "charge_storage" => "One Float64 q[1:N] array per run. Same particle order as coordinates.",
        "offset_units" => "bytes, zero-based from beginning of each binary file",
        "frame_bytes" => "16*N",
        "boundaries" => "periodic square; wrapped coordinates in [0,1); Lx=Ly=1",
        "state_order" => "rho outermost, T next, E0 innermost; IDs are global within this scan's grid",
        "seed_rule" => "seed_base + 10000*(replicate_id-1) + state_id",
        "samples_policy" => "Production sweeps 1, 1+sample_every, ... <= n_sweeps; no burn-in configurations.",
        "step_size_policy" => "step_size in states.csv is the STARTING step. If adapt_step, it is tuned only during burn-in toward target_accept and then frozen; the frozen value is in run_extra.csv.",
        "recovery" => "Each parameter point is exported only after the MC function returns. Only complete rows in progress.csv mark valid runs.",
        "restart" => "No exact restart checkpoint or RNG state is stored. Reruns use a new output directory.",
        "definitions" => Dict(
            "rho_star" => "N*sigma^2/L^2; N counts both charge species",
            "a" => "sigma/L=sqrt(rho_star/N); hard-core exclusion distance in unit-box coordinates",
            "T_star" => "k_B*T/K; K is the Coulomb logarithmic energy scale",
            "kappa" => "K/(k_B*T)=1/T_star",
            "E0_star" => "E0/K",
            "E0_over_T" => "E0/(k_B*T)=E0_star/T_star",
            "xiF_over_xi" => "xiF/sigma in this implementation (xi is identified with hard-core diameter)",
            "U1" => "dimensionless Coulomb Ewald energy including self and polarization terms",
            "U2" => "sum over i<j of -logcosh((E0/(k_B*T))*exp(-r_ij/xiF)); 0 when off",
            "beta_U" => "kappa*U1+U2; dimensionless effective sampling energy",
            "cold_start" => "compact exactly-neutral checkerboard droplet, lattice spacing 1.15*sigma, centred in the box"))
    open(joinpath(out, "metadata.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    ns = cld(c.n_sweeps, c.sample_every)
    coord_offset = 0; charge_offset = 0
    open(joinpath(out, "states.csv"), "w") do io
        csvrow(io, ["state_id", "run_id", "replicate_id", "N", "rho_star", "Lx", "Ly", "a",
            "T_star", "kappa", "E0_star", "E0_over_T", "xiF_over_xi", "xiF_over_L", "interaction",
            "alpha", "n_cut", "m_cut", "step_size", "n_burnin", "n_sweeps", "sample_every",
            "seed", "initial_condition", "n_samples_planned", "coordinates_offset_bytes",
            "charges_offset_bytes", "code_version_sha256"])
        for s in states
            a = sqrt(s.rho_star / s.N)
            csvrow(io, (s.state_id, s.run_id, s.replicate_id, s.N, s.rho_star, 1.0, 1.0, a,
                s.T_star, 1 / s.T_star, s.E0_star, s.E0_star / s.T_star, c.xiF_over_xi, c.xiF_over_xi * a,
                iszero(s.E0_star) ? "log" : "both", c.alpha, c.n_cut, c.m_cut, c.step_size,
                c.n_burnin, c.n_sweeps, c.sample_every, s.seed, init_label(c),
                ns, coord_offset, charge_offset, version))
            coord_offset += 16 * s.N * ns; charge_offset += 8 * s.N
        end
    end
end

function run_states(; states, settings, ngrid, output_root, dry_run=false)
    c = settings
    isempty(states) && error("No states selected")
    length(unique(s.state_id for s in states)) == length(states) || error("Duplicate state IDs")
    c.n_burnin >= 0 && c.n_sweeps > 0 && c.sample_every > 0 || error("Invalid sweep settings")
    c.init in (:hot, :cold) || error("init must be hot or cold")
    isfinite(c.alpha) && c.alpha > 0 && c.n_cut >= 0 && c.m_cut >= 1 &&
        isfinite(c.step_size) && c.step_size > 0 && isfinite(c.xiF_over_xi) && c.xiF_over_xi > 0 || error("Invalid settings")
    for s in states
        s.N > 0 && iseven(s.N) && isfinite(s.rho_star) && s.rho_star > 0 &&
            isfinite(s.T_star) && s.T_star > 0 && isfinite(s.E0_star) && s.E0_star >= 0 || error("Invalid state")
    end
    ns = cld(c.n_sweeps, c.sample_every)
    println("Grid: $ngrid parameter points; selected $(length(states)); first=$(first(states).run_id), last=$(last(states).run_id)")
    println("N=$(first(states).N), rho*=$(first(states).rho_star), init=$(c.init), adaptive step=$(c.adapt_step) (target acc $(c.target_accept))")
    println("Settings: ", c, "; planned production frames/run=", ns)
    println("Majorana OFF for E0_star=0; raw configurations only, correlations are postprocessed.")
    if dry_run
        length(states) <= 6 && foreach(println, states)
        println("Dry run: no files written and no simulation started.")
        return nothing
    end
    mkpath(output_root)
    out = mktempdir(abspath(output_root); prefix="wide_scan_", cleanup=false)
    write_plan(out, states, c, ngrid)
    println("Output: ", out); flush(stdout)
    open(joinpath(out, "coordinates.f64"), "w") do coords
      open(joinpath(out, "charges.f64"), "w") do charges
       open(joinpath(out, "samples.csv"), "w") do samples
        open(joinpath(out, "progress.csv"), "w") do progress
         open(joinpath(out, "run_extra.csv"), "w") do extra
          csvrow(samples, ["state_id", "run_id", "sample", "production_sweep", "total_sweep",
            "U1", "U2", "beta_U"])
          csvrow(progress, ["state_id", "run_id", "status", "phase", "total_sweep", "production_sweep",
            "n_samples_committed", "n_trials", "n_accept", "coordinates_end_bytes", "charges_end_bytes",
            "samples_start_bytes", "samples_end_bytes", "elapsed_seconds", "timestamp_utc"])
          csvrow(extra, ["state_id", "run_id", "replicate_id", "init", "step_size_start", "step_size_final",
            "adapt_step", "n_accept_prod", "n_trials_prod", "acceptance_prod", "n_accept_total", "n_trials_total"])
          flush(samples); flush(progress); flush(extra)
          for (index, s) in enumerate(states)
            started = time(); samples_start = position(samples)
            @printf("[%s] START %d/%d %s rho*=%.4g T*=%.3g E0*=%.3g\n",
                string(now(UTC)), index, length(states), s.run_id, s.rho_star, s.T_star, s.E0_star)
            flush(stdout)
            try
                result = MC.run_mc_one_state(s.N, 1 / s.T_star, sqrt(s.rho_star / s.N), c.alpha, c.n_cut,
                    c.m_cut, c.step_size, c.n_burnin, c.n_sweeps, c.sample_every;
                    E0_over_T=s.E0_star / s.T_star, ξF_over_ξ=c.xiF_over_xi,
                    interaction=iszero(s.E0_star) ? :log : :both, seed=s.seed,
                    init=c.init, adapt_step=c.adapt_step, target_accept=c.target_accept)
                result.n_samples == ns == length(result.snapshots) == length(result.U1_trace) ==
                    length(result.U2_trace) == length(result.beta_U_trace) || error("Unexpected sample count")
                write(charges, result.q)
                for (i, R) in enumerate(result.snapshots)
                    write(coords, R)
                    sweep = 1 + (i - 1) * c.sample_every
                    csvrow(samples, (s.state_id, s.run_id, i, sweep, c.n_burnin + sweep,
                        result.U1_trace[i], result.U2_trace[i], result.beta_U_trace[i]))
                end
                flush(coords); flush(charges); flush(samples)
                csvrow(extra, (s.state_id, s.run_id, s.replicate_id, result.init, result.step_size_start,
                    result.step_size_final, c.adapt_step, result.n_accept_prod, result.n_trials_prod,
                    result.n_accept_prod / result.n_trials_prod, result.n_accept, result.n_trials))
                flush(extra)
                csvrow(progress, (s.state_id, s.run_id, "complete", "production", c.n_burnin + c.n_sweeps,
                    c.n_sweeps, result.n_samples, result.n_trials, result.n_accept,
                    position(coords), position(charges), samples_start, position(samples),
                    time() - started, string(now(UTC))))
                flush(progress)
                @printf("[%s] COMPLETE %d/%d %s frames=%d acc_prod=%.3f step=%.4g elapsed=%.1fs\n",
                    string(now(UTC)), index, length(states), s.run_id, result.n_samples,
                    result.n_accept_prod / result.n_trials_prod, result.step_size_final, time() - started)
                flush(stdout)
            catch err
                println(stderr, "FAILED ", s.run_id, ". Earlier completed points remain in ", out)
                rethrow()
            end
          end
         end
        end
       end
      end
    end
    println("Finished: ", out)
    return out
end

function main(args=ARGS)
    values = Dict{String,String}(); dry = false
    allowed = ("--state-id", "--start-id", "--end-id", "--replicate-id", "--seed-base",
        "--burnin", "--sweeps", "--sample-every", "--output-root", "--N", "--rho",
        "--T-range", "--E0-range", "--init", "--adapt-step", "--target-accept", "--step-size")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            println("""
            julia --project=. Wide_Scan.jl [options]
              --dry-run                  Validate/print plan without simulating or writing
              --N N                      Particles (even; default 50)
              --rho R[,R2,...]           rho* value(s) (default $DEFAULT_RHO)
              --T-range a:step:b         T* grid (default $DEFAULT_T_RANGE)
              --E0-range a:step:b        E0* grid (default $DEFAULT_E0_RANGE)
              --init hot|cold            Random non-overlap start, or compact neutral droplet (default hot)
              --adapt-step true|false    Tune step size during burn-in only (default true)
              --target-accept X          Burn-in acceptance target (default 0.35)
              --step-size S              Starting step (default 0.05)
              --state-id ID | --start-id A --end-id B   Select points of this grid
              --replicate-id R           Independent repeat; seed = seed_base + 10000*(R-1) + state_id
              --seed-base S              Default 921000000
              --burnin B --sweeps P --sample-every S   Default 5000 + 10000, every 10
              --output-root DIR          Default ./Data; always creates a fresh batch directory
            """)
            return nothing
        elseif arg == "--dry-run"
            dry = true
        elseif arg in allowed
            haskey(values, arg) && error("Duplicate option: $arg")
            i < length(args) || error("Missing value for $arg")
            i += 1; values[arg] = args[i]
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    integer(key, default) = parse(Int, get(values, key, string(default)))
    flt(key, default) = parse(Float64, get(values, key, string(default)))
    init = Symbol(get(values, "--init", "hot"))
    adapt = lowercase(get(values, "--adapt-step", "true"))
    adapt in ("true", "false") || error("--adapt-step must be true or false")
    rhos = Tuple(parse.(Float64, split(get(values, "--rho", DEFAULT_RHO), ",")))
    Ts = parse_range(get(values, "--T-range", DEFAULT_T_RANGE))
    E0s = parse_range(get(values, "--E0-range", DEFAULT_E0_RANGE))
    states = state_grid(N=integer("--N", 50), rhos=rhos, Ts=Ts, E0s=E0s,
        replicate_id=integer("--replicate-id", 1), seed_base=integer("--seed-base", 921000000))
    if haskey(values, "--state-id")
        !haskey(values, "--start-id") && !haskey(values, "--end-id") || error("Use state-id OR start/end-id")
        a = b = integer("--state-id", 1)
    else
        a = integer("--start-id", 1); b = integer("--end-id", length(states))
    end
    1 <= a <= b <= length(states) || error("State range must satisfy 1 <= start <= end <= $(length(states))")
    settings = merge(SETTINGS, (n_burnin=integer("--burnin", SETTINGS.n_burnin),
        n_sweeps=integer("--sweeps", SETTINGS.n_sweeps),
        sample_every=integer("--sample-every", SETTINGS.sample_every),
        step_size=flt("--step-size", SETTINGS.step_size),
        target_accept=flt("--target-accept", SETTINGS.target_accept),
        adapt_step=(adapt == "true"), init=init))
    run_states(states=states[a:b], settings=settings, ngrid=length(states), dry_run=dry,
        output_root=get(values, "--output-root", get(ENV, "SCAN_OUTPUT_ROOT", joinpath(@__DIR__, "Data"))))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
