# 1210-point raw-configuration scan. Run from any working directory.
# julia --project=9.21 9.21/Task_Animation_Dielectric.jl --dry-run
# --start-id/--end-id select a cluster batch; --replicate-id selects an independent seed.
# Output schema and binary array layout are described in each batch's metadata.toml.
include("Monte_Carlo.jl")
using Printf, Dates, SHA, TOML
const MC = Monte_Carlo
const RHO_VALUES = (0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2)
const T_VALUES = Tuple(i/100 for i in 15:2:35)
const E0_VALUES = Tuple(i/100 for i in 0:5:50)
# Provisional pilot settings, not a claim of equilibration or independent frames.
const SETTINGS = (alpha=sqrt(π), n_cut=1, m_cut=6, step_size=0.05,
    n_burnin=5000, n_sweeps=10000, sample_every=10,
    xiF_over_xi=10.0)

function state_grid(;replicate_id::Int=1, seed_base::Int=921000000)
    1<=replicate_id<=1000000 || error("replicate-id must be 1..1000000")
    seed_base>=0 || error("seed-base must be nonnegative")
    states=NamedTuple[]
    for rho in RHO_VALUES, T in T_VALUES, E0 in E0_VALUES
        id=length(states)+1
        seed=Base.checked_add(seed_base, Base.checked_add(Base.checked_mul(10000,replicate_id-1),id))
        push!(states,(state_id=id,run_id=@sprintf("s%04d_r%04d",id,replicate_id),
            replicate_id=replicate_id,N=50,rho_star=rho,T_star=T,E0_star=E0,seed=seed))
    end
    return states
end

function csvrow(io,values)
    println(io,join(("\""*replace(string(v),"\""=>"\"\"")*"\"" for v in values),","))
end

function source_hashes()
    files=("Task_Animation_Dielectric.jl","Monte_Carlo.jl","Setup.jl",
        "Plasma_Interaction.jl","Project.toml","Manifest.toml")
    return Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
end

function write_plan(out,states,c)
    hashes=source_hashes()
    version=bytes2hex(sha256(join((f*":"*hashes[f] for f in sort(collect(keys(hashes)))),"\n")))
    # Native Float64 binary; byte order is explicit so Python/other platforms can read it.
    endian=ENDIAN_BOM==0x04030201 ? "little" : "big"
    metadata=Dict(
        "schema_version"=>1,"created_at_utc"=>string(now(UTC)),
        "julia_version"=>string(VERSION),"code_version_sha256"=>version,
        "source_sha256"=>hashes,"n_selected_states"=>length(states),"n_grid_states"=>1210,
        "coordinate_dtype"=>"Float64","byte_order"=>endian,
        "coordinates_file"=>"coordinates.f64","charges_file"=>"charges.f64",
        "coordinate_storage"=>"For each frame: x[1:N], then y[1:N]; frames in sampling order. No header.",
        "charge_storage"=>"One Float64 q[1:N] array per run. Same particle order as coordinates.",
        "offset_units"=>"bytes, zero-based from beginning of each binary file",
        "frame_bytes"=>"16*N",
        "boundaries"=>"periodic square; wrapped coordinates in [0,1); Lx=Ly=1",
        "state_order"=>"rho outermost, T next, E0 innermost; IDs remain global when selecting a batch",
        "seed_rule"=>"seed_base + 10000*(replicate_id-1) + state_id",
        "samples_policy"=>"Production sweeps 1, 1+sample_every, ... <= n_sweeps; no burn-in configurations.",
        "recovery"=>"Each parameter point is exported only after the unchanged MC function returns. Only complete rows in progress.csv mark valid runs. Interrupted MC points have no saved frames; earlier completed points remain readable.",
        "progress_policy"=>"stdout reports START and COMPLETE per parameter point; no within-point sweep progress",
        "restart"=>"No exact restart checkpoint or RNG state is stored. Reruns use a new output directory.",
        "definitions"=>Dict(
            "rho_star"=>"N*sigma^2/L^2; N counts both charge species",
            "a"=>"sigma/L=sqrt(rho_star/N); hard-core exclusion distance in unit-box coordinates",
            "T_star"=>"k_B*T/K; K is the Coulomb logarithmic energy scale",
            "kappa"=>"K/(k_B*T)=1/T_star",
            "E0_star"=>"E0/K",
            "E0_over_T"=>"E0/(k_B*T)=E0_star/T_star",
            "xiF_over_xi"=>"xiF/sigma in this implementation (xi is identified with hard-core diameter)",
            "xiF_over_L"=>"xiF_over_xi*a; Majorana distances use minimum image",
            "charge"=>"dimensionless +/-1, in units of charge magnitude q; fixed during a run",
            "U1"=>"dimensionless Coulomb Ewald energy including self and polarization terms",
            "U2"=>"sum over i<j of -logcosh((E0/(k_B*T))*exp(-r_ij/xiF)); already dimensionless; 0 when off",
            "beta_U"=>"kappa*U1+U2; dimensionless effective sampling energy",
            "alpha"=>"Ewald alpha in unit-box coordinates",
            "n_cut"=>"real-space image index cutoff in each direction",
            "m_cut"=>"reciprocal Ewald integer cutoff per axis; not a postprocessing Fourier cutoff",
            "step_size"=>"per-coordinate uniform displacement half-width in box units",
            "sweep"=>"N attempted single-particle moves, particle chosen randomly with replacement; not physical time",
            "attempt_counts"=>"Include moves rejected by hard-core overlap. Counts include both burn-in and production; the original MC function does not return separate phase acceptance counts.",
            "postprocessing"=>"N_m=sum exp(2pi*i*m.R_i), Q_m=sum q_i*exp(2pi*i*m.R_i); no fixed normalization or binning is imposed by storage."))
    open(joinpath(out,"metadata.toml"),"w") do io
        TOML.print(io,metadata;sorted=true)
    end
    ns=cld(c.n_sweeps,c.sample_every)
    coord_offset=0; charge_offset=0
    open(joinpath(out,"states.csv"),"w") do io
        csvrow(io,["state_id","run_id","replicate_id","N","rho_star","Lx","Ly","a",
            "T_star","kappa","E0_star","E0_over_T","xiF_over_xi","xiF_over_L","interaction",
            "alpha","n_cut","m_cut","step_size","n_burnin","n_sweeps","sample_every",
            "seed","initial_condition","n_samples_planned","coordinates_offset_bytes",
            "charges_offset_bytes","code_version_sha256"])
        for s in states
            a=sqrt(s.rho_star/s.N)
            csvrow(io,(s.state_id,s.run_id,s.replicate_id,s.N,s.rho_star,1.0,1.0,a,
                s.T_star,1/s.T_star,s.E0_star,s.E0_star/s.T_star,c.xiF_over_xi,c.xiF_over_xi*a,
                iszero(s.E0_star) ? "log" : "both",c.alpha,c.n_cut,c.m_cut,c.step_size,
                c.n_burnin,c.n_sweeps,c.sample_every,s.seed,"random_nonoverlap",
                ns,coord_offset,charge_offset,version))
            coord_offset+=16*s.N*ns; charge_offset+=8*s.N
        end
    end
end

function run_states(;states=state_grid(),settings=SETTINGS,
        output_root=joinpath(@__DIR__,"Data"),dry_run=false)
    c=settings
    isempty(states) && error("No states selected")
    length(unique(s.state_id for s in states))==length(states) || error("Duplicate state IDs")
    length(unique(s.run_id for s in states))==length(states) || error("Duplicate run IDs")
    c.n_burnin>=0 && c.n_sweeps>0 && c.sample_every>0 || error("Invalid sweep settings")
    isfinite(c.alpha) && c.alpha>0 && c.n_cut>=0 && c.m_cut>=1 &&
        isfinite(c.step_size) && c.step_size>0 && isfinite(c.xiF_over_xi) && c.xiF_over_xi>0 || error("Invalid settings")
    for s in states
        s.N>0 && iseven(s.N) && isfinite(s.rho_star) && s.rho_star>0 &&
            isfinite(s.T_star) && s.T_star>0 && isfinite(s.E0_star) && s.E0_star>=0 || error("Invalid state")
    end
    ns=cld(c.n_sweeps,c.sample_every)
    println("Grid: 10 densities x 11 temperatures x 11 strengths = 1210 parameter points")
    println("Selected: $(length(states)); first=$(first(states).run_id), last=$(last(states).run_id)")
    println("Settings: ",c,"; planned production frames/run=",ns)
    println("Majorana OFF for E0_star=0; raw configurations only, correlations are postprocessed.")
    if dry_run
        length(states)<=12 && foreach(println,states)
        println("Dry run: no files written and no simulation started.")
        return nothing
    end
    mkpath(output_root)
    out=mktempdir(abspath(output_root);prefix="scan_921_",cleanup=false)
    write_plan(out,states,c)
    println("Output: ",out); flush(stdout)
    open(joinpath(out,"coordinates.f64"),"w") do coords
      open(joinpath(out,"charges.f64"),"w") do charges
       open(joinpath(out,"samples.csv"),"w") do samples
        open(joinpath(out,"progress.csv"),"w") do progress
         csvrow(samples,["state_id","run_id","sample","production_sweep","total_sweep",
            "U1","U2","beta_U"])
         csvrow(progress,["state_id","run_id","status","phase","total_sweep","production_sweep",
            "n_samples_committed","n_trials","n_accept","coordinates_end_bytes","charges_end_bytes",
            "samples_start_bytes","samples_end_bytes","elapsed_seconds","timestamp_utc"])
         flush(samples); flush(progress)
         for (index,s) in enumerate(states)
            started=time(); samples_start=position(samples)
            @printf("[%s] START %d/%d %s rho*=%.4g T*=%.3g E0*=%.3g\n",
                string(now(UTC)),index,length(states),s.run_id,s.rho_star,s.T_star,s.E0_star)
            flush(stdout)
            try
                # Use only the original Monte_Carlo.jl API and its existing return values.
                result=MC.run_mc_one_state(s.N,1/s.T_star,sqrt(s.rho_star/s.N),c.alpha,c.n_cut,
                    c.m_cut,c.step_size,c.n_burnin,c.n_sweeps,c.sample_every;
                    E0_over_T=s.E0_star/s.T_star,ξF_over_ξ=c.xiF_over_xi,
                    interaction=iszero(s.E0_star) ? :log : :both,seed=s.seed)
                result.n_samples==ns==length(result.snapshots)==length(result.U1_trace)==
                    length(result.U2_trace)==length(result.beta_U_trace) || error("Unexpected sample count")
                write(charges,result.q)
                for (i,R) in enumerate(result.snapshots)
                    # Julia column-major N x 2: all x followed by all y for each frame.
                    write(coords,R)
                    sweep=1+(i-1)*c.sample_every
                    csvrow(samples,(s.state_id,s.run_id,i,sweep,c.n_burnin+sweep,
                        result.U1_trace[i],result.U2_trace[i],result.beta_U_trace[i]))
                end
                # Publish completion only after this point's complete output is flushed.
                flush(coords); flush(charges); flush(samples)
                csvrow(progress,(s.state_id,s.run_id,"complete","production",c.n_burnin+c.n_sweeps,
                    c.n_sweeps,result.n_samples,result.n_trials,result.n_accept,
                    position(coords),position(charges),samples_start,position(samples),
                    time()-started,string(now(UTC))))
                flush(progress)
                @printf("[%s] COMPLETE %d/%d %s frames=%d accepted=%d/%d elapsed=%.1fs\n",
                    string(now(UTC)),index,length(states),s.run_id,result.n_samples,
                    result.n_accept,result.n_trials,time()-started)
                flush(stdout)
            catch err
                # Stop here: subsequent planned offsets assume this point was fully exported.
                println(stderr,"FAILED ",s.run_id,". Earlier completed points remain in ",out)
                rethrow()
            end
         end
        end
       end
      end
    end
    println("Finished: ",out)
    return out
end

function main(args=ARGS)
    values=Dict{String,String}(); dry=false
    allowed=("--state-id","--start-id","--end-id","--replicate-id","--seed-base",
        "--burnin","--sweeps","--sample-every","--output-root")
    i=1
    while i<=length(args)
        arg=args[i]
        if arg=="--help"
            println("""
            julia --project=. Task_Animation_Dielectric.jl [options]
              --dry-run                     Validate/print plan without simulating or writing
              --state-id ID                 One global parameter point (1..1210)
              --start-id A --end-id B        Inclusive batch of global parameter points
              --replicate-id R              Independent repeat (default 1)
              --seed-base S                 Default 921000000
              --burnin B --sweeps P          Default 5000 + 10000 (pilot settings)
              --sample-every S              Default 10; production starts sampling at sweep 1
              --output-root DIR             Default 9.21/Data; always creates a fresh batch directory
            """)
            return nothing
        elseif arg=="--dry-run"
            dry=true
        elseif arg in allowed
            haskey(values,arg) && error("Duplicate option: $arg")
            i<length(args) || error("Missing value for $arg")
            i+=1; values[arg]=args[i]
        else
            error("Unknown option: $arg")
        end
        i+=1
    end
    integer(key,default)=parse(Int,get(values,key,string(default)))
    states=state_grid(replicate_id=integer("--replicate-id",1),seed_base=integer("--seed-base",921000000))
    if haskey(values,"--state-id")
        !haskey(values,"--start-id") && !haskey(values,"--end-id") || error("Use state-id OR start/end-id")
        a=b=integer("--state-id",1)
    else
        a=integer("--start-id",1); b=integer("--end-id",length(states))
    end
    1<=a<=b<=length(states) || error("State range must satisfy 1 <= start <= end <= 1210")
    settings=merge(SETTINGS,(n_burnin=integer("--burnin",SETTINGS.n_burnin),
        n_sweeps=integer("--sweeps",SETTINGS.n_sweeps),
        sample_every=integer("--sample-every",SETTINGS.sample_every)))
    run_states(states=states[a:b],settings=settings,dry_run=dry,
        output_root=get(values,"--output-root",get(ENV,"SCAN_OUTPUT_ROOT",joinpath(@__DIR__,"Data"))))
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    main()
end
