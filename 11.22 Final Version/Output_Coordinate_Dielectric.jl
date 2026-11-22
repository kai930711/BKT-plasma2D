# Raw-configuration scans (11.7 final code). Run from this directory with:
# julia --project=. Output_Coordinate_Dielectric.jl --write-jobs --N 50 --rho 0.001,0.01 --T 0.05,0.1 --E0 0,0.25 --replicates 2
# julia --project=. Output_Coordinate_Dielectric.jl --job J --output-root DIR
# Options: --jobs-file PATH (default Dielectric_jobs.csv); --seed-base S (write-jobs);
# smoke tests only: --burnin B --sweeps P --sample-every K (override all points of the job).
# Output schema and binary array layout are described in each batch's metadata.toml.
include("Monte_Carlo.jl")
include("one_over_epsilon0.jl")
using Printf, Dates, SHA, TOML
const MC = Monte_Carlo
const OE = OneOverEpsilon0
const SHELLS = OE.first_ten_shell_vectors()
const N_FRAMES = 100
const N_SERIES = 23

# Seed lanes 7–9 (N=50/100/200); lanes 0–3 are 10.5 and 4–6 are 11.3.
default_seed_base(N::Int) = 1_050_000_000 + 10_000_000_000*(6 + findfirst(==(N),(50,100,200)))
const SETTINGS = (alpha=4.0, n_cut=1, m_cut=5, step_size=0.05, xiF_over_xi=5.0,
    cluster_fraction=0.02, cluster_cutoff=1.5, cluster_step=0.5)
# (burn-in, production, sample interval) in sweeps; 11.7_final_code_review/sweep_length.md.
run_length(E0) = iszero(E0) ? (5_000, 10_000, 10) : (20_000, 400_000, 200)
frame_indices(n) = (1:max(1, n ÷ N_FRAMES):n)[1:min(n, N_FRAMES)]
# Seconds per sweep: fit to N=50 (T016, 1.25 ms) and N=200 (11.7 local, 26 ms); to be calibrated on the cluster.
seconds_per_sweep(N) = 1.25e-3*(N/50)^2.2
const MAX_JOB_HOURS = 3.0
const JOB_HEADER = "job,state_id,replicate_id,N,rho_star,T_star,E0_star,seed,n_burnin,n_sweeps,sample_every,estimated_hours"

state(id,rep,N,rho,T,E0,seed,b,p,every) = (state_id=id,run_id=@sprintf("N%d_s%04d_r%04d",N,id,rep),
    replicate_id=rep,N=N,rho_star=rho,T_star=T,E0_star=E0,seed=seed,n_burnin=b,n_sweeps=p,sample_every=every,
    n_samples=cld(p,every),n_frames=length(frame_indices(cld(p,every))))

function pack_jobs(N,rhos,temperatures,strengths,replicates,seed_base)
    grid=[(rho,T,E0) for rho in rhos for T in temperatures for E0 in strengths]
    rows=NamedTuple[]; job=-1
    for rep in 1:replicates
        hours=Inf
        # Short and long runs are packed separately, each in state_id order.
        for id in sortperm([sum(run_length(E0)[1:2]) for (_,_,E0) in grid]; alg=MergeSort)
            rho,T,E0=grid[id]
            b,p,every=run_length(E0)
            cost=(b+p)*seconds_per_sweep(N)*1.05/3600
            if hours+cost>MAX_JOB_HOURS
                job+=1; hours=0.0
            end
            hours+=cost
            push!(rows,(job=job,state_id=id,replicate_id=rep,N=N,rho_star=rho,T_star=T,E0_star=E0,
                seed=seed_base+10000*(rep-1)+id,n_burnin=b,n_sweeps=p,sample_every=every,estimated_hours=cost))
        end
    end
    return rows
end

function write_jobs(path,rows)
    open(path,"w") do io
        println(io,JOB_HEADER)
        foreach(r->println(io,join(values(r),',')),rows)
    end
    hours=Dict{Int,Float64}()
    foreach(r->hours[r.job]=get(hours,r.job,0.0)+r.estimated_hours,rows)
    println("Wrote $path: $(length(rows)) runs, $(length(hours)) jobs, longest job $(round(maximum(values(hours));digits=2)) h")
    for (b,p,every) in unique((r.n_burnin,r.n_sweeps,r.sample_every) for r in rows)
        println("Length: burn-in $b + production $p sweeps, sample every $every, $(count(r->r.n_sweeps==p,rows)) runs")
    end
    over=count(>(MAX_JOB_HOURS),values(hours))
    over>0 && println("Note: $over single-run jobs exceed $(MAX_JOB_HOURS) h")
end

function read_job(path,job,override)
    states=NamedTuple[]
    for line in readlines(path)[2:end]
        f=Dict(zip(split(JOB_HEADER,','),split(line,',')))
        parse(Int,f["job"])==job || continue
        i(k)=parse(Int,f[k]); x(k)=parse(Float64,f[k])
        b,p,every=isnothing(override) ? (i("n_burnin"),i("n_sweeps"),i("sample_every")) : override
        push!(states,state(i("state_id"),i("replicate_id"),i("N"),x("rho_star"),x("T_star"),x("E0_star"),
            i("seed"),b,p,every))
    end
    return states
end

function csvrow(io,values)
    println(io,join(("\""*replace(string(v),"\""=>"\"\"")*"\"" for v in values),","))
end

function source_hashes()
    files=(basename(@__FILE__),"Monte_Carlo.jl","Setup.jl","Plasma_Interaction.jl",
        "one_over_epsilon0.jl","Project.toml","Manifest.toml")
    return Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
end

function series_matrix(result)
    shell2(q) = OE.shell_abs_Q2_series(OE.measure_fourier_series(result.snapshots, q; shells=SHELLS).Q; shells=SHELLS)
    return hcat(result.U1_trace, result.U2_trace, result.beta_U_trace, shell2(ones(length(result.q))), shell2(result.q))
end

function write_plan(out,states,c,job)
    hashes=source_hashes()
    version=bytes2hex(sha256(join((f*":"*hashes[f] for f in sort(collect(keys(hashes)))),"\n")))
    # Native Float64 binary; byte order is explicit so Python/other platforms can read it.
    endian=ENDIAN_BOM==0x04030201 ? "little" : "big"
    metadata=Dict(
        "schema_version"=>2,"created_at_utc"=>string(now(UTC)),
        "julia_version"=>string(VERSION),"code_version_sha256"=>version,
        "source_sha256"=>hashes,"job"=>job,"n_states"=>length(states),"particle_count"=>first(states).N,
        "rho_star_values"=>sort(unique(s.rho_star for s in states)),"T_star_values"=>sort(unique(s.T_star for s in states)),
        "E0_star_values"=>sort(unique(s.E0_star for s in states)),
        "coordinate_dtype"=>"Float64","byte_order"=>endian,
        "coordinates_file"=>"coordinates.f64","series_file"=>"series.f64","charges_file"=>"charges.f64",
        "coordinate_storage"=>"For each frame: x[1:N], then y[1:N]; n_frames frames per run, taken at samples 1, 1+s, 1+2s, ... with s = max(1, n_samples ÷ 100). No header.",
        "series_storage"=>"For each sample, 23 Float64 in this order: U1, U2, beta_U, |rho_m|^2 for shells 1..10, |Q_m|^2 for shells 1..10; samples in production order. No header.",
        "shell_m2"=>SHELLS.shell_m2,"shell_degeneracy"=>SHELLS.degeneracies,
        "charge_storage"=>"One Float64 q[1:N] array per run. Same particle order as coordinates.",
        "offset_units"=>"bytes, zero-based from beginning of each binary file",
        "boundaries"=>"periodic square; wrapped coordinates in [0,1); Lx=Ly=1",
        "state_order"=>"rho outermost, T next, E0 innermost in the jobs table; state_id is global over the scan",
        "seed_rule"=>"seed_base + 10000*(replicate_id-1) + state_id; default seed_base lane 7/8/9 for N=50/100/200",
        "samples_policy"=>"Production sweeps 1, 1+sample_every, ...; no burn-in configurations. Lengths per state in states.csv.",
        "recovery"=>"Each parameter point is exported only after the MC function returns. Only complete rows in progress.csv mark valid runs.",
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
            "xiF_over_L"=>"xiF_over_xi*a",
            "charge"=>"dimensionless +/-1, in units of charge magnitude q; fixed during a run",
            "U1"=>"dimensionless Coulomb Ewald energy including self and polarization terms",
            "U2"=>"sum over i<j of -logcosh((E0/(k_B*T))*exp(-r/xiF)) over all periodic images (minimum image exact, other images from a cubic-spline table, total error <= 5e-5), plus the self-image constant U2_self; 0 when off",
            "U2_self"=>"(N/2)*sum over n!=0 of -logcosh((E0/(k_B*T))*exp(-|n|L/xiF)); constant per run, in progress.csv",
            "image_table"=>"true if the Majorana image table was built; false if minimum image alone meets the 5e-5 bound",
            "beta_U"=>"kappa*U1+U2; dimensionless effective sampling energy",
            "rho_m, Q_m"=>"rho_m=sum exp(2pi*i*m.R_j), Q_m=sum q_j exp(2pi*i*m.R_j); |.|^2 averaged over the directions of each shell",
            "S_NN, S_QQ"=>"S_NN(k)=<|rho_m|^2>/N, S_QQ(k)=<|Q_m|^2>/N with kL=2pi*sqrt(m^2)",
            "alpha"=>"Ewald alpha in unit-box coordinates",
            "n_cut"=>"real-space image index cutoff in each direction",
            "m_cut"=>"reciprocal Ewald integer cutoff per axis",
            "step_size"=>"per-coordinate uniform displacement half-width in box units",
            "cluster_moves"=>"O&P 1994 rigid cluster translations per sweep, cutoff in units of a, step in box units",
            "sweep"=>"N attempted single-particle moves, particle chosen randomly with replacement, then cluster moves; not physical time",
            "attempt_counts"=>"Include moves rejected by hard-core overlap; counts include burn-in and production"))
    open(joinpath(out,"metadata.toml"),"w") do io
        TOML.print(io,metadata;sorted=true)
    end
    coord_offset=0; series_offset=0; charge_offset=0
    open(joinpath(out,"states.csv"),"w") do io
        csvrow(io,["state_id","run_id","replicate_id","N","rho_star","Lx","Ly","a",
            "T_star","kappa","E0_star","E0_over_T","xiF_over_xi","xiF_over_L","interaction",
            "alpha","n_cut","m_cut","step_size","cluster_moves_per_sweep","cluster_cutoff","cluster_step",
            "n_burnin","n_sweeps","sample_every","n_samples_planned","n_frames_planned","seed","initial_condition",
            "coordinates_offset_bytes","series_offset_bytes","charges_offset_bytes","code_version_sha256"])
        for s in states
            a=sqrt(s.rho_star/s.N)
            csvrow(io,(s.state_id,s.run_id,s.replicate_id,s.N,s.rho_star,1.0,1.0,a,
                s.T_star,1/s.T_star,s.E0_star,s.E0_star/s.T_star,c.xiF_over_xi,c.xiF_over_xi*a,
                iszero(s.E0_star) ? "log" : "both",c.alpha,c.n_cut,c.m_cut,c.step_size,
                round(Int,c.cluster_fraction*s.N),c.cluster_cutoff,c.cluster_step,
                s.n_burnin,s.n_sweeps,s.sample_every,s.n_samples,s.n_frames,s.seed,"random_nonoverlap",
                coord_offset,series_offset,charge_offset,version))
            coord_offset+=16*s.N*s.n_frames; series_offset+=8*N_SERIES*s.n_samples; charge_offset+=8*s.N
        end
    end
end

function run_states(;states,job,settings=SETTINGS,output_root=joinpath(@__DIR__,"Data"))
    c=settings
    length(unique(s.N for s in states))==1 || error("A batch must contain one particle count")
    println("Job $job: N=$(first(states).N), $(length(states)) runs; first=$(first(states).run_id), last=$(last(states).run_id)")
    println("Settings: ",c)
    mkpath(output_root)
    out=mktempdir(abspath(output_root);prefix="job$(job)_N$(first(states).N)_",cleanup=false)
    write_plan(out,states,c,job)
    println("Output: ",out); flush(stdout)
    open(joinpath(out,"coordinates.f64"),"w") do coords
      open(joinpath(out,"series.f64"),"w") do series
       open(joinpath(out,"charges.f64"),"w") do charges
        open(joinpath(out,"progress.csv"),"w") do progress
         csvrow(progress,["state_id","run_id","status","n_samples","n_frames","n_trials","n_accept",
            "n_cluster_trials","n_cluster_accept","n_cluster_contact","cluster_size_sum","image_table","U2_self",
            "coordinates_end_bytes","series_end_bytes","charges_end_bytes","elapsed_seconds","timestamp_utc"])
         flush(progress)
         for (index,s) in enumerate(states)
            started=time()
            @printf("[%s] START %d/%d %s N=%d rho*=%.4g T*=%.3g E0*=%.3g sweeps=%d+%d\n",
                string(now(UTC)),index,length(states),s.run_id,s.N,s.rho_star,s.T_star,s.E0_star,s.n_burnin,s.n_sweeps)
            flush(stdout)
            try
                result=MC.run_mc_one_state(s.N,1/s.T_star,sqrt(s.rho_star/s.N),c.alpha,c.n_cut,c.m_cut,
                    c.step_size,s.n_burnin,s.n_sweeps,s.sample_every;
                    E0_over_T=s.E0_star/s.T_star,ξF_over_ξ=c.xiF_over_xi,
                    interaction=iszero(s.E0_star) ? :log : :both,seed=s.seed,
                    cluster_moves_per_sweep=round(Int,c.cluster_fraction*s.N),
                    cluster_cutoff=c.cluster_cutoff,cluster_step=c.cluster_step)
                write(charges,result.q)
                # Row-major: the 23 values of one sample are contiguous.
                write(series,permutedims(series_matrix(result)))
                foreach(R->write(coords,R),result.snapshots[frame_indices(result.n_samples)])
                # Publish completion only after this point's complete output is flushed.
                flush(coords); flush(series); flush(charges)
                csvrow(progress,(s.state_id,s.run_id,"complete",result.n_samples,s.n_frames,result.n_trials,
                    result.n_accept,result.n_cluster_trials,result.n_cluster_accept,result.n_cluster_contact,
                    result.cluster_size_sum,result.image_table,result.U2_self,
                    position(coords),position(series),position(charges),time()-started,string(now(UTC))))
                flush(progress)
                @printf("[%s] COMPLETE %d/%d %s samples=%d accepted=%d/%d elapsed=%.1fs\n",
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
    o=Dict{String,String}("--jobs-file"=>joinpath(@__DIR__,"Dielectric_jobs.csv"),"--replicates"=>"1"); write=false; i=1
    while i<=length(args)
        if args[i]=="--write-jobs"
            write=true; i+=1
        else
            o[args[i]]=args[i+1]; i+=2
        end
    end
    values(k)=parse.(Float64,split(o[k],','))
    if write
        N=parse(Int,o["--N"])
        seed_base=haskey(o,"--seed-base") ? parse(Int,o["--seed-base"]) : default_seed_base(N)
        write_jobs(o["--jobs-file"],pack_jobs(N,values("--rho"),values("--T"),values("--E0"),parse(Int,o["--replicates"]),seed_base))
    end
    if haskey(o,"--job")
        override=haskey(o,"--sweeps") ? (parse(Int,get(o,"--burnin","0")),parse(Int,o["--sweeps"]),parse(Int,get(o,"--sample-every","10"))) : nothing
        job=parse(Int,o["--job"])
        run_states(states=read_job(o["--jobs-file"],job,override),job=job,
            output_root=get(o,"--output-root",get(ENV,"SCAN_OUTPUT_ROOT",joinpath(@__DIR__,"Data"))))
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    main()
end
