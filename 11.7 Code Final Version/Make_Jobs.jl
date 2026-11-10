# From params.toml: write the jobs table next to it, and fill upload/ with everything the cluster needs
# (jobs table, code, .sh, .sub with queue set, empty logs/). Usage is described at the end of params_*.toml.
include("Output_Coordinate_Dielectric.jl")
include("Output_FreeEnergy.jl")
using TOML

const COMMON = ["Project.toml", "Manifest.toml", "Setup.jl", "Plasma_Interaction.jl", "Monte_Carlo.jl"]
const CODE = Dict("Dielectric" => [COMMON; "one_over_epsilon0.jl"; "Output_Coordinate_Dielectric.jl"; "run_Dielectric.sh"],
    "FreeEnergy" => [COMMON; "Free_Energy.jl"; "Output_FreeEnergy.jl"; "run_FreeEnergy.sh"])

# A list, a single number, "start:step:stop", or "log:start:stop:points_per_decade" (3 significant digits).
function axis(v)
    v isa AbstractVector && return Float64.(v)
    v isa Real && return [Float64(v)]
    if startswith(v, "log:")
        a, b, n = parse.(Float64, split(v, ':')[2:end])
        return round.(a .* 10 .^ ((0:floor(Int, n * log10(b / a) + 1e-9)) ./ n); sigdigits=3)
    end
    a, h, b = parse.(Float64, split(v, ':'))
    return round.(a .+ h .* (0:round(Int, (b - a) / h)); digits=12)
end

function make_jobs(path)
    p = TOML.parsefile(path)
    N, reps, driver = p["N"], p["replicates"], p["driver"]
    dir = dirname(abspath(path))
    up = joinpath(dir, "upload")
    mkpath(joinpath(up, "logs"))
    if driver == "Dielectric"
        rows = pack_jobs(N, axis(p["rho"]), axis(p["T"]), axis(p["E0"]), reps, get(p, "seed_base", default_seed_base(N)))
        write_jobs(joinpath(dir, "Dielectric_jobs.csv"), rows)
        n_jobs = maximum(r.job for r in rows) + 1
    else
        jobs = Output_FreeEnergy.pack_jobs(N, Float64(p["T"]), Float64(p["E0"]), axis(p["densities"]), reps)
        Output_FreeEnergy.write_jobs(joinpath(dir, "FreeEnergy_jobs.csv"), jobs)
        n_jobs = length(jobs)
        println("Wrote $(length(jobs)) jobs, longest job $(round(maximum(j.estimated_hours for j in jobs); digits=2)) h")
    end
    foreach(f -> cp(joinpath(@__DIR__, f), joinpath(up, f); force=true), CODE[driver])
    cp(joinpath(dir, "$(driver)_jobs.csv"), joinpath(up, "$(driver)_jobs.csv"); force=true)
    sub = replace(read(joinpath(@__DIR__, "$driver.sub"), String), r"\nqueue 1\n" => "\nqueue $n_jobs\n")
    write(joinpath(up, "$driver.sub"), sub)
    println("Ready ($n_jobs jobs). Upload the contents of: $up")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    make_jobs(ARGS[1])
end
