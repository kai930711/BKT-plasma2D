# Regression/sanity tests for the modified Monte_Carlo.jl (run: julia --project=. test_monte_carlo.jl)
module Orig
include("Monte_Carlo.orig.jl")
end
include("Monte_Carlo.jl")
using Test

N, rho = 50, 0.05
a = sqrt(rho / N)
args(T) = (N, 1/T, a, sqrt(π), 1, 6, 0.05, 200, 400, 10)
kw = (E0_over_T = 0.25/0.2, ξF_over_ξ = 10.0, interaction = :both, seed = 921000727)

@testset "defaults reproduce the original MC exactly" begin
    o = Orig.Monte_Carlo.run_mc_one_state(args(0.2)...; kw...)
    n = Monte_Carlo.run_mc_one_state(args(0.2)...; kw...)
    @test o.snapshots == n.snapshots
    @test o.U1_trace == n.U1_trace && o.U2_trace == n.U2_trace && o.beta_U_trace == n.beta_U_trace
    @test o.n_accept == n.n_accept && o.n_trials == n.n_trials
    @test n.step_size_final == 0.05 && n.init == :hot
end

@testset "cold start is neutral, non-overlapping, compact" begin
    q, R = Monte_Carlo.cold_condition(N, a)
    @test sum(q) == 0 && length(q) == N && all(abs.(q) .== 1)
    dmin = minimum(sqrt(Monte_Carlo.ST.d²(R[i,1],R[i,2],R[j,1],R[j,2])) for i in 1:N for j in i+1:N)
    @test dmin >= a
    @test maximum(R[:,1]) - minimum(R[:,1]) < 0.5      # one compact patch, not spread over the box
    r = Monte_Carlo.run_mc_one_state(args(0.2)...; kw..., init=:cold)
    @test all(isfinite, r.beta_U_trace) && r.init == :cold
    println("cold start: min distance/a = ", round(dmin/a, digits=3), ", initial beta_U ≈ ", round(r.beta_U_trace[1], digits=2))
end

@testset "adaptive step moves toward the target and is frozen after burn-in" begin
    cold_T = 0.06                                   # low T*: fixed step 0.05 would barely accept
    fixed = Monte_Carlo.run_mc_one_state((N, 1/cold_T, a, sqrt(π), 1, 6, 0.05, 600, 400, 10)...; kw..., E0_over_T=0.0, interaction=:log)
    adapt = Monte_Carlo.run_mc_one_state((N, 1/cold_T, a, sqrt(π), 1, 6, 0.05, 600, 400, 10)...; kw..., E0_over_T=0.0, interaction=:log, adapt_step=true)
    accp(r) = r.n_accept_prod / r.n_trials_prod
    println("T*=0.06: fixed step 0.05 -> acc_prod=", round(accp(fixed), digits=3),
            " | adaptive -> step_final=", round(adapt.step_size_final, digits=4), ", acc_prod=", round(accp(adapt), digits=3))
    @test adapt.step_size_final != 0.05
    @test accp(adapt) > accp(fixed)
end
