module Free_Energy

using LinearAlgebra: SymTridiagonal, eigen, dot, norm
using Statistics: mean, std

export gauss_legendre_rule, coulomb_scale_correction, hard_disk_beta_f,
    block_statistics, lambda_statistics, estimate_free_energy

# Quadrature: Golub & Welsch (1969), doi:10.1090/S0025-5718-69-99647-1, Eq. (2.6).
# HD reference: Rovigatti et al. (2013), doi:10.1063/1.4802026, Eqs. (9), (11).
# Reference-state integration: Fortini et al., arXiv:cond-mat/0607442, Sec. III.
# MC errors below exclude quadrature bias, HD-reference error, and incomplete equilibration.

"""Gauss-Legendre nodes and weights on [0,1], obtained by the Golub-Welsch method."""
function gauss_legendre_rule(n::Integer=10)
    n > 0 || throw(ArgumentError("The number of quadrature nodes must be positive."))
    offdiag = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    decomposition = eigen(SymTridiagonal(zeros(n), offdiag))
    return (lambdas=(decomposition.values .+ 1.0) ./ 2.0,
        weights=decomposition.vectors[1, :] .^ 2)
end

"""Unit-box to r/sigma energy shift, for a neutral system of N charges +/-1; a=sigma/L."""
function coulomb_scale_correction(N::Integer, a::Real)
    N > 0 && iseven(N) || throw(ArgumentError("N must be positive and even."))
    isfinite(a) && a > 0 || throw(ArgumentError("a must be positive and finite."))
    return -0.5 * N * log(a)
end

"""Bulk-fluid approximation to beta*F_HD/N, including equal-species ideal mixing; not exact at finite N."""
function hard_disk_beta_f(rho_star::Real; thermal_wavelength_over_sigma::Real=1.0)
    isfinite(rho_star) && rho_star > 0 || throw(ArgumentError("rho_star must be positive and finite."))
    wavelength = thermal_wavelength_over_sigma
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError("The thermal wavelength must be positive and finite."))
    eta = π * rho_star / 4
    eta < 1 || throw(ArgumentError("The Henderson expression requires packing fraction eta < 1."))
    ideal = log(rho_star) + 2log(wavelength) - log(2.0) - 1.0
    excess = -(7 / 8) * log1p(-eta) + (9 / 8) * eta / (1 - eta)
    return ideal + excess
end

"""Equal contiguous blocks; SEM assumes that block means are effectively independent."""
function block_statistics(values::AbstractVector{<:Real}; n_blocks::Integer=5)
    n = length(values)
    2 <= n_blocks <= n || throw(ArgumentError("Require 2 <= n_blocks <= number of samples."))
    n % n_blocks == 0 || throw(ArgumentError("The number of samples must be divisible by n_blocks."))
    all(isfinite, values) || throw(ArgumentError("Samples must be finite."))
    block_size = n ÷ n_blocks
    blocks = [mean(view(values, ((b - 1) * block_size + 1):(b * block_size))) for b in 1:n_blocks]
    block_std = std(blocks; corrected=true)
    return (mean=mean(blocks), sem=block_std / sqrt(n_blocks), block_means=blocks,
        block_std=block_std, n_samples=n, n_blocks=n_blocks, block_size=block_size)
end

"""Summarize d(beta*F)/d(lambda) from one Monte_Carlo_Lambda result; kappa is the target coupling."""
function lambda_statistics(result; kappa::Real, rho_star::Real, n_blocks::Integer=5)
    kappa, rho_star = Float64(kappa), Float64(rho_star)
    isfinite(kappa) && kappa >= 0 || throw(ArgumentError("kappa must be nonnegative and finite."))
    isfinite(rho_star) && rho_star > 0 || throw(ArgumentError("rho_star must be positive and finite."))
    N = length(result.q)
    N > 0 && iseven(N) && all(q -> abs(q) == 1, result.q) && sum(result.q) == 0 ||
        throw(ArgumentError("The result must contain equal numbers of charges +1 and -1."))
    isfinite(result.lambda) && 0 <= result.lambda <= 1 || throw(ArgumentError("Invalid lambda in result."))
    length(result.U1_trace) == length(result.U2_trace) == length(result.beta_U_trace) == result.n_samples ||
        throw(ArgumentError("Energy trace lengths do not match n_samples."))
    target = kappa .* result.U1_trace .+ result.U2_trace
    isapprox(target, result.beta_U_trace; atol=1e-8, rtol=1e-9) ||
        throw(ArgumentError("Energy traces do not match the supplied target kappa."))
    a = sqrt(rho_star / N)
    shift = result.include_log ? kappa * coulomb_scale_correction(N, a) : 0.0
    stats = block_statistics(target .+ shift; n_blocks=n_blocks)
    return merge(stats, (N=N, rho_star=rho_star, kappa=kappa, interaction=result.interaction,
        lambda=Float64(result.lambda), a=a, coulomb_beta_shift=shift))
end

"""
Combine one lambda_statistics summary per Gaussian node. Node simulations must be independent.
Keep Majorana parameters, Ewald settings, and boundaries fixed across the nodes.
Returns beta*F and beta*F/N; beta_f_reference can replace the default bulk Henderson reference.
The reported MC SEM excludes reference uncertainty and quadrature error.
"""
function estimate_free_energy(nodes::AbstractVector;
        beta_f_reference::Union{Nothing,Real}=nothing, thermal_wavelength_over_sigma::Real=1.0)
    isempty(nodes) && throw(ArgumentError("At least one node summary is required."))
    ordered = sort(collect(nodes); by=node -> node.lambda)
    rule = gauss_legendre_rule(length(ordered))
    first_node = first(ordered)
    N, rho_star, kappa = first_node.N, first_node.rho_star, first_node.kappa
    all(node -> node.N == N && node.rho_star == rho_star && node.kappa == kappa &&
        node.interaction == first_node.interaction, ordered) ||
        throw(ArgumentError("All nodes must describe the same target state and interaction."))
    all(i -> isapprox(ordered[i].lambda, rule.lambdas[i]; atol=1e-12, rtol=1e-10), eachindex(ordered)) ||
        throw(ArgumentError("Node lambdas do not match the complete Gauss-Legendre rule."))
    means, sems = [node.mean for node in ordered], [node.sem for node in ordered]
    all(isfinite, means) && all(x -> isfinite(x) && x >= 0, sems) ||
        throw(ArgumentError("Node means and nonnegative MC SEMs must be finite."))
    reference_kind = isnothing(beta_f_reference) ? :henderson_bulk_fluid : :provided_beta_f
    reference = isnothing(beta_f_reference) ? hard_disk_beta_f(rho_star;
        thermal_wavelength_over_sigma=thermal_wavelength_over_sigma) : Float64(beta_f_reference)
    isfinite(reference) || throw(ArgumentError("The reference free energy must be finite."))
    beta_delta_F = dot(rule.weights, means)
    beta_F_mc_sem = norm(rule.weights .* sems)
    beta_F_reference = N * reference
    beta_F = beta_F_reference + beta_delta_F
    return (N=N, rho_star=rho_star, eta=π * rho_star / 4, v_star=inv(rho_star),
        kappa=kappa, interaction=first_node.interaction, n_lambda=length(ordered),
        lambdas=rule.lambdas, weights=rule.weights, nodes=ordered,
        beta_delta_F=beta_delta_F, beta_F_reference=beta_F_reference,
        beta_F=beta_F, beta_F_mc_sem=beta_F_mc_sem,
        beta_f=beta_F / N, beta_f_mc_sem=beta_F_mc_sem / N,
        beta_free_energy_density_star=rho_star * beta_F / N,
        reference_kind=reference_kind,
        reference_thermal_wavelength_over_sigma=isnothing(beta_f_reference) ? thermal_wavelength_over_sigma : nothing)
end

end
