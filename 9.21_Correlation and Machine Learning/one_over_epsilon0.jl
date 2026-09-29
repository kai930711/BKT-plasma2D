module OneOverEpsilon0

using Statistics


function charge_fourier_mode(R, q, m)
    N = length(q)

    mx = m[1]
    my = m[2]

    Qm = 0.0 + 0.0im

    @inbounds for j in 1:N
        phase = mx * R[j, 1] + my * R[j, 2]
        Qm += q[j] * cis(2π * phase)
    end

    return Qm
end


function abs_Q2_one_snapshot(R, q, m)
    Qm = charge_fourier_mode(R, q, m)

    return abs2(Qm)
end


function mean_abs_Q2_by_m(snapshots, q, m)
    n_samples = length(snapshots)

    n_samples > 0 || throw(ArgumentError("snapshots cannot be empty."))

    total = 0.0

    @inbounds for R in snapshots
        total += abs_Q2_one_snapshot(R, q, m)
    end

    return total / n_samples
end


function abs_Q2_first_shell_one_snapshot(R, q)
    Qx = charge_fourier_mode(R, q, (1, 0))
    Qy = charge_fourier_mode(R, q, (0, 1))

    abs_Q2_x = abs2(Qx)
    abs_Q2_y = abs2(Qy)

    return 0.5 * (abs_Q2_x + abs_Q2_y)
end


function mean_abs_Q2_first_shell(snapshots, q)
    n_samples = length(snapshots)

    n_samples > 0 || throw(ArgumentError("snapshots cannot be empty."))

    total = 0.0

    @inbounds for R in snapshots
        total += abs_Q2_first_shell_one_snapshot(R, q)
    end

    return total / n_samples
end


function ϵ0_inv_first_shell(
    snapshots,
    q,
    kappa::Float64;
    n_blocks::Integer = 5,
)
    n_samples = length(snapshots)

    n_blocks >= 2 || throw(ArgumentError("n_blocks must be at least 2."))
    n_samples >= n_blocks || throw(ArgumentError("Need at least one snapshot per block."))

    mean_Q2 = mean_abs_Q2_first_shell(snapshots, q)
    inv_eps = 1.0 - kappa * mean_Q2 / (2π)

    inv_eps_blocks = Vector{Float64}(undef, n_blocks)

    for block_index in 1:n_blocks
        first_sample = fld((block_index - 1) * n_samples, n_blocks) + 1
        last_sample = fld(block_index * n_samples, n_blocks)
        block_snapshots = @view snapshots[first_sample:last_sample]
        block_mean_Q2 = mean_abs_Q2_first_shell(block_snapshots, q)

        inv_eps_blocks[block_index] = 1.0 - kappa * block_mean_Q2 / (2π)
    end

    return (
        mean_abs_Q2_first_shell = mean_Q2,
        ϵ0_inv = inv_eps,
        n_blocks = n_blocks,
        ϵ0_inv_blocks = inv_eps_blocks,
        ϵ0_inv_block_std = std(inv_eps_blocks; corrected=true),
    )
end


# ---------------------------------------------------------------------------
# Multi-shell measurements (unit periodic box). Existing functions above are
# unchanged. These functions measure finite-k response; no k→0 fit is performed.
# ---------------------------------------------------------------------------

"""
    first_ten_shell_vectors()

Return the first ten nonzero square-box reciprocal shells. `mvecs` contains ALL
directions, including ±m; `shell_indices[s]` selects shell s in that vector list.
For real charges Q(-m)=conj(Q(m)); retaining both signs makes the saved spectrum
explicit. Equal weighting within each shell gives the same result as averaging
one representative from each ± pair. Wavevectors are kL=2πm.
"""
function first_ten_shell_vectors()
    shell_m2 = [1, 2, 4, 5, 8, 9, 10, 13, 16, 17]
    mvecs = Tuple{Int,Int}[]
    shell_indices = Vector{Vector{Int}}()
    for m2 in shell_m2
        indices = Int[]
        radius = isqrt(m2)
        for mx in -radius:radius, my in -radius:radius
            if mx^2 + my^2 == m2
                push!(mvecs, (mx, my))
                push!(indices, length(mvecs))
            end
        end
        push!(shell_indices, indices)
    end
    return (shell_m2=shell_m2, mvecs=mvecs, shell_indices=shell_indices,
        degeneracies=length.(shell_indices), kL=2π .* sqrt.(shell_m2))
end


"""Measure complex Q_m for every supplied nonzero integer wavevector.
Coordinates R are N×2 in box-length units; no normalization by N is applied.
"""
function charge_fourier_modes(R, q, mvecs)
    ndims(R) == 2 && size(R) == (length(q), 2) ||
        throw(ArgumentError("R must have size (length(q), 2)."))
    all(isfinite, R) && all(isfinite, q) ||
        throw(ArgumentError("Coordinates and charges must be finite."))
    isempty(mvecs) && throw(ArgumentError("mvecs cannot be empty."))
    for m in mvecs
        length(m) == 2 && all(x -> x isa Integer, m) && m != (0, 0) &&
            (m[1] != 0 || m[2] != 0) ||
            throw(ArgumentError("Each mode must be a nonzero integer pair."))
    end
    return ComplexF64[charge_fourier_mode(R, q, m) for m in mvecs]
end


"""
    measure_fourier_series(snapshots, q; shells=first_ten_shell_vectors())

Rows of Q are chronological snapshots; columns correspond to shells.mvecs.
Save real.(Q), imag.(Q) and mvecs to retain direction-resolved measurements.
The series contains no burn-in selection: pass the desired snapshots explicitly.
"""
function measure_fourier_series(snapshots, q; shells=first_ten_shell_vectors())
    isempty(snapshots) && throw(ArgumentError("snapshots cannot be empty."))
    Q = Matrix{ComplexF64}(undef, length(snapshots), length(shells.mvecs))
    for (i, R) in enumerate(snapshots)
        Q[i, :] = charge_fourier_modes(R, q, shells.mvecs)
    end
    return (shells=shells, Q=Q)
end


"""Average |Q_m|² over directions, NOT |average(Q_m)|².
Input: sample×mode complex matrix. Output: sample×shell real matrix.
"""
function shell_abs_Q2_series(Q::AbstractMatrix; shells=first_ten_shell_vectors())
    size(Q, 2) == length(shells.mvecs) || throw(ArgumentError("Q/mode count mismatch."))
    size(Q, 1) > 0 || throw(ArgumentError("Q must contain samples."))
    all(isfinite, Q) || throw(ArgumentError("Q must be finite."))
    C = Matrix{Float64}(undef, size(Q, 1), length(shells.shell_m2))
    for (s, indices) in enumerate(shells.shell_indices), i in axes(Q, 1)
        C[i, s] = mean(abs2(Q[i, j]) for j in indices)
    end
    return C
end


"""
    shell_block_statistics(values; n_blocks=5)

For a sample×shell REAL matrix, compute equal contiguous time-block averages,
the sample standard deviation of those averages, and SEM=block_std/sqrt(B).
All samples are used. Unequal blocks are rejected rather than silently dropping
samples or giving them unequal weights. SEM assumes sufficiently independent
blocks; increase block length and inspect drift before interpreting this error.
"""
function shell_block_statistics(values::AbstractMatrix{<:Real}; n_blocks::Integer=5)
    n_samples, n_shells = size(values)
    2 <= n_blocks <= n_samples || throw(ArgumentError("Need 2 ≤ n_blocks ≤ n_samples."))
    n_samples % n_blocks == 0 || throw(ArgumentError("n_samples must be divisible by n_blocks."))
    n_shells > 0 && all(isfinite, values) || throw(ArgumentError("Need finite, nonempty shell data."))
    block_size = n_samples ÷ n_blocks
    blocks = Matrix{Float64}(undef, n_blocks, n_shells)
    for b in 1:n_blocks
        rows = (b-1)*block_size+1:b*block_size
        blocks[b, :] = vec(mean(view(values, rows, :); dims=1))
    end
    block_std = vec(std(blocks; dims=1, corrected=true))
    return (mean=vec(mean(values; dims=1)), blocks=blocks,
        block_std=block_std, sem=block_std ./ sqrt(n_blocks),
        n_samples=n_samples, n_blocks=n_blocks, block_size=block_size)
end


"""
    dielectric_by_shell(snapshots, q, kappa; n_blocks=5, a=nothing)

Measure the first ten shells and their finite-k inverse dielectric response:
    D_m = 1 - kappa * <|Q_m|²> / (2π*m²).
Returns raw complex mode series, direction-resolved means, shell statistics,
and response statistics. No clipping of negative values or screening fit.
`a=σ/L`, if supplied, adds k_sigma=2π*a*sqrt(m²). kL is always available.
"""
function dielectric_by_shell(snapshots, q, kappa::Real;
        n_blocks::Integer=5, a::Union{Nothing,Real}=nothing)
    isfinite(kappa) && kappa > 0 || throw(ArgumentError("kappa must be positive and finite."))
    if a !== nothing
        isfinite(a) && a > 0 || throw(ArgumentError("a must be positive and finite."))
    end
    n = length(snapshots)
    2 <= n_blocks <= n && n % n_blocks == 0 ||
        throw(ArgumentError("Need equal blocks: 2 ≤ n_blocks ≤ n_samples and exact divisibility."))
    measured = measure_fourier_series(snapshots, q)
    shells = measured.shells
    C = shell_abs_Q2_series(measured.Q; shells=shells)
    prefactors = kappa ./ (2π .* shells.shell_m2)
    D = 1 .- C .* reshape(prefactors, 1, :)
    return (shells=shells, Q=measured.Q,
        mean_Q_by_mode=vec(mean(measured.Q; dims=1)),
        mean_abs_Q2_by_mode=vec(mean(abs2.(measured.Q); dims=1)),
        Q2_series=C, inv_eps_series=D,
        Q2=shell_block_statistics(C; n_blocks=n_blocks),
        inv_eps=shell_block_statistics(D; n_blocks=n_blocks),
        k_sigma=a === nothing ? nothing : a .* shells.kL)
end

end
