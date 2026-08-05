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

end
