module OneOverEpsilon0


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


function ϵ0_inv_first_shell(snapshots, q, kappa::Float64)
    mean_Q2 = mean_abs_Q2_first_shell(snapshots, q)

    inv_eps = 1.0 - kappa * mean_Q2 / (2π)

    return (
        mean_abs_Q2_first_shell = mean_Q2,
        ϵ0_inv = inv_eps,
    )
end

end