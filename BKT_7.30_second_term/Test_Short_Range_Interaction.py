"""
Tests for the second ("cosh") pair interaction added in
Short_Range_Interaction.py / Monte_Carlo.run_mc.

Run directly:

    python Test_Short_Range_Interaction.py

or with pytest:

    pytest Test_Short_Range_Interaction.py -v
"""

import numpy as np

from Short_Range_Interaction import (
    minimum_image_distance,
    energy2_pair,
    energy2,
    energy2_diff,
    energy2_exact_from_shifted,
)
from Monte_Carlo import run_mc1, run_mc


# ---------------------------------------------------------------------------
# (a) minimum-image distance across box boundaries
# ---------------------------------------------------------------------------

def test_minimum_image_across_boundary():
    Ri = np.array([0.01, 0.5])
    Rj = np.array([0.99, 0.5])

    r_min_image = minimum_image_distance(Ri, Rj, L=1.0)
    r_naive = np.linalg.norm(Ri - Rj)

    assert r_naive > 0.9
    assert np.isclose(r_min_image, 0.02, atol=1e-12)
    assert r_min_image < r_naive

    # energy2_pair using the raw (non-minimum-image) separation and the
    # minimum-image separation must differ once E0_over_T/xi_F make the
    # interaction resolve the difference.
    E0_over_T, xi_F = 5.0, 0.1
    e_min_image = energy2_pair(r_min_image, E0_over_T, xi_F)
    e_naive = energy2_pair(r_naive, E0_over_T, xi_F)
    assert not np.isclose(e_min_image, e_naive)

    # energy2() on a 2-particle configuration must use the minimum-image
    # value, not the raw one.
    R_all = np.array([Ri, Rj])
    U2 = energy2(R_all, E0_over_T, xi_F, L=1.0)
    assert np.isclose(U2, e_min_image)


def test_minimum_image_general_L():
    L = 2.5
    Ri = np.array([0.05, 0.0]) * L
    Rj = np.array([0.95, 0.0]) * L

    r = minimum_image_distance(Ri, Rj, L=L)
    assert np.isclose(r, 0.10 * L, atol=1e-12)


# ---------------------------------------------------------------------------
# (b) energy2_diff matches direct recomputation of energy2
# ---------------------------------------------------------------------------

def test_energy2_diff_matches_full_recompute():
    rng = np.random.default_rng(12345)
    N = 12
    E0_over_T = 3.0
    xi_F = 0.15
    L = 1.0

    R_all = rng.random((N, 2))

    n_trials = 200
    for _ in range(n_trials):
        i = rng.integers(N)
        R_old_i = R_all[i].copy()
        R_new_i = rng.random(2)

        dU2 = energy2_diff(R_all, i, R_old_i, R_new_i, E0_over_T, xi_F, L=L)

        U2_before = energy2(R_all, E0_over_T, xi_F, L=L)
        R_after = R_all.copy()
        R_after[i] = R_new_i
        U2_after = energy2(R_after, E0_over_T, xi_F, L=L)

        assert np.isclose(dU2, U2_after - U2_before, atol=1e-10, rtol=1e-8)

        # advance the "trajectory" so successive trials aren't independent
        R_all = R_after


# ---------------------------------------------------------------------------
# stability / exact-vs-shifted bookkeeping
# ---------------------------------------------------------------------------

def test_energy2_pair_numerically_stable_for_large_x():
    r = np.array([0.0])
    E0_over_T = 1e6
    xi_F = 1.0

    val = energy2_pair(r, E0_over_T, xi_F)
    assert np.isfinite(val).all()
    # for large x, -ln cosh(x) ~= -x + ln(2)
    x = E0_over_T
    assert np.isclose(val[0], -x + np.log(2.0), rtol=1e-6)


def test_energy2_exact_from_shifted():
    N = 5
    U2_shifted = 3.7
    n_pairs = N * (N - 1) / 2.0
    expected = U2_shifted - n_pairs * np.log(2.0)
    assert np.isclose(energy2_exact_from_shifted(U2_shifted, N), expected)


# ---------------------------------------------------------------------------
# (c) beta_U_total bookkeeping during sampling
# ---------------------------------------------------------------------------

def _small_run_kwargs(N=6, seed=1):
    q = np.array([1, -1] * (N // 2), dtype=float)
    a = 0.02
    return dict(
        N=N,
        q=q,
        kappa=2.0,
        a=a,
        alpha=np.sqrt(np.pi),
        n_cut=1,
        m_cut=3,
        step_size=0.1,
        n_sweeps=20,
        sample_every=2,
        n_burnin=5,
        seed=seed,
    )


def test_beta_U_total_bookkeeping():
    kwargs = _small_run_kwargs()

    (snapshots, U1_trace, U2_shifted_trace, U2_exact_trace,
     beta_U_total_trace, acc) = run_mc(
        **kwargs,
        include_log=True,
        include_cosh=True,
        E0_over_T=2.0,
        xi_F=0.2,
    )

    expected = kwargs["kappa"] * U1_trace + U2_shifted_trace
    assert np.allclose(beta_U_total_trace, expected)

    N = kwargs["N"]
    n_pairs = N * (N - 1) / 2.0
    assert np.allclose(U2_exact_trace, U2_shifted_trace - n_pairs * np.log(2.0))


# ---------------------------------------------------------------------------
# (d) include_cosh=False reproduces the previous log-only behavior
# ---------------------------------------------------------------------------

def test_include_cosh_false_matches_run_mc1():
    kwargs = _small_run_kwargs(seed=7)

    snapshots1, U1_trace1, acc1 = run_mc1(**kwargs)

    (snapshots2, U1_trace2, U2_shifted_trace2, U2_exact_trace2,
     beta_U_total_trace2, acc2) = run_mc(
        **kwargs,
        include_log=True,
        include_cosh=False,
    )

    assert acc1 == acc2
    assert np.allclose(U1_trace1, U1_trace2)
    for s1, s2 in zip(snapshots1, snapshots2):
        assert np.allclose(s1, s2)

    assert np.all(U2_shifted_trace2 == 0.0)
    assert np.all(U2_exact_trace2 == 0.0)
    assert np.allclose(beta_U_total_trace2, kwargs["kappa"] * U1_trace2)


# ---------------------------------------------------------------------------
# (e) E0_over_T = 0 gives zero shifted U2 and does not affect the sampler
# ---------------------------------------------------------------------------

def test_E0_over_T_zero_is_inert():
    kwargs = _small_run_kwargs(seed=42)

    (snapshots_off, U1_off, U2s_off, U2e_off, betaU_off, acc_off) = run_mc(
        **kwargs, include_log=True, include_cosh=False,
    )
    (snapshots_on, U1_on, U2s_on, U2e_on, betaU_on, acc_on) = run_mc(
        **kwargs, include_log=True, include_cosh=True, E0_over_T=0.0, xi_F=0.3,
    )

    assert np.all(U2s_on == 0.0)
    assert acc_off == acc_on
    assert np.allclose(U1_off, U1_on)
    for s_off, s_on in zip(snapshots_off, snapshots_on):
        assert np.allclose(s_off, s_on)


def test_include_cosh_only_mode_runs():
    kwargs = _small_run_kwargs(seed=3)

    (snapshots, U1_trace, U2s, U2e, betaU, acc) = run_mc(
        **kwargs, include_log=False, include_cosh=True, E0_over_T=1.5, xi_F=0.2,
    )

    assert np.all(U1_trace == 0.0)
    assert np.all(betaU == U2s)
    assert not np.all(U2s == 0.0)


if __name__ == "__main__":
    tests = [
        test_minimum_image_across_boundary,
        test_minimum_image_general_L,
        test_energy2_diff_matches_full_recompute,
        test_energy2_pair_numerically_stable_for_large_x,
        test_energy2_exact_from_shifted,
        test_beta_U_total_bookkeeping,
        test_include_cosh_false_matches_run_mc1,
        test_E0_over_T_zero_is_inert,
        test_include_cosh_only_mode_runs,
    ]

    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")

    print(f"\nAll {len(tests)} tests passed.")
