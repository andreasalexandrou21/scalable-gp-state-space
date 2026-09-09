"""The claim this repository makes is that the state-space formulation is *exact*.

That is the one thing worth testing automatically: not that the code runs, but that the
O(n) filter-plus-smoother posterior is the same object as the O(n^3) dense GP posterior,
and that the hand-derived closed-form matrices agree with general-purpose linear algebra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import ELL, SIGMA2, SIGMA_NOISE  # noqa: E402
from data import make_dataset  # noqa: E402
from data import test_grid as make_test_grid  # noqa: E402
from dense_gpr import fit_predict  # noqa: E402
from kalman_rts import gp_regression_ss, joint_f_posterior  # noqa: E402
from matern_ssm import Matern32StateSpace, _thesis_Q_closed_form  # noqa: E402

TOL = 1e-10


def _setup():
    t_train, y_train = make_dataset()
    t_test = make_test_grid()
    mean_dense, cov_dense = fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)
    model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
    return t_train, y_train, t_test, mean_dense, cov_dense, model


def test_posterior_mean_matches_dense_gp():
    t_train, y_train, t_test, mean_dense, _, model = _setup()
    mean_ss, _ = gp_regression_ss(model, t_train, y_train, t_test, SIGMA_NOISE)
    assert np.max(np.abs(mean_dense - mean_ss)) < TOL


def test_marginal_variance_matches_dense_gp():
    t_train, y_train, t_test, _, cov_dense, model = _setup()
    _, var_ss = gp_regression_ss(model, t_train, y_train, t_test, SIGMA_NOISE)
    assert np.max(np.abs(np.diag(cov_dense) - var_ss)) < TOL


def test_joint_covariance_matches_dense_gp():
    """The cross-covariances too, not just the diagonal."""
    t_train, y_train, t_test, mean_dense, cov_dense, model = _setup()
    mean_j, cov_j = joint_f_posterior(model, t_train, y_train, t_test, SIGMA_NOISE)
    assert np.max(np.abs(mean_dense - mean_j)) < TOL
    assert np.max(np.abs(cov_dense - cov_j)) < TOL


def test_closed_form_matrices_match_scipy():
    """A_k and P_inf are hand-derived; check them against expm and a Lyapunov solver."""
    from scipy.linalg import expm, solve_continuous_lyapunov

    model = Matern32StateSpace(sigma2=2.0, ell=0.7)
    dt = 0.123
    A, Q = model.discretize(dt)

    P_inf_scipy = solve_continuous_lyapunov(model.F, -(model.L * model.q) @ model.L.T)
    assert np.max(np.abs(model.P_inf - P_inf_scipy)) < TOL
    assert np.max(np.abs(A - expm(model.F * dt))) < TOL
    assert np.max(np.abs(Q - _thesis_Q_closed_form(model, dt))) < TOL


def test_stationary_covariance_is_a_fixed_point():
    """P_inf must be invariant under the discrete update, for any step size."""
    model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
    for dt in (0.01, 0.5, 3.0):
        A, Q = model.discretize(dt)
        assert np.max(np.abs(A @ model.P_inf @ A.T + Q - model.P_inf)) < TOL
