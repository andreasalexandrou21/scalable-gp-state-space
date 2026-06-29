"""
State-space GP regression: Kalman filter (forward) + RTS smoother (backward).

This is the thesis contribution, coded from scratch. It computes the *same* posterior as
dense GPR (dense_gpr.py) but in O(n) time instead of O(n^3).

The pipeline (Chapter 5):
  1. Build one sorted time grid = training times UNION test times.
       - At a *training* time we have an observation y  -> do predict + update.
       - At a *test-only* time there is no observation     -> do predict only ("missing
         observation"). This is the standard trick that lets the filter/smoother report a
         posterior at arbitrary test points t_*, not just where we observed.
  2. Kalman filter forward: prediction (eq. kalman_pred) + update (eq. kalman_update).
  3. RTS smoother backward (eq. rts_smoother): turns the *filtering* posterior p(x_k | y_{1:k})
     into the *smoothing* posterior p(x_k | y_{1:T}) -- the analogue of conditioning on the
     whole dataset, which is what GPR does.
  4. Read off f(t_k) = H x_k:  mean = m_k[0], variance = P_k[0,0].

As a free by-product the filter accumulates the log marginal likelihood log p(y), which is
exactly what empirical-Bayes hyperparameter optimisation will later maximise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matern_ssm import Matern32StateSpace


@dataclass
class SmootherResult:
    """Everything the forward+backward pass produces, kept for reuse in later experiments."""

    t_grid: np.ndarray  # (N,) sorted grid times
    m_smooth: np.ndarray  # (N, d) smoothed state means     m_k^s
    P_smooth: np.ndarray  # (N, d, d) smoothed state covs    P_k^s
    m_filt: np.ndarray  # (N, d) filtered state means      m_k
    P_filt: np.ndarray  # (N, d, d) filtered state covs    P_k
    m_pred: np.ndarray  # (N, d) predicted state means     m_k^-
    P_pred: np.ndarray  # (N, d, d) predicted state covs   P_k^-
    gain: np.ndarray  # (N, d, d) smoother gains G_k (G[k] maps step k+1 back to k)
    log_marginal_likelihood: float


def filter_smooth(
    model: Matern32StateSpace,
    t_grid: np.ndarray,
    y_grid: np.ndarray,
    is_obs: np.ndarray,
    sigma_noise: float,
) -> SmootherResult:
    """Run the Kalman filter then the RTS smoother over a prepared, sorted grid.

    Parameters
    ----------
    model : Matern32StateSpace
        Supplies F, H, P_inf and discretize(dt) -> (A_k, Q_k).
    t_grid : (N,) ndarray
        Sorted grid times.
    y_grid : (N,) ndarray
        Observation at each grid point (NaN where there is no observation).
    is_obs : (N,) ndarray of bool
        True where the grid point carries an observation.
    sigma_noise : float
        Observation noise standard deviation.
    """
    H = model.H  # (1, d)
    d = model.F.shape[0]
    N = t_grid.shape[0]
    R = sigma_noise**2  # measurement noise variance

    # Storage for every quantity at every step (the smoother needs them all).
    m_pred = np.zeros((N, d))
    P_pred = np.zeros((N, d, d))
    m_filt = np.zeros((N, d))
    P_filt = np.zeros((N, d, d))
    A_step = np.zeros((N, d, d))  # A_step[k] = transition that produced step k from k-1

    log_ml = 0.0

    # ---------------- Forward pass: Kalman filter ----------------
    for k in range(N):
        if k == 0:
            # Initialise at the first grid point with the stationary prior:
            #   m_0^- = 0,  P_0^- = P_inf.   (no time step precedes the first point)
            m_pred[k] = np.zeros(d)
            P_pred[k] = model.P_inf
            A_step[k] = np.eye(d)  # placeholder, never used by the smoother
        else:
            dt = t_grid[k] - t_grid[k - 1]
            A, Q = model.discretize(dt)
            A_step[k] = A
            # Prediction step (eq. kalman_pred):
            #   m_k^- = A m_{k-1},   P_k^- = A P_{k-1} A^T + Q.
            m_pred[k] = A @ m_filt[k - 1]
            P_pred[k] = A @ P_filt[k - 1] @ A.T + Q

        if is_obs[k]:
            # Update step (eq. kalman_update): fold in the scalar observation y_k.
            v = y_grid[k] - (H @ m_pred[k])[0]  # innovation        v_k = y_k - H m_k^-
            S = (H @ P_pred[k] @ H.T)[0, 0] + R  # innovation var   S_k = H P_k^- H^T + R
            K = (P_pred[k] @ H.T) / S  # Kalman gain (d,1)  K_k = P_k^- H^T S_k^{-1}
            m_filt[k] = m_pred[k] + (K[:, 0] * v)
            P_filt[k] = P_pred[k] - K @ (S * K.T)  # = P_k^- - K_k S_k K_k^T
            # Accumulate log p(y_k | y_{1:k-1}) for a scalar Gaussian innovation.
            log_ml += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)
        else:
            # No observation: the filtered estimate is just the prediction.
            m_filt[k] = m_pred[k]
            P_filt[k] = P_pred[k]

    # ---------------- Backward pass: RTS smoother (eq. rts_smoother) ----------------
    m_smooth = np.zeros((N, d))
    P_smooth = np.zeros((N, d, d))
    gain = np.zeros((N, d, d))
    # Initialise the smoother at the final step: m_T^s = m_T, P_T^s = P_T.
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for k in range(N - 2, -1, -1):
        A_next = A_step[k + 1]  # transition from step k to step k+1
        # Smoother gain  G_k = P_k A_{k+1}^T (P_{k+1}^-)^{-1}.
        G = P_filt[k] @ A_next.T @ np.linalg.inv(P_pred[k + 1])
        gain[k] = G
        m_smooth[k] = m_filt[k] + G @ (m_smooth[k + 1] - m_pred[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

    return SmootherResult(
        t_grid=t_grid,
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        gain=gain,
        log_marginal_likelihood=log_ml,
    )


def _build_grid(
    t_train: np.ndarray, y_train: np.ndarray, t_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge train and test times into one sorted grid.

    Returns the sorted grid times, the per-point observation values (NaN at test points),
    the boolean observation mask, and `test_pos`: for each original test point, its index
    in the sorted grid (so we can read its posterior back out afterwards).
    """
    n_train = t_train.shape[0]
    n_test = t_test.shape[0]

    t_all = np.concatenate([t_train, t_test])
    y_all = np.concatenate([y_train, np.full(n_test, np.nan)])
    is_obs = np.concatenate([np.ones(n_train, dtype=bool), np.zeros(n_test, dtype=bool)])
    # Tag each test point with its original index; train points get -1.
    test_tag = np.concatenate([np.full(n_train, -1, dtype=int), np.arange(n_test)])

    # Stable sort keeps train before test when times coincide (train updates first).
    order = np.argsort(t_all, kind="stable")
    t_all, y_all, is_obs, test_tag = t_all[order], y_all[order], is_obs[order], test_tag[order]

    # Where in the sorted grid does each original test point live?
    test_pos = np.empty(n_test, dtype=int)
    test_pos[test_tag[test_tag >= 0]] = np.nonzero(test_tag >= 0)[0]
    return t_all, y_all, is_obs, test_pos


def gp_regression_ss(
    model: Matern32StateSpace,
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    sigma_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """State-space GP regression: posterior mean and variance of f at the test points.

    Returns
    -------
    mean : (n_test,) ndarray   posterior mean of f(t_*)
    var  : (n_test,) ndarray   posterior variance of f(t_*)
    """
    t_grid, y_grid, is_obs, test_pos = _build_grid(t_train, y_train, t_test)
    res = filter_smooth(model, t_grid, y_grid, is_obs, sigma_noise)
    # f(t_k) = H x_k:  mean is the first state component, variance the (0,0) entry.
    mean = res.m_smooth[test_pos, 0]
    var = res.P_smooth[test_pos, 0, 0]
    return mean, var


def joint_f_posterior(
    model: Matern32StateSpace,
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    sigma_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Full *joint* posterior of f at the test points: mean vector and dense covariance matrix.

    The smoother gives the marginal covariance P_k^s at each point, but two points are also
    correlated in the posterior. Those cross-covariances come from a backward recursion: for a
    fixed later grid index j, conditioning on the future makes

        Cov(x_k, x_j | y_{1:T}) = G_k Cov(x_{k+1}, x_j | y_{1:T}),     k < j,

    with base case Cov(x_j, x_j) = P_j^s. Reading off f = H x gives the joint covariance of f.
    We need this only for the small equivalence/sample figures (it is O(N^2)); the cheap O(N)
    `gp_regression_ss` above is what we time in the scalability experiment.

    Returns
    -------
    mean : (n_test,) ndarray
    cov  : (n_test, n_test) ndarray   joint posterior covariance of f at the test points
    """
    t_grid, y_grid, is_obs, test_pos = _build_grid(t_train, y_train, t_test)
    res = filter_smooth(model, t_grid, y_grid, is_obs, sigma_noise)
    H = model.H
    d = model.F.shape[0]
    n_test = t_test.shape[0]

    mean = res.m_smooth[test_pos, 0]
    cov = np.zeros((n_test, n_test))

    # test_pos is ascending, so column b sits at grid index kb >= every earlier test point.
    for b in range(n_test):
        kb = test_pos[b]
        # cross[k] = Cov(x_k, x_kb | y_{1:T}) for k = 0..kb, built backward from the marginal.
        cross = np.zeros((kb + 1, d, d))
        cross[kb] = res.P_smooth[kb]
        for k in range(kb - 1, -1, -1):
            cross[k] = res.gain[k] @ cross[k + 1]
        # Fill every test row a <= b (its grid index test_pos[a] <= kb); use symmetry.
        for a in range(b + 1):
            ka = test_pos[a]
            val = (H @ cross[ka] @ H.T)[0, 0]  # Cov(f_a, f_b) = H Cov(x_a, x_b) H^T
            cov[a, b] = val
            cov[b, a] = val
    return mean, cov


if __name__ == "__main__":
    # Sanity print: run the pipeline once and report the log marginal likelihood.
    from config import ELL, SIGMA2, SIGMA_NOISE
    from data import make_dataset, test_grid

    model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
    t_train, y_train = make_dataset()
    t_test = test_grid()
    mean, var = gp_regression_ss(model, t_train, y_train, t_test, SIGMA_NOISE)
    print("state-space posterior computed at", t_test.shape[0], "test points")
    print("mean[:3] =", mean[:3])
    print("var[:3]  =", var[:3])
