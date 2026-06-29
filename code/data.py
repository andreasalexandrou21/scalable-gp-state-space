"""
Synthetic data for the experiments.

We regress on a known, deterministic ground-truth function corrupted by Gaussian noise:

        f_true(t) = sin(t) + 0.5 * sin(3t),
        y_i       = f_true(t_i) + eps_i,     eps_i ~ N(0, sigma_noise^2).

Using a *known* curve (rather than a sample from the GP prior) means we can overlay the
truth on every plot and judge the fit honestly. The observation model matches Chapter 3:
        y = f + eps,   eps ~ N(0, sigma_noise^2 I).
"""

from __future__ import annotations

import numpy as np

from config import N_TEST, N_TRAIN, SEED, SIGMA_NOISE, T_MAX, T_MIN


def true_function(t: np.ndarray) -> np.ndarray:
    """The ground-truth function f_true(t) we are trying to recover."""
    return np.sin(t) + 0.5 * np.sin(3.0 * t)


def make_dataset(
    n_train: int = N_TRAIN,
    sigma_noise: float = SIGMA_NOISE,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n_train` random input locations and noisy observations.

    Returns
    -------
    t_train : (n_train,) ndarray
        Input times, **sorted** ascending. Sorting matters for the state-space method,
        which steps through time in order; it is irrelevant to dense GPR.
    y_train : (n_train,) ndarray
        Noisy observations y_i = f_true(t_i) + eps_i.
    """
    rng = np.random.default_rng(seed)
    t_train = np.sort(rng.uniform(t_min, t_max, size=n_train))
    eps = rng.normal(0.0, sigma_noise, size=n_train)
    y_train = true_function(t_train) + eps
    return t_train, y_train


def test_grid(n_test: int = N_TEST, t_min: float = T_MIN, t_max: float = T_MAX) -> np.ndarray:
    """A dense, evenly spaced set of test inputs t_* spanning the domain."""
    return np.linspace(t_min, t_max, n_test)
