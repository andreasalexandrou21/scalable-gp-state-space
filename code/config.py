"""
Central place for all hyperparameters and experiment settings.

Keeping these in one module means every experiment uses the *same* numbers, and when we
later optimise the hyperparameters (empirical Bayes) there is exactly one place to change.
"""

from __future__ import annotations

# --- GP / kernel hyperparameters (fixed by hand for now) ---
SIGMA2: float = 1.0  # kernel magnitude sigma^2  (prior variance of f)
ELL: float = 1.0  # length-scale ell           (how fast the function wiggles)
SIGMA_NOISE: float = 0.2  # observation noise std sigma_noise (so noise *variance* is SIGMA_NOISE**2)

# --- Data settings ---
N_TRAIN: int = 40  # number of noisy training observations
T_MIN: float = 0.0  # left end of the input domain
T_MAX: float = 10.0  # right end of the input domain
SEED: int = 0  # RNG seed, so every run is reproducible

# --- Test grid (where we evaluate / plot the posterior) ---
N_TEST: int = 400  # number of dense test points across [T_MIN, T_MAX]
