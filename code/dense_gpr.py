"""
Dense Gaussian Process Regression baseline (Chapter 3), via scikit-learn.

This is the O(n^3) "textbook" method: build the n x n covariance matrix K, add the noise
variance to its diagonal, and apply the conditioning formulas (eq. in Chapter 3)

        mu(t*)    = k_{*,f} (K + sigma_noise^2 I)^{-1} y,
        sigma2(t*) = k(t*,t*) - k_{*,f} (K + sigma_noise^2 I)^{-1} k_{*,f}^T.

We let scikit-learn do the linear algebra. It acts as an *independent ground truth*:
the hand-coded state-space method (next batch) must reproduce these numbers.

Matching the thesis exactly
---------------------------
* Kernel: ConstantKernel(sigma2) * Matern(length_scale=ell, nu=1.5).
  sklearn's Matern(nu=1.5) is (1 + sqrt(3)|tau|/ell) exp(-sqrt(3)|tau|/ell); multiplying by
  sigma2 gives the thesis kernel k(tau) = sigma^2 (1 + lambda|tau|) e^{-lambda|tau|},
  lambda = sqrt(3)/ell.
* alpha = sigma_noise**2: this is the noise *variance* added to the diagonal (the "+ sigma^2 I").
* optimizer=None: do NOT re-fit hyperparameters; keep them fixed at the values we pass.
* normalize_y=False: the GP prior has zero mean, just like the thesis.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def make_gpr(sigma2: float, ell: float, sigma_noise: float) -> GaussianProcessRegressor:
    """Build a GaussianProcessRegressor configured to match the thesis assumptions."""
    kernel = ConstantKernel(sigma2, constant_value_bounds="fixed") * Matern(
        length_scale=ell, length_scale_bounds="fixed", nu=1.5
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=sigma_noise**2,  # noise variance added to K's diagonal
        optimizer=None,  # keep hyperparameters fixed
        normalize_y=False,  # zero-mean prior
    )


def fit_predict(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    sigma2: float,
    ell: float,
    sigma_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit dense GPR and return the posterior mean and full covariance at the test points.

    Returns
    -------
    mean : (n_test,) ndarray
        Posterior mean of f at each test input.
    cov : (n_test, n_test) ndarray
        Full posterior covariance matrix of f at the test inputs (the joint posterior).
        We return the *full* covariance, not just the diagonal, so that later we can both
        (a) compare covariances against the state-space method and (b) draw posterior samples.
    """
    gpr = make_gpr(sigma2, ell, sigma_noise)
    # sklearn expects a 2-D array of inputs: shape (n, 1) for our 1-D time inputs.
    gpr.fit(t_train.reshape(-1, 1), y_train)
    mean, cov = gpr.predict(t_test.reshape(-1, 1), return_cov=True)
    return mean, cov


if __name__ == "__main__":
    # Quick look: fit the baseline and plot the posterior. This is the "answer" the
    # state-space method must match in Batch 3.
    import matplotlib.pyplot as plt

    from config import ELL, SIGMA2, SIGMA_NOISE
    from data import make_dataset, test_grid, true_function

    t_train, y_train = make_dataset()
    t_test = test_grid()

    mean, cov = fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)
    std = np.sqrt(np.diag(cov))  # marginal posterior std at each test point

    plt.figure(figsize=(9, 5))
    plt.plot(t_test, true_function(t_test), "k--", lw=1.5, label="true f(t)")
    plt.plot(t_test, mean, "C0", lw=2, label="dense GPR mean")
    plt.fill_between(
        t_test, mean - 1.96 * std, mean + 1.96 * std, color="C0", alpha=0.2, label="95% band"
    )
    plt.plot(t_train, y_train, "kx", ms=6, label="noisy data")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Dense GPR posterior (Matérn 3/2) — the baseline")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = "code/experiments/baseline_posterior.png"
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")
