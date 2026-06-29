"""
Figure: posterior function samples from each method.

Equal mean and variance at each point is necessary but not sufficient for "same posterior" --
two processes can share marginals yet differ in how neighbouring points co-vary. To check the
*joint* distribution, we draw sample functions.

Trick: a Gaussian sample is  mean + L z,  where L L^T = covariance and z is standard normal.
We use the *same* z for both methods. If their joint covariances are equal, the sampled paths
are identical (up to round-off). So overlapping solid/dashed paths prove equality of the full
posterior process, not just its first two marginal moments.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELL, SIGMA2, SIGMA_NOISE
from data import make_dataset, test_grid, true_function
from dense_gpr import fit_predict
from kalman_rts import joint_f_posterior
from matern_ssm import Matern32StateSpace

N_SAMPLES = 5
JITTER = 1e-9  # tiny diagonal nudge so the Cholesky factorisation stays numerically stable

t_train, y_train = make_dataset()
t_test = test_grid()
n_test = t_test.shape[0]

mean_dense, cov_dense = fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)
model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
mean_ss, cov_ss = joint_f_posterior(model, t_train, y_train, t_test, SIGMA_NOISE)

# Cholesky factors of each posterior covariance.
L_dense = np.linalg.cholesky(cov_dense + JITTER * np.eye(n_test))
L_ss = np.linalg.cholesky(cov_ss + JITTER * np.eye(n_test))

# Shared standard-normal draws -> the only difference between the two sample sets is the covariance.
rng = np.random.default_rng(1)
Z = rng.standard_normal((n_test, N_SAMPLES))
samples_dense = mean_dense[:, None] + L_dense @ Z
samples_ss = mean_ss[:, None] + L_ss @ Z

print(f"max |sample_dense - sample_ss| = {np.max(np.abs(samples_dense - samples_ss)):.2e}")

plt.figure(figsize=(9, 5))
plt.plot(t_test, true_function(t_test), "k--", lw=1.2, label="true f(t)")
plt.plot(t_train, y_train, "kx", ms=6, label="noisy data")
for i in range(N_SAMPLES):
    plt.plot(t_test, samples_dense[:, i], "C0", lw=2, alpha=0.5,
             label="dense GPR samples" if i == 0 else None)
    plt.plot(t_test, samples_ss[:, i], "C1--", lw=1.0,
             label="state-space samples" if i == 0 else None)
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Posterior samples: dense GPR vs state-space (same random draws)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.png")
plt.savefig(out, dpi=130)
print(f"Saved {out}")
