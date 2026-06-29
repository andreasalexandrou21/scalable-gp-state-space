"""
Figure: numerical discrepancy between dense GPR and the state-space method.

The equivalence figure shows the curves overlapping; this one *quantifies* it. We plot the
absolute differences in posterior mean and posterior std at every test point, on a log scale.
They sit at ~1e-14, i.e. floating-point round-off -- the two methods are exactly equal, not
merely close. We also report the difference of the full joint covariance matrices.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELL, SIGMA2, SIGMA_NOISE
from data import make_dataset, test_grid
from dense_gpr import fit_predict
from kalman_rts import joint_f_posterior
from matern_ssm import Matern32StateSpace

t_train, y_train = make_dataset()
t_test = test_grid()

# Dense GPR: mean + full covariance.
mean_dense, cov_dense = fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)
std_dense = np.sqrt(np.diag(cov_dense))

# State-space: mean + full joint covariance (same quantities, different algorithm).
model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
mean_ss, cov_ss = joint_f_posterior(model, t_train, y_train, t_test, SIGMA_NOISE)
std_ss = np.sqrt(np.diag(cov_ss))

mean_err = np.abs(mean_dense - mean_ss)
std_err = np.abs(std_dense - std_ss)
cov_err = np.max(np.abs(cov_dense - cov_ss))
print(f"max |mean diff|            = {mean_err.max():.2e}")
print(f"max |std diff|             = {std_err.max():.2e}")
print(f"max |covariance matrix diff| = {cov_err:.2e}")

plt.figure(figsize=(9, 4.5))
plt.semilogy(t_test, mean_err + 1e-300, "C0", label="|mean_dense - mean_ss|")
plt.semilogy(t_test, std_err + 1e-300, "C1", label="|std_dense - std_ss|")
plt.axhline(np.finfo(float).eps, color="gray", ls=":", label="machine epsilon")
plt.xlabel("t")
plt.ylabel("absolute difference (log scale)")
plt.title("Dense GPR vs state-space: pointwise discrepancy")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error.png")
plt.savefig(out, dpi=130)
print(f"Saved {out}")
