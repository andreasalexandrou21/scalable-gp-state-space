"""
Figure: dense GPR posterior vs state-space posterior (the equivalence result).

Overlays the two posterior means and 95% credible bands on the same axes. The thesis claim
is that they coincide, so the state-space curves should sit exactly on top of the dense ones.
We also print the maximum mean/std discrepancy as a hard number.
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
from kalman_rts import gp_regression_ss
from matern_ssm import Matern32StateSpace

# --- data + both posteriors ---
t_train, y_train = make_dataset()
t_test = test_grid()

mean_dense, cov_dense = fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)
std_dense = np.sqrt(np.diag(cov_dense))

model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)
mean_ss, var_ss = gp_regression_ss(model, t_train, y_train, t_test, SIGMA_NOISE)
std_ss = np.sqrt(var_ss)

# --- quantitative agreement ---
print(f"max |mean_dense - mean_ss| = {np.max(np.abs(mean_dense - mean_ss)):.2e}")
print(f"max |std_dense  - std_ss | = {np.max(np.abs(std_dense - std_ss)):.2e}")

# --- plot ---
plt.figure(figsize=(9, 5))
plt.plot(t_test, true_function(t_test), "k--", lw=1.2, label="true f(t)")
plt.plot(t_train, y_train, "kx", ms=6, label="noisy data")

# Dense GPR: solid line + shaded band.
plt.plot(t_test, mean_dense, "C0", lw=3, alpha=0.6, label="dense GPR mean")
plt.fill_between(
    t_test, mean_dense - 1.96 * std_dense, mean_dense + 1.96 * std_dense, color="C0", alpha=0.15
)
# State-space: dashed orange on top -- if it overlaps the blue, the methods agree.
plt.plot(t_test, mean_ss, "C1--", lw=1.5, label="state-space mean")
plt.plot(t_test, mean_ss - 1.96 * std_ss, "C1:", lw=1.0)
plt.plot(t_test, mean_ss + 1.96 * std_ss, "C1:", lw=1.0, label="state-space 95% band")

plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Equivalence: dense GPR vs state-space posterior (Matérn 3/2)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equivalence.png")
plt.savefig(out, dpi=130)
print(f"Saved {out}")
