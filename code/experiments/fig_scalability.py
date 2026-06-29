"""
Figure: runtime vs number of observations n -- the scalability payoff.

Dense GPR must factorise the n x n matrix (K + sigma^2 I), costing O(n^3). The state-space
method runs a Kalman filter + RTS smoother whose cost is O(n m^3) with the state dimension m
fixed (m = 2 here), i.e. O(n). We time both over a range of n and fit the slopes on a log-log
plot; we expect slope ~3 for dense and ~1 for state-space.

We evaluate at a small fixed set of test points so that n (the number of *observations*) is the
only thing driving the cost.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELL, SIGMA2, SIGMA_NOISE, T_MAX, T_MIN
from data import make_dataset
from dense_gpr import fit_predict
from kalman_rts import gp_regression_ss
from matern_ssm import Matern32StateSpace

N_LIST = [64, 128, 256, 512, 1024, 2048, 4096]
REPEATS = 3  # take the fastest of a few runs to reduce timing noise
N_TEST_FIXED = 20  # small fixed test set: keep the focus on scaling in n
t_test = np.linspace(T_MIN, T_MAX, N_TEST_FIXED)
model = Matern32StateSpace(sigma2=SIGMA2, ell=ELL)


def time_call(fn) -> float:
    """Return the fastest wall-clock time (seconds) over REPEATS runs of `fn`."""
    best = np.inf
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


dense_times, ss_times = [], []
for n in N_LIST:
    # New dataset of size n (T_MAX scales with n so the point density stays sane).
    t_train, y_train = make_dataset(n_train=n, t_max=T_MIN + (T_MAX - T_MIN) * n / N_LIST[0])
    dense_times.append(time_call(lambda: fit_predict(t_train, y_train, t_test, SIGMA2, ELL, SIGMA_NOISE)))
    ss_times.append(time_call(lambda: gp_regression_ss(model, t_train, y_train, t_test, SIGMA_NOISE)))
    print(f"n={n:5d}   dense={dense_times[-1]:.4f}s   state-space={ss_times[-1]:.4f}s")

n_arr = np.array(N_LIST, dtype=float)
dense_arr = np.array(dense_times)
ss_arr = np.array(ss_times)

# Fit asymptotic slopes on the upper half (small-n is dominated by Python/library overhead).
half = len(N_LIST) // 2
slope_dense = np.polyfit(np.log(n_arr[half:]), np.log(dense_arr[half:]), 1)[0]
slope_ss = np.polyfit(np.log(n_arr[half:]), np.log(ss_arr[half:]), 1)[0]
print(f"\nfitted slope  dense       = {slope_dense:.2f}  (theory: 3)")
print(f"fitted slope  state-space = {slope_ss:.2f}  (theory: 1)")

plt.figure(figsize=(8, 6))
plt.loglog(n_arr, dense_arr, "C0o-", label=f"dense GPR (slope {slope_dense:.2f})")
plt.loglog(n_arr, ss_arr, "C1s-", label=f"state-space (slope {slope_ss:.2f})")
# Reference guide lines anchored at the last point.
plt.loglog(n_arr, dense_arr[-1] * (n_arr / n_arr[-1]) ** 3, "C0:", alpha=0.6, label="$O(n^3)$ reference")
plt.loglog(n_arr, ss_arr[-1] * (n_arr / n_arr[-1]) ** 1, "C1:", alpha=0.6, label="$O(n)$ reference")
plt.xlabel("number of observations n")
plt.ylabel("runtime (s)")
plt.title("Scalability: dense GPR $O(n^3)$ vs state-space $O(n)$")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, which="both", ls=":", alpha=0.4)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scalability.png")
plt.savefig(out, dpi=130)
print(f"Saved {out}")
