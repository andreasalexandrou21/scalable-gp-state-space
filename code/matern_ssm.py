"""
Matérn (nu = 3/2, i.e. p = 1) Gaussian process as a state-space model.

This file builds the *continuous-time* LTI SDE from Section 6.2 of the thesis and the
*discrete-time* matrices needed by the Kalman filter / RTS smoother (Chapter 5).

The Matérn nu = 3/2 prior corresponds to the second-order scalar SDE

        d^2 f/dt^2 + 2*lambda * df/dt + lambda^2 * f = w(t),                       (companion form, eq. lti-sde-scalar)

with lambda = sqrt(2*nu)/ell = sqrt(3)/ell and white-noise intensity q.
Writing the state as x(t) = (f(t), f'(t))^T turns this into the first-order system

        dx/dt = F x + L w(t),     f(t) = H x(t).                                   (eq. lti-sde-1)

Everything below is a direct transcription of the matrices in Section 6.2.
The `__main__` block at the bottom checks the closed-form expressions against
scipy (matrix exponential + Lyapunov solver), so you can *see* they agree.
"""

from __future__ import annotations

import numpy as np


class Matern32StateSpace:
    """State-space representation of a Matérn nu = 3/2 GP prior.

    Parameters
    ----------
    sigma2 : float
        Magnitude (variance) hyperparameter sigma^2 of the kernel.
    ell : float
        Length-scale hyperparameter ell.

    Notes
    -----
    We keep `sigma2` and `ell` as explicit attributes (not baked-in constants) so that
    later we can optimise them by empirical Bayes without touching the rest of the code.
    """

    def __init__(self, sigma2: float, ell: float) -> None:
        self.sigma2 = float(sigma2)
        self.ell = float(ell)

        # lambda = sqrt(2*nu)/ell with nu = 3/2  =>  sqrt(3)/ell.   (Section 6.2)
        self.lam = np.sqrt(3.0) / self.ell

        # --- Continuous-time LTI SDE matrices (eq. lti-sde-1, Section 6.2, p = 1) ---
        #
        #         F = [[ 0      ,  1      ],
        #              [-lambda^2, -2*lambda]]
        #
        # This is the companion matrix: its last row holds the coefficients
        # (-h0, -h1) = (-lambda^2, -2*lambda) of P(i*w) = (lambda + i*w)^2.
        self.F = np.array(
            [[0.0, 1.0], [-self.lam**2, -2.0 * self.lam]],
            dtype=float,
        )

        # L picks out which component the noise enters: only the last (highest-derivative) one.
        self.L = np.array([[0.0], [1.0]], dtype=float)

        # H reads off f(t) = first component of the state:  f(t) = H x(t).
        self.H = np.array([[1.0, 0.0]], dtype=float)

        # White-noise spectral density / intensity q.
        # For Matérn nu = p + 1/2 the thesis gives q = sigma^2 * 2*sqrt(pi)*Gamma(p+1)/Gamma(p+1/2) * lambda^(2p+1).
        # For p = 1 this collapses to q = 4 * lambda^3 * sigma^2   (the "S_w(w) = 4 lambda^3 sigma^2" in Section 6.2).
        self.q = 4.0 * self.lam**3 * self.sigma2

        # --- Stationary state covariance P_inf (solves the Lyapunov eq. lyapunov) ---
        # Section 6.2 solves it in closed form for p = 1:
        #         P_inf = diag(sigma^2, lambda^2 * sigma^2).
        # Interpretation: Var[f] = sigma^2 and Var[f'] = lambda^2 * sigma^2, and f, f' are
        # uncorrelated in the stationary regime.
        self.P_inf = np.array(
            [[self.sigma2, 0.0], [0.0, self.lam**2 * self.sigma2]],
            dtype=float,
        )

    def discretize(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Discrete-time transition matrix A_k and process-noise covariance Q_k for a step dt.

        These are the quantities the Kalman filter needs:

            x_k = A_k x_{k-1} + q_{k-1},   q_{k-1} ~ N(0, Q_k).      (eq. lti-sde-discrete)

        Parameters
        ----------
        dt : float
            Time gap Delta t_k = t_k - t_{k-1} between consecutive grid points.

        Returns
        -------
        A : (2, 2) ndarray
            A_k = exp(F * dt). Closed form from Section 6.2 (repeated eigenvalue -lambda):

                A_k = e^{-lambda*dt} * [[1 + lambda*dt,        dt      ],
                                        [-lambda^2*dt,    1 - lambda*dt ]].

        Q : (2, 2) ndarray
            Process-noise covariance. We use the *stationarity identity* (eq. Q-stationary)

                Q_k = P_inf - A_k P_inf A_k^T,

            which is exact here and avoids evaluating the Itô-isometry integral directly.
        """
        lam = self.lam
        decay = np.exp(-lam * dt)
        A = decay * np.array(
            [[1.0 + lam * dt, dt], [-(lam**2) * dt, 1.0 - lam * dt]],
            dtype=float,
        )
        Q = self.P_inf - A @ self.P_inf @ A.T
        return A, Q


# Explicit closed-form Q from Section 6.2, used only to cross-check `discretize`.
def _thesis_Q_closed_form(model: Matern32StateSpace, dt: float) -> np.ndarray:
    """The fully expanded Q_k matrix written out in Section 6.2 (the big matrix)."""
    lam = model.lam
    s2 = model.sigma2
    e2 = np.exp(-2.0 * lam * dt)  # e^{-2 lambda dt}
    q11 = 1.0 - e2 * ((1.0 + lam * dt) ** 2 + (lam * dt) ** 2)
    q12 = 2.0 * lam**3 * dt**2 * e2
    q22 = (lam**2) * (1.0 - e2 * ((1.0 - lam * dt) ** 2 + (lam * dt) ** 2))
    return s2 * np.array([[q11, q12], [q12, q22]], dtype=float)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Verification: confirm the closed-form matrices match general-purpose scipy routines.
    # If these pass, we trust the hand-derived Section 6.2 formulas.
    # ---------------------------------------------------------------------------------
    from scipy.linalg import expm, solve_continuous_lyapunov

    # Arbitrary but representative hyperparameters and a step size.
    model = Matern32StateSpace(sigma2=2.0, ell=0.7)
    dt = 0.123

    print("lambda =", model.lam)
    print("q      =", model.q)
    print("F =\n", model.F)
    print("P_inf =\n", model.P_inf)

    # 1) P_inf should satisfy the continuous Lyapunov equation  F P + P F^T + L q L^T = 0.
    #    scipy.solve_continuous_lyapunov solves  F X + X F^T = -(L q L^T).
    P_inf_scipy = solve_continuous_lyapunov(model.F, -(model.L * model.q) @ model.L.T)
    err_Pinf = np.max(np.abs(model.P_inf - P_inf_scipy))
    print(f"\n[check] P_inf vs scipy Lyapunov solver: max abs diff = {err_Pinf:.2e}")

    # 2) A_k = exp(F dt): closed form vs scipy matrix exponential.
    A, Q = model.discretize(dt)
    A_scipy = expm(model.F * dt)
    err_A = np.max(np.abs(A - A_scipy))
    print(f"[check] A_k closed form vs scipy expm:   max abs diff = {err_A:.2e}")

    # 3) Q_k: stationarity-identity version vs the fully expanded thesis matrix.
    Q_thesis = _thesis_Q_closed_form(model, dt)
    err_Q = np.max(np.abs(Q - Q_thesis))
    print(f"[check] Q_k identity vs expanded thesis: max abs diff = {err_Q:.2e}")

    assert err_Pinf < 1e-10, "P_inf disagrees with the Lyapunov solver"
    assert err_A < 1e-10, "A_k disagrees with scipy expm"
    assert err_Q < 1e-10, "Q_k identity disagrees with the expanded thesis matrix"
    print("\nAll checks passed: the Section 6.2 closed forms are correct.")
