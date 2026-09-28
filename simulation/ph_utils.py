"""Phase-type (PH) distribution utilities.

A PH distribution is given by (alpha, T): alpha is the initial phase probability
row vector, T the sub-generator of the transient phases. The exit-rate vector is
t = -T 1. Everything here works for any PH distribution, so switching arrival or
service distributions later is a parameter change, not new code.
"""
import math

import numpy as np


class PH:
    def __init__(self, alpha, T, name="PH"):
        self.alpha = np.asarray(alpha, dtype=float).reshape(-1)
        self.T = np.asarray(T, dtype=float)
        self.t = -self.T.sum(axis=1)          # exit rates
        self.m = len(self.alpha)
        self.name = name
        assert self.T.shape == (self.m, self.m)
        assert abs(self.alpha.sum() - 1) < 1e-12, "alpha must sum to 1 (no mass at zero)"

    # ---------- moments ----------
    def moment(self, k):
        """E[X^k] = k! * alpha (-T)^{-k} 1."""
        M = np.linalg.inv(-self.T)
        return math.factorial(k) * float(self.alpha @ np.linalg.matrix_power(M, k) @ np.ones(self.m))

    @property
    def mean(self):
        return self.moment(1)

    @property
    def scv(self):
        """Squared coefficient of variation Var/mean^2."""
        m1, m2 = self.moment(1), self.moment(2)
        return (m2 - m1 ** 2) / m1 ** 2

    # ---------- sampling ----------
    def sample(self, n, rng):
        """Draw n i.i.d. samples by simulating the absorbing Markov chain (vectorised)."""
        m = self.m
        out = np.zeros(n)
        rates = -np.diag(self.T)                                  # total leaving rate per phase
        # jump probabilities from each phase: to other phases (cols 0..m-1) or absorption (col m)
        P = np.zeros((m, m + 1))
        for i in range(m):
            P[i, :m] = self.T[i] / rates[i]
            P[i, i] = 0.0
            P[i, m] = self.t[i] / rates[i]
        cumP = np.cumsum(P, axis=1)
        phase = rng.choice(m, size=n, p=self.alpha)
        alive = np.ones(n, dtype=bool)
        while alive.any():
            idx = np.nonzero(alive)[0]
            ph = phase[idx]
            out[idx] += rng.exponential(1.0 / rates[ph])
            u = rng.random(len(idx))
            nxt = (u[:, None] > cumP[ph]).sum(axis=1)             # next state index, m = absorb
            phase[idx] = np.minimum(nxt, m - 1)
            alive[idx[nxt == m]] = False
        return out


# ---------- common distributions, all parameterised by their mean ----------
def exponential(mean):
    return PH([1.0], [[-1.0 / mean]], name=f"Exp(mean={mean:g})")


def erlang(k, mean):
    """Erlang-k: k exponential phases in series, each with rate k/mean. SCV = 1/k."""
    r = k / mean
    T = -r * np.eye(k) + r * np.eye(k, k=1)
    a = np.zeros(k); a[0] = 1
    return PH(a, T, name=f"Erlang-{k}(mean={mean:g})")


def hyperexp2_balanced(mean, scv):
    """Two-phase hyperexponential with balanced means (standard 2-moment fit), SCV > 1."""
    assert scv > 1
    p = 0.5 * (1 + np.sqrt((scv - 1) / (scv + 1)))
    r1, r2 = 2 * p / mean, 2 * (1 - p) / mean
    return PH([p, 1 - p], [[-r1, 0], [0, -r2]], name=f"H2(mean={mean:g}, scv={scv:g})")