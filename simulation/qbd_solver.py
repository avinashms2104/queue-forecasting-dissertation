"""Exact stationary solution of the PH/PH/1 queue as a quasi-birth-death (QBD) process.

State: level n = number in system. Level 0 keeps only the arrival phase i;
levels n >= 1 keep (arrival phase i, service phase j), ordered i-major (Kronecker).

Repeating blocks for n >= 1 (arr = (alpha_a, T_a), srv = (beta_s, T_s)):
    A0 = (t_a alpha_a) (x) I_s        arrival completes          -> level up
    A1 = T_a (+) T_s                  phase change, no event     -> same level
    A2 = I_a (x) (t_s beta_s)         service completes          -> level down
Boundary:
    B00 = T_a,  B01 = (t_a alpha_a) (x) beta_s,  B10 = I_a (x) t_s

Matrix-geometric solution: pi_n = pi_1 R^{n-1} (n >= 1), where R is the minimal
non-negative solution of A0 + R A1 + R^2 A2 = 0.

The same function takes a MAP arrival process via (D0, D1) in place of the PH
arrival blocks, which is what the MAP/PH/1 stage needs.
"""
import numpy as np


def _solve_R(A0, A1, A2, tol=1e-14, max_iter=200000):
    """Iterate R <- A0 (-A1 - R A2)^{-1} (monotone, converges to the minimal solution)."""
    R = np.zeros_like(A0)
    for it in range(max_iter):
        R_new = A0 @ np.linalg.inv(-A1 - R @ A2)
        if np.max(np.abs(R_new - R)) < tol:
            return R_new, it + 1
        R = R_new
    raise RuntimeError("R iteration did not converge")


def solve_qbd(arr_D0, arr_D1, srv, n_max_dist=200):
    """Solve the MAP/PH/1 queue (PH arrivals are the special case D0=T_a, D1=t_a alpha_a).

    Returns a dict with rho, L, Lq, W, Wq, P(N=n) for n < n_max_dist, and diagnostics.
    """
    D0, D1 = np.asarray(arr_D0, float), np.asarray(arr_D1, float)
    ma, ms = D0.shape[0], srv.m
    Ia, Is = np.eye(ma), np.eye(ms)
    beta, S, s = srv.alpha, srv.T, srv.t

    A0 = np.kron(D1, Is)
    A1 = np.kron(D0, Is) + np.kron(Ia, S)
    A2 = np.kron(Ia, np.outer(s, beta))
    B00 = D0
    B01 = np.kron(D1, beta.reshape(1, -1))
    B10 = np.kron(Ia, s.reshape(-1, 1))

    # arrival rate from the stationary distribution of the arrival phase process
    Q = D0 + D1
    theta = np.linalg.lstsq(np.vstack([Q.T, np.ones(ma)]), np.r_[np.zeros(ma), 1], rcond=None)[0]
    lam = float(theta @ D1 @ np.ones(ma))
    rho = lam * srv.mean
    if rho >= 1:
        raise ValueError(f"unstable: rho = {rho:.4f}")

    R, iters = _solve_R(A0, A1, A2)

    # boundary equations: [pi0, pi1] * [[B00, B01], [B10, A1 + R A2]] = 0, plus normalisation
    n0, n1 = ma, ma * ms
    M = np.zeros((n0 + n1, n0 + n1))
    M[:n0, :n0], M[:n0, n0:] = B00, B01
    M[n0:, :n0], M[n0:, n0:] = B10, A1 + R @ A2
    inv_IR = np.linalg.inv(np.eye(n1) - R)
    norm = np.r_[np.ones(n0), inv_IR @ np.ones(n1)]
    A = np.vstack([M.T, norm])
    b = np.r_[np.zeros(n0 + n1), 1.0]
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    pi0, pi1 = x[:n0], x[n0:]

    one = np.ones(n1)
    L = float(pi1 @ inv_IR @ inv_IR @ one)                 # sum_n n pi_1 R^{n-1} 1
    Lq = float(pi1 @ R @ inv_IR @ inv_IR @ one)            # sum_n (n-1) pi_1 R^{n-1} 1
    dist = [float(pi0.sum())]
    v = pi1.copy()
    for _ in range(1, n_max_dist):
        dist.append(float(v.sum())); v = v @ R

    return dict(lam=lam, rho=rho, L=L, Lq=Lq, W=L / lam, Wq=Lq / lam,
                P_empty=float(pi0.sum()), dist=np.array(dist),
                sp_R=float(max(abs(np.linalg.eigvals(R)))), iters=iters,
                residual=float(np.max(np.abs(A0 + R @ A1 + R @ R @ A2))))


def solve_phph1(arr, srv, **kw):
    """PH/PH/1: arrival PH (alpha_a, T_a) becomes D0 = T_a, D1 = t_a alpha_a."""
    return solve_qbd(arr.T, np.outer(arr.t, arr.alpha), srv, **kw)


# ---------- reference formulas ----------
def pk_lq(lam, srv):
    """Pollaczek-Khinchine mean queue length (M/G/1 only)."""
    rho = lam * srv.mean
    return lam ** 2 * srv.moment(2) / (2 * (1 - rho))


def kingman_lq(arr, srv):
    """Kingman's heavy-traffic approximation for GI/G/1 (approximate, not exact)."""
    lam = 1 / arr.mean
    rho = lam * srv.mean
    wq = (rho / (1 - rho)) * ((arr.scv + srv.scv) / 2) * srv.mean
    return lam * wq