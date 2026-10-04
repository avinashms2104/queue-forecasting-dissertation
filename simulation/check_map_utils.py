"""Checks for map_utils.py and for qbd_solver.py with MAP arrivals.

Part A: the MAP sampler matches the exact formulas (mean, scv, lag correlations).
Part B: the QBD solver with MAP input agrees with every case we already trust,
        and with an independent brute-force solution of the full Markov chain.

Usage:  python simulation/check_map_utils.py
"""
import numpy as np

from ph_utils import erlang
from map_utils import poisson_map, ph_renewal_map, bursty_mmpp2
from qbd_solver import solve_qbd, solve_phph1
from mg1_simulator import theoretical_Lq_pollaczek_khinchine, RHOS

rng = np.random.default_rng(2026)
N = 500_000
LAGS = [1, 2, 5, 10]


def sample_corr(x, j):
    return float(np.corrcoef(x[:-j], x[j:])[0, 1])


def report(M, n=N):
    x, _ = M.sample(n, rng)
    print(f"  {M.name}")
    print(f"    {'':14}{'exact':>9}{'sampled':>10}")
    print(f"    {'mean gap':14}{M.mean:9.4f}{x.mean():10.4f}")
    print(f"    {'scv':14}{M.scv:9.4f}{x.var() / x.mean() ** 2:10.4f}")
    for j in LAGS:
        print(f"    {'corr lag ' + str(j):14}{M.lag_corr(j):9.4f}{sample_corr(x, j):10.4f}")
    return x


print("=" * 70)
print("PART A: sampler against exact formulas")
print("=" * 70)

print("\nA1. Poisson written as a MAP (should be scv 1, no correlation)")
report(poisson_map(0.7))

print("\nA2. Erlang-2 renewal written as a MAP (scv 0.5, no correlation; same as PH/PH/1 arrivals)")
report(ph_renewal_map(erlang(2, 1 / 0.7)))

print("\nA3. The MAP/PH/1 arrival process: bursty MMPP-2 at rate 0.7")
M = bursty_mmpp2(0.7)
x = report(M, n=1_000_000)
print(f"    limiting index of dispersion: exact {M.idc_limit():.3f}, "
      f"scv*(1+2*sum corr) = {M.scv * (1 + 2 * sum(M.lag_corr(j) for j in range(1, 3000))):.3f}")

print("\nA4. Rescaling to another rho changes only the time scale")
for rho in RHOS:
    Mr = bursty_mmpp2(rho)
    print(f"  rho={rho:<5} rate {Mr.lam:.4f}  mean gap {Mr.mean:7.4f}  "
          f"scv {Mr.scv:.4f}  corr lag 1 {Mr.lag_corr(1):.4f}")

print("\nA5. Chunked sampling keeps the correlation across chunk boundaries")
M = bursty_mmpp2(0.7)
parts, ph = [], None
for _ in range(100):
    g, ph = M.sample(10_000, rng, phase0=ph)
    parts.append(g)
x = np.concatenate(parts)
print(f"  100 chunks of 10,000: corr lag 1 sampled {sample_corr(x, 1):.4f}, exact {M.lag_corr(1):.4f}")

print("\n" + "=" * 70)
print("PART B: QBD solver with MAP arrivals")
print("=" * 70)
srv = erlang(2, 1.0)

print("\nB1. Poisson MAP + Erlang-2 service must equal Pollaczek-Khinchine (M/PH/1)")
worst = 0.0
for rho in RHOS:
    lq = solve_qbd(*poisson_map(rho).as_qbd_input(), srv)["Lq"]
    pk = theoretical_Lq_pollaczek_khinchine(rho, rho, 1.0, 0.5)
    worst = max(worst, abs(lq - pk))
    print(f"  rho={rho:<5} QBD {lq:9.5f}   P-K {pk:9.5f}")
print(f"  largest difference {worst:.1e}")

print("\nB2. Erlang-2 renewal MAP must equal the PH/PH/1 solver")
worst = 0.0
for rho in RHOS:
    a = solve_qbd(*ph_renewal_map(erlang(2, 1 / rho)).as_qbd_input(), srv)["Lq"]
    b = solve_phph1(erlang(2, 1 / rho), srv)["Lq"]
    worst = max(worst, abs(a - b))
print(f"  largest difference over all rho {worst:.1e}")

print("\nB3. P(empty) = 1 - rho must still hold with bursty MAP arrivals")
for rho in RHOS:
    r = solve_qbd(*bursty_mmpp2(rho).as_qbd_input(), srv)
    print(f"  rho={rho:<5} P(empty) {r['P_empty']:.6f}   1 - rho {1 - rho:.6f}   (solver rho {r['rho']:.6f})")


def brute_force_L(Mp, srv, n_max):
    """Build the full generator of (number in system, MAP phase, service phase), truncated
    at n_max customers, and solve pi Q = 0 directly. Independent of the matrix-geometric R."""
    ma, ms = Mp.m, srv.m
    D0, D1, S, s, beta = Mp.D0, Mp.D1, srv.T, srv.t, srv.alpha

    def idx(n, i, j=0):
        return i if n == 0 else ma + (n - 1) * ma * ms + i * ms + j

    size = ma + n_max * ma * ms
    Q = np.zeros((size, size))
    for i in range(ma):
        for k in range(ma):
            if k != i:
                Q[idx(0, i), idx(0, k)] += D0[i, k]
            for j in range(ms):
                Q[idx(0, i), idx(1, k, j)] += D1[i, k] * beta[j]
    for n in range(1, n_max + 1):
        for i in range(ma):
            for j in range(ms):
                r = idx(n, i, j)
                for k in range(ma):
                    if k != i:
                        Q[r, idx(n, k, j)] += D0[i, k]
                    if n < n_max:
                        Q[r, idx(n + 1, k, j)] += D1[i, k]
                for l in range(ms):
                    if l != j:
                        Q[r, idx(n, i, l)] += S[j, l]
                if n == 1:
                    Q[r, idx(0, i)] += s[j]
                else:
                    for l in range(ms):
                        Q[r, idx(n - 1, i, l)] += s[j] * beta[l]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    A = np.vstack([Q.T, np.ones(size)])
    pi = np.linalg.lstsq(A, np.r_[np.zeros(size), 1.0], rcond=None)[0]
    L = sum(n * pi[idx(n, 0):idx(n, 0) + ma * ms].sum() for n in range(1, n_max + 1))
    return L


print("\nB4. Independent brute-force solution of the truncated Markov chain (bursty MAP)")
for rho, n_max in [(0.3, 150), (0.5, 250), (0.7, 600), (0.8, 900)]:
    L_qbd = solve_qbd(*bursty_mmpp2(rho).as_qbd_input(), srv)["L"]
    L_bf = brute_force_L(bursty_mmpp2(rho), srv, n_max)
    print(f"  rho={rho:<5} QBD L {L_qbd:9.5f}   brute force L {L_bf:9.5f}   diff {abs(L_qbd - L_bf):.1e}")

print("\nWhat the bursty MAP does to the queue (exact L_q, Erlang-2 service):")
print(f"  {'rho':>5} {'M/PH/1':>9} {'PH/PH/1':>9} {'MAP/PH/1':>9}")
for rho in RHOS:
    mph = solve_qbd(*poisson_map(rho).as_qbd_input(), srv)["Lq"]
    phph = solve_phph1(erlang(2, 1 / rho), srv)["Lq"]
    mapph = solve_qbd(*bursty_mmpp2(rho).as_qbd_input(), srv)["Lq"]
    print(f"  {rho:5.2f} {mph:9.3f} {phph:9.3f} {mapph:9.3f}")