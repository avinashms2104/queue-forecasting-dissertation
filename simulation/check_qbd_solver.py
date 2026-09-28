"""Crosscheck qbd_solver.py against formulas already used in this project.

If the QBD solver is right, then with Poisson (exponential) arrivals it must give
exactly the same Lq as:
  - M/M/1:  Lq = rho^2 / (1 - rho)                      (mm1 validation)
  - M/PH/1: Pollaczek-Khinchine from mg1_simulator.py   (mph1 validation, Table 3)
"""
from ph_utils import exponential, erlang
from qbd_solver import solve_phph1, kingman_lq
from mg1_simulator import theoretical_Lq_pollaczek_khinchine, RHOS

print("Check 1: Poisson arrivals, where the answer is already known")
print(f"{'rho':>5} | {'M/M/1 QBD':>10} {'rho^2/(1-rho)':>14} | {'M/E2/1 QBD':>11} {'P-K (your code)':>16}")
worst = 0.0
for rho in RHOS:
    lam = rho
    mm1 = solve_phph1(exponential(1 / lam), exponential(1.0))["Lq"]
    mph1 = solve_phph1(exponential(1 / lam), erlang(2, 1.0))["Lq"]
    mm1_exact = rho ** 2 / (1 - rho)
    pk = theoretical_Lq_pollaczek_khinchine(rho, lam, 1.0, 0.5)   # Erlang-2: mean 1, variance 1/2
    worst = max(worst, abs(mm1 - mm1_exact), abs(mph1 - pk))
    print(f"{rho:5.2f} | {mm1:10.4f} {mm1_exact:14.4f} | {mph1:11.4f} {pk:16.4f}")
print(f"Largest difference: {worst:.1e}  (should be tiny, e.g. below 1e-9)")

print("\nCheck 2: P(system empty) must equal 1 - rho for any stable single-server queue")
for rho in RHOS:
    p0 = solve_phph1(erlang(2, 1 / rho), erlang(2, 1.0))["P_empty"]
    print(f"  E2/E2/1 rho={rho}: P(empty) = {p0:.6f}, 1 - rho = {1 - rho:.6f}")

print("\nCheck 3: M/M/1 full distribution P(N=n) = (1-rho) rho^n, at rho = 0.8")
d = solve_phph1(exponential(1 / 0.8), exponential(1.0))["dist"]
for n in [0, 1, 5, 10]:
    print(f"  n={n:2d}: QBD {d[n]:.6f}, exact {(1 - 0.8) * 0.8 ** n:.6f}")

print("\nThe new case, E2/E2/1 (no formula exists, so QBD is the reference):")
for rho in RHOS:
    arr, srv = erlang(2, 1 / rho), erlang(2, 1.0)
    lq = solve_phph1(arr, srv)["Lq"]
    kg = kingman_lq(arr, srv)
    print(f"  rho={rho}: exact Lq = {lq:.4f}, Kingman approx = {kg:.4f} ({100 * (kg - lq) / lq:+.1f}%)")