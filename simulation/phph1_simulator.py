"""
PH/PH/1 Queue Time-Series Simulator (Erlang-2 arrivals, Erlang-2 service)
=========================================================================

Next stage of Azam's plan: M/M/1 -> M/G/1 -> M/PH/1 -> PH/PH/1 -> MAP/PH/1.
Arrivals are now PHASE-TYPE as well as service, so neither side is Poisson.

Distribution choice
-------------------
Azam has not yet specified the arrival distribution, so we start with the
simplest case: Erlang-2 interarrival times with mean 1/lambda (C_a^2 = 0.5,
smoother than Poisson) and Erlang-2 service with mean 1 (C_s^2 = 0.5, as in
M/PH/1). The sampler in ph_utils.py works for ANY phase-type distribution,
so switching (e.g. to a hyperexponential with C_a^2 > 1) only means editing
SETTINGS below.

Method
------
Identical to the earlier simulators: Lindley recursion for the exact sample
path, then build_time_series() from mg1_simulator.py (reused unchanged).

Warm-up scaling
---------------
mg1_simulator.scaled_lengths() uses the variability factor (1 + C_s^2)/2.
With non-Poisson arrivals the natural generalisation is (C_a^2 + C_s^2)/2,
the variability term in Kingman's formula; it reduces to the old factor when
C_a^2 = 1. Everything else (BASE_WARMUP, 1/(1-rho)^2 scaling relative to
rho=0.7, floor at 1, fixed 20,000 recorded time units) is unchanged.

Validation
----------
Pollaczek-Khinchine needs Poisson arrivals, so it no longer applies. We
validate against the EXACT matrix-analytic (QBD) solution in qbd_solver.py,
which was first checked against M/M/1 and M/PH/1 (P-K) to within 1e-10.
Kingman's approximation is printed for reference only.

Usage
-----
    python simulation/phph1_simulator.py

Writes ../data/phph1/phph1_timeseries_arrE{ka}_srvE{ks}_rho{rho}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from mg1_simulator import (
    build_time_series,
    RHOS,
    MEAN_SERVICE_TIME,
    N_RUNS,
    DT,
    ROLLING_WINDOW,
    BASE_WARMUP,
    POST_WARMUP_LENGTH,
)
from ph_utils import erlang
from qbd_solver import solve_phph1, kingman_lq

# (arrival Erlang order, service Erlang order). Integers only, so the
# hash()-based seeds below stay reproducible across Python sessions.
SETTINGS = [(2, 2)]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "phph1"


def scaled_lengths_phph1(rho, ca2, cs2):
    """mg1_simulator.scaled_lengths with (C_a^2 + C_s^2)/2 as the variability factor."""
    rho_scale = 1.0 / (1.0 - rho) ** 2
    base_rho_scale = 1.0 / (1.0 - 0.7) ** 2
    variability_factor = (ca2 + cs2) / 2.0
    factor = max((rho_scale / base_rho_scale) * variability_factor, 1.0)
    return BASE_WARMUP * factor, POST_WARMUP_LENGTH


def simulate_phph1_path(arr_ph, srv_ph, total_time, seed):
    """One exact PH/PH/1 sample path via the Lindley recursion."""
    rng = np.random.default_rng(seed)

    n_est = int(total_time / arr_ph.mean * 1.3) + 100
    interarrival = arr_ph.sample(n_est, rng)
    arrival_time = np.cumsum(interarrival)
    while arrival_time[-1] < total_time:
        extra = arr_ph.sample(n_est, rng)
        interarrival = np.concatenate([interarrival, extra])
        arrival_time = np.concatenate([arrival_time, arrival_time[-1] + np.cumsum(extra)])
    keep = arrival_time <= total_time
    arrival_time = arrival_time[keep]
    n = len(arrival_time)

    service_time = srv_ph.sample(n, rng)

    start_service = np.empty(n)
    departure_time = np.empty(n)
    prev_departure = 0.0
    for i in range(n):
        start = arrival_time[i] if arrival_time[i] > prev_departure else prev_departure
        start_service[i] = start
        prev_departure = start + service_time[i]
        departure_time[i] = prev_departure

    waiting_time = start_service - arrival_time
    return arrival_time, start_service, departure_time, waiting_time, interarrival[: n]


def run_all(write_csv=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for ka, ks in SETTINGS:
        srv = erlang(ks, MEAN_SERVICE_TIME)
        print(f"\n=== Erlang-{ka} arrivals / Erlang-{ks} service === "
              f"C_a^2={1/ka:.3f}, C_s^2={srv.scv:.3f} (Poisson/exponential reference: 1.0)")

        for rho in RHOS:
            lam = rho
            arr = erlang(ka, 1.0 / lam)
            warmup, sim_time = scaled_lengths_phph1(rho, arr.scv, srv.scv)
            exact = solve_phph1(arr, srv)

            rho_dfs, run_lq, ia_all = [], [], []
            for run_id in range(N_RUNS):
                seed = hash((ka, ks, rho, run_id)) % (2 ** 32)
                arrival_time, start_service, departure_time, waiting_time, ia = simulate_phph1_path(
                    arr, srv, warmup + sim_time, seed)
                df = build_time_series(arrival_time, departure_time, waiting_time,
                                       warmup, sim_time, DT, ROLLING_WINDOW)
                df.insert(0, "run_id", run_id)
                df.insert(0, "rho", rho)
                df.insert(0, "service_erlang_k", ks)
                df.insert(0, "arrival_erlang_k", ka)
                rho_dfs.append(df)
                run_lq.append(df["queue_length"].mean())
                ia_all.append(ia)

            rho_df = pd.concat(rho_dfs, ignore_index=True)
            if write_csv:
                out_path = OUT_DIR / f"phph1_timeseries_arrE{ka}_srvE{ks}_rho{rho}.csv"
                rho_df.to_csv(out_path, index=False)

            ia_all = np.concatenate(ia_all)
            run_lq = np.array(run_lq)
            sim_Lq = rho_df["queue_length"].mean()
            se = run_lq.std(ddof=1) / np.sqrt(N_RUNS)
            z = (sim_Lq - exact["Lq"]) / se
            summary.append(dict(
                arrival_erlang_k=ka, service_erlang_k=ks, rho=rho, warmup=warmup,
                interarrival_mean_sim=ia_all.mean(), interarrival_mean_exact=arr.mean,
                interarrival_scv_sim=ia_all.var() / ia_all.mean() ** 2, interarrival_scv_exact=arr.scv,
                Lq_sim=sim_Lq, Lq_run_std=run_lq.std(ddof=1), Lq_qbd=exact["Lq"],
                error_pct=100 * (sim_Lq - exact["Lq"]) / exact["Lq"], z=z,
                Lq_kingman=kingman_lq(arr, srv)))
            print(f"  rho={rho}: {len(rho_df)} rows | warmup={warmup:.0f} | "
                  f"C_a^2 sim={summary[-1]['interarrival_scv_sim']:.3f} | "
                  f"simulated Lq={sim_Lq:.3f} (run std {run_lq.std(ddof=1):.3f}), "
                  f"exact Lq (QBD)={exact['Lq']:.3f}, error={summary[-1]['error_pct']:+.2f}%, z={z:+.2f} | "
                  f"Kingman={summary[-1]['Lq_kingman']:.3f}")

    res_dir = Path(__file__).resolve().parent.parent / "results"
    pd.DataFrame(summary).to_csv(res_dir / "phph1_validation.csv", index=False)
    print(f"\nSaved validation summary -> results/phph1_validation.csv")


if __name__ == "__main__":
    run_all()