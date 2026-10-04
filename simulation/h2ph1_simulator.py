"""
H2/PH/1 Queue Time-Series Simulator (hyperexponential arrivals, Erlang-2 service)
================================================================================

A second, burstier PH/PH/1 setting (open question 1 to Azam), designed as a
control for MAP/PH/1:

    H2/PH/1    arrival gaps C_a^2 = 1.684, gaps INDEPENDENT (renewal process)
    MAP/PH/1   arrival gaps C_a^2 = 1.684, gaps CORRELATED (lag-1 corr 0.165)

The hyperexponential is fitted (balanced means, two moments) to exactly the same
mean and C_a^2 as the MMPP-2 used in MAP/PH/1, and service is the same Erlang-2,
so the ONLY difference between the two queues is the correlation between
successive arrival gaps. Compared with PH/PH/1 (Erlang-2 arrivals, C_a^2 = 0.5)
and M/PH/1 (Poisson, C_a^2 = 1), it completes the arrival-side ladder:
smooth -> random -> variable -> variable and correlated.

Exact L_q (QBD), Erlang-2 service:
    rho     H2/PH/1   MAP/PH/1
    0.3      0.127     0.233
    0.8      3.437     8.542
    0.95    19.652    46.774
i.e. with the same C_a^2, correlation alone makes the queue about 2.5x longer.

Method: identical to mapph1_simulator.py. The H2 renewal process is written as a
MAP (D0 = T, D1 = t alpha, see map_utils.ph_renewal_map), so the same sampler,
Lindley recursion, warm-up rule ((IDC + C_s^2)/2, here IDC = C_a^2 because the
gaps are independent) and exact QBD validation are reused unchanged.

Usage
-----
    python simulation/h2ph1_simulator.py

Writes ../data/h2ph1/h2ph1_timeseries_rho{rho}.csv
and    ../results/h2ph1_validation.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from mg1_simulator import (
    build_time_series, RHOS, MEAN_SERVICE_TIME, N_RUNS, DT, ROLLING_WINDOW,
)
from ph_utils import erlang, hyperexp2_balanced
from map_utils import ph_renewal_map, bursty_mmpp2
from qbd_solver import solve_qbd
from mapph1_simulator import simulate_mapph1_path, scaled_lengths_mapph1, lag1_corr

SERVICE_ERLANG_K = 2
ARRIVAL_LABEL = "H2"
SEED_TAG = 88                                   # distinct from PH/PH/1 and MAP/PH/1 seeds
ARRIVAL_SCV = bursty_mmpp2(1.0).scv             # 1.684..., matched to the MAP/PH/1 arrivals

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "h2ph1"
RES_DIR = ROOT / "results"


def make_arrivals(rho):
    """H2 renewal arrivals at rate lambda = rho, written as a MAP."""
    lam = rho / MEAN_SERVICE_TIME
    return ph_renewal_map(hyperexp2_balanced(1.0 / lam, ARRIVAL_SCV))


def run_all(write_csv=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    srv = erlang(SERVICE_ERLANG_K, MEAN_SERVICE_TIME)
    summary = []

    probe = make_arrivals(0.7)
    print(f"\n=== H2/PH/1: hyperexponential arrivals / Erlang-{SERVICE_ERLANG_K} service ===")
    print(f"    arrival gaps: C_a^2={probe.scv:.3f}, lag-1 corr={probe.lag_corr(1):.3f}, "
          f"IDC={probe.idc_limit():.3f} | service C_s^2={srv.scv:.3f}")

    for rho in RHOS:
        arr = make_arrivals(rho)
        warmup, sim_time = scaled_lengths_mapph1(rho, arr.idc_limit(), srv.scv)
        exact = solve_qbd(*arr.as_qbd_input(), srv)

        rho_dfs, run_lq, ia_all, ia_corr = [], [], [], []
        for run_id in range(N_RUNS):
            seed = hash((SEED_TAG, SERVICE_ERLANG_K, rho, run_id)) % (2 ** 32)
            arrival_time, start_service, departure_time, waiting_time, ia = simulate_mapph1_path(
                arr, srv, warmup + sim_time, seed)
            df = build_time_series(arrival_time, departure_time, waiting_time,
                                   warmup, sim_time, DT, ROLLING_WINDOW)
            df.insert(0, "run_id", run_id)
            df.insert(0, "rho", rho)
            df.insert(0, "service_erlang_k", SERVICE_ERLANG_K)
            df.insert(0, "arrival_process", ARRIVAL_LABEL)
            rho_dfs.append(df)
            run_lq.append(df["queue_length"].mean())
            ia_all.append(ia)
            ia_corr.append(lag1_corr(ia))

        rho_df = pd.concat(rho_dfs, ignore_index=True)
        if write_csv:
            rho_df.to_csv(OUT_DIR / f"h2ph1_timeseries_rho{rho}.csv", index=False)

        ia_all = np.concatenate(ia_all)
        run_lq = np.array(run_lq)
        sim_Lq = rho_df["queue_length"].mean()
        se = run_lq.std(ddof=1) / np.sqrt(N_RUNS)
        z = (sim_Lq - exact["Lq"]) / se
        summary.append(dict(
            arrival_process=ARRIVAL_LABEL, service_erlang_k=SERVICE_ERLANG_K, rho=rho, warmup=warmup,
            interarrival_mean_sim=ia_all.mean(), interarrival_mean_exact=arr.mean,
            interarrival_scv_sim=ia_all.var() / ia_all.mean() ** 2, interarrival_scv_exact=arr.scv,
            lag1_corr_sim=float(np.mean(ia_corr)), lag1_corr_exact=arr.lag_corr(1),
            Lq_sim=sim_Lq, Lq_run_std=run_lq.std(ddof=1), Lq_qbd=exact["Lq"],
            error_pct=100 * (sim_Lq - exact["Lq"]) / exact["Lq"], z=z))
        s = summary[-1]
        print(f"  rho={rho}: {len(rho_df)} rows | warmup={warmup:.0f} | "
              f"C_a^2 sim={s['interarrival_scv_sim']:.3f} | lag-1 corr sim={s['lag1_corr_sim']:.3f} | "
              f"simulated Lq={sim_Lq:.3f} (run std {s['Lq_run_std']:.3f}), "
              f"exact Lq (QBD)={exact['Lq']:.3f}, error={s['error_pct']:+.2f}%, z={z:+.2f}")

    res = pd.DataFrame(summary)
    res.to_csv(RES_DIR / "h2ph1_validation.csv", index=False)
    print(f"\n|z| > 2 in {int((res.z.abs() > 2).sum())} of {len(res)} settings "
          f"(largest |z| = {res.z.abs().max():.2f})")
    print("Saved validation summary -> results/h2ph1_validation.csv")


if __name__ == "__main__":
    run_all()