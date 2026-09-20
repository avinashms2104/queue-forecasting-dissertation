"""
M/PH/1 Queue Time-Series Simulator (Erlang-k service; k=2 as Azam specified)
============================================================================

Poisson arrivals, single server, FCFS, PHASE-TYPE service times.

Erlang-k service = the sum of k independent exponential phases, each with
rate k*mu. Equivalently a Gamma(shape=k, scale=mean/k) distribution, which
is what numpy draws below. With mean service time 1 (mu=1, as in the
M/M/1 and M/G/1 simulators):

    E[S]   = 1
    Var(S) = 1/k
    C_s^2  = 1/k          -> Erlang-2: C_s^2 = 0.5

So Erlang-2 has LOWER variability than exponential (C_s^2 = 1) but HIGHER
than the Weibull shape=2.0 case used for M/G/1 (C_s^2 ~ 0.27). It slots
between M/M/1 and that case on the service-variability scale.

Everything except how service times are drawn is reused from
mg1_simulator.py (Lindley recursion, time-series builder, warmup scaling,
Pollaczek-Khinchine validation), so results stay directly comparable.

Usage
-----
    python simulation/mph1_simulator.py

Writes ../data/mph1/mph1_timeseries_erlang{k}_rho{rho}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from mg1_simulator import (
    build_time_series,
    scaled_lengths,
    theoretical_Lq_pollaczek_khinchine,
    RHOS,
    MEAN_SERVICE_TIME,
    N_RUNS,
    DT,
    ROLLING_WINDOW,
)

ERLANG_KS = [2]     # Azam specified Erlang-2; add e.g. 4 later for lower variance

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mph1"


def simulate_mph1_path(lam, k, total_time, seed):
    """One exact M/PH/1 (Erlang-k service) sample path via the Lindley recursion."""
    rng = np.random.default_rng(seed)

    n_est = int(lam * total_time * 1.3) + 100
    arrival_time = np.cumsum(rng.exponential(1.0 / lam, size=n_est))
    while arrival_time[-1] < total_time:
        extra = rng.exponential(1.0 / lam, size=n_est)
        arrival_time = np.concatenate([arrival_time, arrival_time[-1] + np.cumsum(extra)])
    arrival_time = arrival_time[arrival_time <= total_time]
    n = len(arrival_time)

    # Erlang-k = Gamma(shape=k, scale=mean/k) = sum of k Exp(rate k/mean) phases
    service_time = rng.gamma(shape=k, scale=MEAN_SERVICE_TIME / k, size=n)

    start_service = np.empty(n)
    departure_time = np.empty(n)
    prev_departure = 0.0
    for i in range(n):
        start = arrival_time[i] if arrival_time[i] > prev_departure else prev_departure
        start_service[i] = start
        prev_departure = start + service_time[i]
        departure_time[i] = prev_departure

    waiting_time = start_service - arrival_time
    return arrival_time, start_service, departure_time, waiting_time


def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for k in ERLANG_KS:
        var_service = MEAN_SERVICE_TIME ** 2 / k
        cv_squared = var_service / MEAN_SERVICE_TIME ** 2      # = 1/k
        print(f"\n=== Erlang-{k} service === mean={MEAN_SERVICE_TIME:.3f}, "
              f"variance={var_service:.4f}, C_s^2={cv_squared:.3f} "
              f"(exponential reference: C_s^2=1.0)")

        for rho in RHOS:
            lam = rho
            warmup, sim_time = scaled_lengths(rho, cv_squared=cv_squared)

            rho_dfs = []
            for run_id in range(N_RUNS):
                seed = hash((k, rho, run_id)) % (2 ** 32)
                arrival_time, start_service, departure_time, waiting_time = simulate_mph1_path(
                    lam, k, warmup + sim_time, seed)
                df = build_time_series(arrival_time, departure_time, waiting_time,
                                       warmup, sim_time, DT, ROLLING_WINDOW)
                df.insert(0, "run_id", run_id)
                df.insert(0, "rho", rho)
                df.insert(0, "erlang_k", k)
                rho_dfs.append(df)

            rho_df = pd.concat(rho_dfs, ignore_index=True)
            out_path = OUT_DIR / f"mph1_timeseries_erlang{k}_rho{rho}.csv"
            rho_df.to_csv(out_path, index=False)

            sim_Lq = rho_df["queue_length"].mean()
            th_Lq = theoretical_Lq_pollaczek_khinchine(rho, lam, MEAN_SERVICE_TIME, var_service)
            print(f"  rho={rho}: saved {len(rho_df)} rows -> {out_path.name} | warmup={warmup:.0f} | "
                  f"simulated Lq={sim_Lq:.3f}, theoretical Lq (P-K)={th_Lq:.3f}, "
                  f"error={100 * (sim_Lq - th_Lq) / th_Lq:+.2f}%")


if __name__ == "__main__":
    run_all()