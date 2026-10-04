"""
MAP/PH/1 Queue Time-Series Simulator (bursty MAP arrivals, Erlang-2 service)
===========================================================================

Final stage of Azam's plan: M/M/1 -> M/G/1 -> M/PH/1 -> PH/PH/1 -> MAP/PH/1.
Arrivals now come from a Markovian arrival process (MAP), so the gaps between
arrivals can be CORRELATED: short gaps cluster together and arrivals come in
bursts. This is the one thing PH/PH/1 could not do.

Arrival process
---------------
Azam has not specified a MAP, so we use a two-state Markov-modulated Poisson
process (bursty_mmpp2 in map_utils.py): a busy spell with 4x the arrival rate
of a quiet spell, busy 25% of the time, about 10 arrivals per busy spell.
Gap scv = 1.68, lag-1 correlation = 0.165, limiting index of dispersion = 4.6.
All rates scale with lambda = rho, so the shape is the same at every rho.
Service is Erlang-2 with mean 1, exactly as in M/PH/1 and PH/PH/1, so the
only change from those stages is the arrival process. Any other MAP (for
example one fitted to the call-centre data) only means editing make_arrivals().

Method
------
Identical to the earlier simulators: the MAP gives consecutive arrival gaps
(sampled in chunks that carry the phase over, so the correlation is never
broken), the Lindley recursion gives the exact sample path, and
build_time_series() from mg1_simulator.py (unchanged) gives the snapshots.

Warm-up scaling
---------------
PH/PH/1 used the Kingman variability factor (C_a^2 + C_s^2)/2. With correlated
arrivals the right measure of arrival variability is the limiting index of
dispersion (IDC), which equals C_a^2 for any renewal process and also counts
the correlation. So the factor becomes (IDC + C_s^2)/2. Everything else
(BASE_WARMUP, 1/(1-rho)^2 scaling relative to rho=0.7, floor at 1, fixed
20,000 recorded time units) is unchanged.

Validation
----------
Against the EXACT QBD solution (qbd_solver.solve_qbd with D0, D1), which was
checked for MAP input in check_map_utils.py: against Pollaczek-Khinchine,
against the PH/PH/1 solver, and against a brute-force solution of the full
Markov chain. z = (simulated - exact) / standard error over the 30 runs.

Usage
-----
    python simulation/mapph1_simulator.py

Writes ../data/mapph1/mapph1_timeseries_rho{rho}.csv
and    ../results/mapph1_validation.csv
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
from map_utils import bursty_mmpp2
from qbd_solver import solve_qbd

SERVICE_ERLANG_K = 2
ARRIVAL_LABEL = "MMPP2"          # written into every row, so the datasets say which MAP was used
SEED_TAG = 77                    # integer, keeps hash()-based seeds reproducible and distinct from PH/PH/1

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "mapph1"
RES_DIR = ROOT / "results"


def make_arrivals(rho):
    """The arrival MAP at arrival rate lambda = rho (mean service time is 1)."""
    return bursty_mmpp2(rho * 1.0 / MEAN_SERVICE_TIME)


def scaled_lengths_mapph1(rho, idc, cs2):
    """phph1 warm-up rule with (IDC + C_s^2)/2 as the variability factor."""
    rho_scale = 1.0 / (1.0 - rho) ** 2
    base_rho_scale = 1.0 / (1.0 - 0.7) ** 2
    variability_factor = (idc + cs2) / 2.0
    factor = max((rho_scale / base_rho_scale) * variability_factor, 1.0)
    return BASE_WARMUP * factor, POST_WARMUP_LENGTH


def simulate_mapph1_path(arr_map, srv_ph, total_time, seed):
    """One exact MAP/PH/1 sample path via the Lindley recursion."""
    rng = np.random.default_rng(seed)

    # start the phase process in its stationary distribution, then keep the phase
    # across chunks so consecutive gaps stay correlated
    phase = int(rng.choice(arr_map.m, p=arr_map.theta))
    chunk = int(total_time * arr_map.lam * 1.2) + 100
    gaps_list, t_end = [], 0.0
    while t_end < total_time:
        g, phase = arr_map.sample(chunk, rng, phase0=phase)
        gaps_list.append(g)
        t_end += g.sum()
    interarrival = np.concatenate(gaps_list)
    arrival_time = np.cumsum(interarrival)
    keep = arrival_time <= total_time
    arrival_time = arrival_time[keep]
    interarrival = interarrival[keep]
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
    return arrival_time, start_service, departure_time, waiting_time, interarrival


def lag1_corr(x):
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def run_all(write_csv=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    srv = erlang(SERVICE_ERLANG_K, MEAN_SERVICE_TIME)
    summary = []

    probe = make_arrivals(0.7)
    print(f"\n=== MAP/PH/1: {probe.name} arrivals / Erlang-{SERVICE_ERLANG_K} service ===")
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
            rho_df.to_csv(OUT_DIR / f"mapph1_timeseries_rho{rho}.csv", index=False)

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
    res.to_csv(RES_DIR / "mapph1_validation.csv", index=False)
    print(f"\n|z| > 2 in {int((res.z.abs() > 2).sum())} of {len(res)} settings "
          f"(largest |z| = {res.z.abs().max():.2f})")
    print("Saved validation summary -> results/mapph1_validation.csv")


if __name__ == "__main__":
    run_all()