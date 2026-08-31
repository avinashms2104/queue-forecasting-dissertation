"""
M/M/1 Queue Time-Series Simulator
==================================

Simulates an M/M/1 queue (Poisson arrivals, exponential service, single
server, FCFS) and records the system as a TIME SERIES sampled at regular
observation intervals -- this is what makes the output usable for a
forecasting problem, rather than just a steady-state summary.

For each observation time t_k, we record:
    - queue_length          : number of customers WAITING (excludes the one in service)
    - num_in_system         : total number of customers in the system (waiting + in service)
    - avg_waiting_time      : average waiting time of customers who DEPARTED during this interval
                               (NaN if nobody departed in that interval)
    - arrivals_in_interval  : number of new arrivals during (t_{k-1}, t_k]
    - departures_in_interval: number of departures during (t_{k-1}, t_k]
    - recent_arrival_rate   : estimated arrival rate over a trailing rolling window
    - recent_service_rate   : estimated service (departure) rate over a trailing rolling window

Method
------
Rather than a full event-loop simulator, we exploit the fact that an M/M/1
queue's sample path is fully determined by the arrival times and service
times:

    start_service[i] = max(arrival_time[i], departure_time[i-1])
    waiting_time[i]  = start_service[i] - arrival_time[i]
    departure_time[i] = start_service[i] + service_time[i]

This gives an exact (not approximate) simulated sample path, which we then
sample on a regular time grid to build the time series.

Usage
-----
    python mm1_simulator.py

Produces one CSV per traffic intensity (rho) plus one combined CSV, all
written to ../data/ relative to this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Fixed simulation settings
# ----------------------------------------------------------------------
MU = 1.0              # service rate is fixed at 1 (time is measured in units of mean service time)
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
N_RUNS = 10            # independent replications per rho
DT = 1.0               # observation interval (sampling step for the time series)
ROLLING_WINDOW = 10.0  # trailing window (in time units) used to estimate "recent" arrival/service rates

# Relaxation time (how long the queue takes to reach steady state, and how
# slowly it decorrelates) scales roughly like 1/(1-rho)^2 for an M/M/1
# queue. A fixed warmup/sim_time that's fine at rho=0.7 is NOT long enough
# at rho=0.95 -- we saw this empirically (simulated mean queue length was
# ~30% below the theoretical value at rho=0.95 with a fixed 3000-unit run).
# So we scale both warmup and simulated horizon with 1/(1-rho)^2.
BASE_WARMUP = 200.0
BASE_SIM_TIME = 3000.0


def scaled_lengths(rho):
    scale = 1.0 / (1.0 - rho) ** 2
    base_scale = 1.0 / (1.0 - 0.7) ** 2  # rho=0.7 is our "reference" case, scale=1x there
    factor = max(scale / base_scale, 1.0)
    return BASE_WARMUP * factor, BASE_SIM_TIME * factor


OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def simulate_mm1_path(lam, mu, total_time, seed):
    """
    Generate one exact M/M/1 sample path.

    Returns arrival_time, start_service, departure_time, waiting_time
    arrays for every customer that arrives during [0, total_time].
    """
    rng = np.random.default_rng(seed)

    # Over-generate interarrival times so we're sure to cover total_time,
    # then trim. Expected number of arrivals in total_time is lam*total_time.
    n_est = int(lam * total_time * 1.3) + 100
    interarrival = rng.exponential(1.0 / lam, size=n_est)
    arrival_time = np.cumsum(interarrival)

    # extend further if we didn't generate enough (rare, safety net)
    while arrival_time[-1] < total_time:
        extra = rng.exponential(1.0 / lam, size=n_est)
        arrival_time = np.concatenate([arrival_time, arrival_time[-1] + np.cumsum(extra)])

    arrival_time = arrival_time[arrival_time <= total_time]
    n = len(arrival_time)

    service_time = rng.exponential(1.0 / mu, size=n)

    # Lindley recursion for a single-server FCFS queue:
    #   start_service[i] = max(arrival_time[i], departure_time[i-1])
    #   departure_time[i] = start_service[i] + service_time[i]
    # This is inherently sequential (each departure depends on the previous
    # one), so we use a fast compiled loop via numpy's accumulate-style
    # trick is not directly available for max-recursions, so we loop in
    # plain Python but keep n manageable (a few thousand per run).
    start_service = np.empty(n)
    departure_time = np.empty(n)
    prev_departure = 0.0
    at = arrival_time
    st = service_time
    for i in range(n):
        start = at[i] if at[i] > prev_departure else prev_departure
        start_service[i] = start
        prev_departure = start + st[i]
        departure_time[i] = prev_departure

    waiting_time = start_service - arrival_time
    return arrival_time, start_service, departure_time, waiting_time


def build_time_series(arrival_time, departure_time, waiting_time,
                       warmup, sim_time, dt, rolling_window):
    """
    Sample the exact M/M/1 sample path on a regular observation grid,
    starting after `warmup` and running for `sim_time` further time units.

    Vectorized using searchsorted (arrival_time and departure_time are both
    already non-decreasing for a single-server FCFS queue), so this scales
    to long simulations without a per-timestep Python loop.
    """
    obs_times = np.arange(dt, sim_time + dt / 2, dt)   # t = dt, 2dt, ..., sim_time (relative)
    abs_obs_times = obs_times + warmup                  # absolute simulation time
    prev_obs_times = abs_obs_times - dt                 # start of each interval (previous grid point)

    # --- number in system at each observation time ---
    # count of arrivals <= t minus count of departures <= t
    arrivals_up_to = np.searchsorted(arrival_time, abs_obs_times, side="right")
    departures_up_to = np.searchsorted(departure_time, abs_obs_times, side="right")
    n_sys = np.maximum(arrivals_up_to - departures_up_to, 0)
    queue_len = np.maximum(n_sys - 1, 0)

    # --- arrivals / departures strictly within (prev_t, t] ---
    arrivals_up_to_prev = np.searchsorted(arrival_time, prev_obs_times, side="right")
    departures_up_to_prev = np.searchsorted(departure_time, prev_obs_times, side="right")
    n_arrivals = arrivals_up_to - arrivals_up_to_prev
    n_departures = departures_up_to - departures_up_to_prev

    # --- average waiting time of customers who departed within (prev_t, t] ---
    # waiting_time is indexed in the same (departure) order as departure_time
    cum_wait = np.concatenate([[0.0], np.cumsum(waiting_time)])
    sum_wait_interval = cum_wait[departures_up_to] - cum_wait[departures_up_to_prev]
    with np.errstate(invalid="ignore"):
        avg_wait = np.where(n_departures > 0, sum_wait_interval / np.maximum(n_departures, 1), np.nan)

    # --- rolling-window rate estimates (trailing window ending at t) ---
    w_start = abs_obs_times - rolling_window
    arrivals_up_to_wstart = np.searchsorted(arrival_time, w_start, side="right")
    departures_up_to_wstart = np.searchsorted(departure_time, w_start, side="right")
    recent_arrival_rate = (arrivals_up_to - arrivals_up_to_wstart) / rolling_window
    recent_service_rate = (departures_up_to - departures_up_to_wstart) / rolling_window

    df = pd.DataFrame({
        "t": obs_times,
        "queue_length": queue_len,
        "num_in_system": n_sys,
        "avg_waiting_time": avg_wait,
        "arrivals_in_interval": n_arrivals,
        "departures_in_interval": n_departures,
        "recent_arrival_rate": recent_arrival_rate,
        "recent_service_rate": recent_service_rate,
    })
    return df


def run_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for rho in RHOS:
        lam = rho * MU
        warmup, sim_time = scaled_lengths(rho)
        rho_dfs = []
        for run_id in range(N_RUNS):
            seed = hash((rho, run_id)) % (2**32)
            total_time = warmup + sim_time
            arrival_time, start_service, departure_time, waiting_time = simulate_mm1_path(
                lam, MU, total_time, seed
            )
            df = build_time_series(
                arrival_time, departure_time, waiting_time,
                warmup, sim_time, DT, ROLLING_WINDOW
            )
            df.insert(0, "run_id", run_id)
            df.insert(0, "rho", rho)
            rho_dfs.append(df)

        rho_df = pd.concat(rho_dfs, ignore_index=True)
        out_path = OUT_DIR / f"mm1_timeseries_rho{rho}.csv"
        rho_df.to_csv(out_path, index=False)
        print(f"rho={rho}: saved {len(rho_df)} rows -> {out_path.name} "
              f"(mean queue_length={rho_df['queue_length'].mean():.3f}, "
              f"theory L_q={rho**2/(1-rho):.3f})")

        all_dfs.append(rho_df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = OUT_DIR / "mm1_timeseries_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined dataset: {len(combined)} rows -> {combined_path.name}")
    return combined


if __name__ == "__main__":
    run_all()
