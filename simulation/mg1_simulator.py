"""
M/G/1 Queue Time-Series Simulator
==================================

Simulates an M/G/1 queue: Poisson arrivals (same as M/M/1), single server,
FCFS, but with a GENERAL (non-exponential) service-time distribution.

Distribution choice
--------------------
Azam did not specify a distribution for M/G/1 (only Erlang-2 was
specified, for next week's M/PH/1 stage). We use WEIBULL, for two reasons:
  1. It nests the exponential distribution as the special case shape=1,
     so it directly demonstrates what happens as we move away from the
     memoryless assumption that made M/M/1 tractable (see background
     chapter, Section "M/G/1: relaxing the service-time assumption").
  2. Its hazard rate is monotonic in shape: shape>1 gives an INCREASING
     hazard rate (a customer who has been in service longer becomes MORE
     likely to finish soon), shape<1 gives a DECREASING hazard rate (a
     customer who has been in service longer becomes LESS likely to
     finish soon -- occasional very long service times). This lets us
     show the effect of service-time variability in both directions.

We use two shape settings, both with MEAN SERVICE TIME = 1 (matching
mu=1 from the M/M/1 simulator, so results are directly comparable):
    shape = 2.0  -> LOWER variance than exponential (more regular service)
    shape = 0.5  -> HIGHER variance than exponential (more bursty service)
(shape = 1.0 would recover M/M/1 exactly, as a sanity check if needed.)

Method
------
Same exact-sample-path approach as the M/M/1 simulator: the Lindley
recursion is distribution-agnostic, so we only need to change how service
times are drawn.

Validation against theory
--------------------------
For M/G/1, the Pollaczek-Khinchine formula gives the exact theoretical
mean queue length:
    Wq = lambda * E[S^2] / (2 * (1 - rho))
    Lq = lambda * Wq
where E[S^2] = Var(S) + E[S]^2 is the second moment of the service-time
distribution. We compute this exactly for the Weibull distribution and
compare it to the simulated mean queue length, the same validation
approach used for M/M/1.

Unexpected observation: warmup scaling needed to account for variance too
---------------------------------------------------------------------------
An initial version of this simulator scaled the warmup period the same
way as M/M/1 (proportional to 1/(1-rho)^2 only). This gave good agreement
with theory almost everywhere, EXCEPT at shape=0.5 (high service-time
variance), rho=0.95, where the simulated mean queue length was 17.66%
below the theoretical Pollaczek-Khinchine value -- the largest error by
far. The cause is the same general phenomenon documented for M/M/1
(relaxation time scaling with rho), but for a general service-time
distribution the relaxation time ALSO depends on the service-time
variability: a more variable service process takes longer to reach
steady state, on top of the usual rho effect. The fix scales the warmup
by an additional (1 + C_s^2)/2 factor, where C_s^2 = Var(S)/E[S]^2 is the
squared coefficient of variation of the service-time distribution
(C_s^2=1 recovers the M/M/1 case exactly, since the exponential
distribution has C_s^2=1).

Usage
-----
    python mg1_simulator.py

Produces one CSV per (shape, rho) combination, written to
../data/mg1/ relative to this script.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Fixed simulation settings (mirrors the corrected mm1_simulator.py)
# ----------------------------------------------------------------------
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
WEIBULL_SHAPES = [2.0, 0.5]     # see module docstring for rationale
MEAN_SERVICE_TIME = 1.0          # matches mu=1 in the M/M/1 simulator
N_RUNS = 30                      # independent replications per (shape, rho)
DT = 1.0
ROLLING_WINDOW = 10.0

BASE_WARMUP = 200.0
POST_WARMUP_LENGTH = 20_000.0    # fixed for every rho, consistent with M/M/1 design

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mg1"


def weibull_scale_for_mean(shape, target_mean=MEAN_SERVICE_TIME):
    """
    For X ~ Weibull(shape, scale), E[X] = scale * Gamma(1 + 1/shape).
    Solve for the scale that gives the desired mean.
    """
    return target_mean / math.gamma(1.0 + 1.0 / shape)


def weibull_variance(shape, scale):
    """Var(X) = scale^2 * [Gamma(1+2/shape) - Gamma(1+1/shape)^2]."""
    g1 = math.gamma(1.0 + 1.0 / shape)
    g2 = math.gamma(1.0 + 2.0 / shape)
    return scale ** 2 * (g2 - g1 ** 2)


def theoretical_Lq_pollaczek_khinchine(rho, lam, mean_service, var_service):
    """
    Pollaczek-Khinchine formula (M/G/1):
        Wq = lambda * E[S^2] / (2*(1-rho))
        Lq = lambda * Wq
    """
    second_moment = var_service + mean_service ** 2
    Wq = lam * second_moment / (2 * (1 - rho))
    Lq = lam * Wq
    return Lq


def scaled_lengths(rho, cv_squared=1.0):
    """
    Relaxation-time scaling for M/G/1. The base 1/(1-rho)^2 factor is
    carried over from M/M/1, but for a general service-time distribution
    the relaxation time also depends on the service-time variability
    (higher-variance service processes take longer to reach steady
    state). We approximate this with an extra (1 + cv_squared)/2 factor,
    where cv_squared = Var(S)/E[S]^2 is the squared coefficient of
    variation of service time (cv_squared=1.0 recovers the M/M/1 case
    exactly, since exponential has C_s^2=1).

    This was added after an initial run showed a 17.66% validation error
    at shape=0.5 (high-variance service), rho=0.95 -- the M/M/1-only
    warmup scaling was not long enough once high rho and high service-time
    variance combined. See the corresponding "unexpected observation" in
    the module docstring and the dissertation notebook.
    """
    rho_scale = 1.0 / (1.0 - rho) ** 2
    base_rho_scale = 1.0 / (1.0 - 0.7) ** 2
    variability_factor = (1.0 + cv_squared) / 2.0
    factor = max((rho_scale / base_rho_scale) * variability_factor, 1.0)
    warmup = BASE_WARMUP * factor
    sim_time = POST_WARMUP_LENGTH
    return warmup, sim_time


def simulate_mg1_path(lam, shape, scale, total_time, seed):
    """
    Generate one exact M/G/1 sample path (Poisson arrivals, Weibull
    service times), using the same Lindley recursion as M/M/1.
    """
    rng = np.random.default_rng(seed)

    n_est = int(lam * total_time * 1.3) + 100
    interarrival = rng.exponential(1.0 / lam, size=n_est)
    arrival_time = np.cumsum(interarrival)

    while arrival_time[-1] < total_time:
        extra = rng.exponential(1.0 / lam, size=n_est)
        arrival_time = np.concatenate([arrival_time, arrival_time[-1] + np.cumsum(extra)])

    arrival_time = arrival_time[arrival_time <= total_time]
    n = len(arrival_time)

    # numpy's Generator.weibull(a) draws from a Weibull distribution with
    # shape a and scale 1; multiply by our target scale.
    service_time = rng.weibull(shape, size=n) * scale

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
    """Identical to the M/M/1 version -- distribution-agnostic."""
    obs_times = np.arange(dt, sim_time + dt / 2, dt)
    abs_obs_times = obs_times + warmup
    prev_obs_times = abs_obs_times - dt

    arrivals_up_to = np.searchsorted(arrival_time, abs_obs_times, side="right")
    departures_up_to = np.searchsorted(departure_time, abs_obs_times, side="right")
    n_sys = np.maximum(arrivals_up_to - departures_up_to, 0)
    queue_len = np.maximum(n_sys - 1, 0)

    arrivals_up_to_prev = np.searchsorted(arrival_time, prev_obs_times, side="right")
    departures_up_to_prev = np.searchsorted(departure_time, prev_obs_times, side="right")
    n_arrivals = arrivals_up_to - arrivals_up_to_prev
    n_departures = departures_up_to - departures_up_to_prev

    cum_wait = np.concatenate([[0.0], np.cumsum(waiting_time)])
    sum_wait_interval = cum_wait[departures_up_to] - cum_wait[departures_up_to_prev]
    with np.errstate(invalid="ignore"):
        avg_wait = np.where(n_departures > 0, sum_wait_interval / np.maximum(n_departures, 1), np.nan)

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

    for shape in WEIBULL_SHAPES:
        scale = weibull_scale_for_mean(shape)
        var_service = weibull_variance(shape, scale)
        cv_squared = var_service / (MEAN_SERVICE_TIME ** 2)
        print(f"\n=== Weibull shape={shape} (scale={scale:.4f}) === "
              f"mean service time={MEAN_SERVICE_TIME:.3f}, "
              f"variance={var_service:.4f}, C_s^2={cv_squared:.3f} "
              f"(exponential reference: C_s^2=1.0)")

        for rho in RHOS:
            lam = rho  # mu = 1/MEAN_SERVICE_TIME = 1, so lambda = rho * mu = rho
            warmup, sim_time = scaled_lengths(rho, cv_squared=cv_squared)

            rho_dfs = []
            for run_id in range(N_RUNS):
                seed = hash((shape, rho, run_id)) % (2 ** 32)
                total_time = warmup + sim_time
                arrival_time, start_service, departure_time, waiting_time = simulate_mg1_path(
                    lam, shape, scale, total_time, seed
                )
                df = build_time_series(
                    arrival_time, departure_time, waiting_time,
                    warmup, sim_time, DT, ROLLING_WINDOW
                )
                df.insert(0, "run_id", run_id)
                df.insert(0, "rho", rho)
                df.insert(0, "weibull_shape", shape)
                rho_dfs.append(df)

            rho_df = pd.concat(rho_dfs, ignore_index=True)
            out_path = OUT_DIR / f"mg1_timeseries_shape{shape}_rho{rho}.csv"
            rho_df.to_csv(out_path, index=False)

            simulated_Lq = rho_df["queue_length"].mean()
            theoretical_Lq = theoretical_Lq_pollaczek_khinchine(rho, lam, MEAN_SERVICE_TIME, var_service)
            pct_error = 100 * (simulated_Lq - theoretical_Lq) / theoretical_Lq if theoretical_Lq > 0 else float("nan")

            print(f"  rho={rho}: saved {len(rho_df)} rows -> {out_path.name} | "
                  f"warmup={warmup:.0f} | "
                  f"simulated Lq={simulated_Lq:.3f}, theoretical Lq (P-K)={theoretical_Lq:.3f}, "
                  f"error={pct_error:+.2f}%")


if __name__ == "__main__":
    run_all()