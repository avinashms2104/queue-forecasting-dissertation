# M/M/1 Time-Series Simulation (Step 1)

## What this is
An exact M/M/1 (Poisson arrivals, exponential service, 1 server, FCFS) sample-path
simulator that outputs a **time series** — not just steady-state numbers — suitable
for the forecasting problem in the dissertation.

## Files
- `simulation/mm1_simulator.py` — the simulator. Run with `python mm1_simulator.py`.
- `data/mm1_timeseries_rho{0.3,0.5,0.7,0.8,0.9,0.95}.csv` — one file per traffic
  intensity, 10 independent runs each, columns:
  - `rho`, `run_id`, `t` — traffic intensity, replication id, observation time
  - `queue_length` — number waiting (excludes customer in service)
  - `num_in_system` — total in system
  - `avg_waiting_time` — mean waiting time of customers who departed in this interval (NaN if none did)
  - `arrivals_in_interval`, `departures_in_interval`
  - `recent_arrival_rate`, `recent_service_rate` — trailing rolling-window rate estimates (window = 10 time units)
- `results/plots/validation_sim_vs_theory.png` — mean simulated queue length vs. theoretical L_q = rho^2/(1-rho) across all six rho levels
- `results/plots/example_timeseries.png` — example queue-length trajectories at rho=0.5 vs rho=0.9, showing the qualitative difference in congestion behaviour

## Validation
Simulated mean queue length matches queueing theory (L_q = rho^2/(1-rho)) within
a few percent at every rho level, including rho=0.95:

| rho  | simulated mean L_q | theoretical L_q |
|------|--------------------|-----------------|
| 0.30 | 0.137              | 0.129           |
| 0.50 | 0.459              | 0.500           |
| 0.70 | 1.738              | 1.633           |
| 0.80 | 2.959              | 3.200           |
| 0.90 | 7.635              | 8.100           |
| 0.95 | 17.533             | 18.050          |

**Note on methodology:** the relaxation time of an M/M/1 queue (how long it takes
to reach steady state / how slowly it decorrelates) scales roughly like
1/(1-rho)^2. A fixed warmup + simulation length that works fine at rho=0.7 is
NOT long enough at rho=0.95 — an early version of this simulator underestimated
the mean queue length by ~30% at rho=0.95 for exactly this reason. The final
version scales both the warmup period and total simulated time with
1/(1-rho)^2 to compensate. This is a good "unexpected observation" to note in
the dissertation notebook per Azam's request.

## A note on file size for GitHub
The rho=0.9 and rho=0.95 files are ~11MB and ~47MB respectively (they need much
longer simulated runs to converge, hence far more observation rows). Total repo
data folder is ~63MB. This is under GitHub's 100MB hard limit but is a lot to
push directly — worth deciding whether to keep these in the repo as-is, use
Git LFS, or regenerate them locally via the script instead of committing the
CSVs. Flagging this now before it becomes friction later.

## Next step
Step 2: build ML-ready forecasting datasets from these time series (history-only
vs. richer feature set, congestion labels, multiple horizons).
