"""
Does computing the congestion threshold from ALL 30 runs (as M/M/1, M/G/1 and
M/PH/1 did) instead of from the 20 TRAINING runs only change anything?

Notebook Section 7.4 flags this as a small departure from strict test
separation: the 80th-percentile threshold used to LABEL congestion was
estimated with the 5 test runs included. PH/PH/1 already uses the
training-runs-only threshold. This script checks, for every older queue type
and every rho, whether the two thresholds differ. Because queue length is an
integer, the threshold is an integer-valued percentile, so if the two values
are equal, every congestion label -- and therefore every classification
result -- is exactly identical and nothing needs to be rebuilt or retrained.

Reads only the raw simulated time series in data/ (no models, seconds to run).

Usage:  python modeling/check_threshold_leakage.py
Writes: results/threshold_leakage_check.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
N_TRAIN_RUNS = 20          # run_id 0-19 are training runs (same split as build_datasets.py)
PERCENTILE = 0.80          # same as CONGESTION_PERCENTILE in build_datasets.py

STAGES = [
    ("M/M/1", lambda rho: DATA / f"mm1_timeseries_rho{rho}.csv"),
    ("M/G/1 Weibull 2.0", lambda rho: DATA / "mg1" / f"mg1_timeseries_shape2.0_rho{rho}.csv"),
    ("M/G/1 Weibull 0.5", lambda rho: DATA / "mg1" / f"mg1_timeseries_shape0.5_rho{rho}.csv"),
    ("M/PH/1 Erlang-2", lambda rho: DATA / "mph1" / f"mph1_timeseries_erlang2_rho{rho}.csv"),
    ("PH/PH/1 E2/E2", lambda rho: DATA / "phph1" / f"phph1_timeseries_arrE2_srvE2_rho{rho}.csv"),
]

rows = []
for stage, path_fn in STAGES:
    for rho in RHOS:
        path = path_fn(rho)
        if not path.exists():
            print(f"missing: {path}")
            continue
        df = pd.read_csv(path, usecols=["run_id", "queue_length"])
        t_all = df["queue_length"].quantile(PERCENTILE)
        t_train = df.loc[df["run_id"] < N_TRAIN_RUNS, "queue_length"].quantile(PERCENTILE)
        # how many observations would be labelled differently (queue length > threshold)?
        lab_all = df["queue_length"] > t_all
        lab_train = df["queue_length"] > t_train
        rows.append(dict(queue=stage, rho=rho, threshold_all_runs=t_all, threshold_train_only=t_train,
                         differs=bool(t_all != t_train),
                         labels_changed=int((lab_all != lab_train).sum()), n_obs=len(df)))

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "threshold_leakage_check.csv", index=False)

pd.set_option("display.width", 200)
print(res.pivot(index="rho", columns="queue",
                values="threshold_all_runs").astype(str).add(" / ").add(
    res.pivot(index="rho", columns="queue", values="threshold_train_only").astype(str)).to_string())
print("\n(each cell: threshold using all 30 runs / threshold using the 20 training runs only)")
print(f"\nSettings checked: {len(res)}   settings where the two thresholds differ: {int(res.differs.sum())}")
print(f"Observations relabelled in total: {int(res.labels_changed.sum())} of {int(res.n_obs.sum())}")
print("\nSaved: results/threshold_leakage_check.csv")