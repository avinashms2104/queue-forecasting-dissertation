"""
Step 2 (PH/PH/1 version): Build ML-ready forecasting datasets from the
PH/PH/1 (Erlang-2 arrivals, Erlang-2 service) time series.

Reuses the feature construction and run-level train/val/test split from
build_mg1_datasets.py (which mirrors build_datasets.py), so the PH/PH/1
datasets are built in exactly the same way as M/M/1, M/G/1 and M/PH/1.

ONE deliberate change: training-runs-only congestion threshold
----------------------------------------------------------------
Earlier pipelines computed the 80th-percentile threshold from ALL 30 runs,
including the 5 test runs (flagged in the notebook, Section 7.4, as a small
departure from strict test separation). Here the threshold is computed from
the 20 TRAINING runs only. The all-runs threshold is still printed next to
it, so the size of the change can be checked directly.

Usage
-----
    python modeling/build_phph1_datasets.py

Produces, per (arrival k, service k, rho):
    data/phph1/processed/history_only_arrE{ka}_srvE{ks}_rho{rho}.csv
    data/phph1/processed/rich_features_arrE{ka}_srvE{ks}_rho{rho}.csv
"""

import pandas as pd
from pathlib import Path

from build_mg1_datasets import (
    build_features_for_run,
    congestion_threshold,
    assign_split,
    LOOKBACK,
    HORIZONS,
    RHOS,
)
from build_datasets import N_TRAIN_RUNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "phph1"
OUT_DIR = DATA_DIR / "processed"

SETTINGS = [(2, 2)]     # (arrival Erlang order, service Erlang order), as in phph1_simulator.py


def build_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_lag_cols = [f"queue_length_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols = []
    for col in ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                "recent_arrival_rate", "recent_service_rate"]:
        rich_extra_cols += [f"{col}_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols += ["recent_utilisation_lag0"]

    target_cols = [f"target_qlen_{n}" for n in HORIZONS] + [f"target_congested_{n}" for n in HORIZONS]

    for ka, ks in SETTINGS:
        for rho in RHOS:
            raw = pd.read_csv(DATA_DIR / f"phph1_timeseries_arrE{ka}_srvE{ks}_rho{rho}.csv")

            # threshold from the 20 training runs only (run_id 0-19)
            threshold = congestion_threshold(raw.loc[raw["run_id"] < N_TRAIN_RUNS, "queue_length"])
            threshold_all_runs = congestion_threshold(raw["queue_length"])   # old method, for comparison

            run_feature_dfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                run_feature_dfs.append(build_features_for_run(g, threshold))

            feat = pd.concat(run_feature_dfs, ignore_index=True)
            feat["congestion_threshold"] = threshold
            feat["split"] = feat["run_id"].apply(assign_split)

            base_cols = ["arrival_erlang_k", "service_erlang_k", "rho", "run_id", "t",
                         "congestion_threshold", "split"]
            history_df = feat[base_cols + history_lag_cols + target_cols].copy()
            rich_df = feat[base_cols + history_lag_cols + rich_extra_cols + target_cols].copy()

            tag = f"arrE{ka}_srvE{ks}_rho{rho}"
            history_df.to_csv(OUT_DIR / f"history_only_{tag}.csv", index=False)
            rich_df.to_csv(OUT_DIR / f"rich_features_{tag}.csv", index=False)

            print(f"arrE{ka}/srvE{ks}, rho={rho}: {len(feat)} usable rows | "
                  f"split counts = {feat['split'].value_counts().to_dict()} | "
                  f"threshold (train runs) = {threshold:.2f}, (all runs, old method) = {threshold_all_runs:.2f} | "
                  f"medium-horizon congestion rate = {feat['target_congested_medium'].mean():.1%}")


if __name__ == "__main__":
    build_all()