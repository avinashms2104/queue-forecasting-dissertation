"""
Step 2 (MAP/PH/1 version): Build ML-ready forecasting datasets from the
MAP/PH/1 (bursty MMPP-2 arrivals, Erlang-2 service) time series.

Reuses the feature construction and run-level train/val/test split from
build_mg1_datasets.py, exactly as build_phph1_datasets.py does, so the
MAP/PH/1 datasets are built in the same way as M/M/1, M/G/1, M/PH/1 and
PH/PH/1. The congestion threshold (80th percentile of queue length) is
computed from the 20 TRAINING runs only, as for PH/PH/1; the all-runs value
is printed next to it for comparison.

Usage
-----
    python modeling/build_mapph1_datasets.py

Produces, per rho:
    data/mapph1/processed/history_only_rho{rho}.csv
    data/mapph1/processed/rich_features_rho{rho}.csv
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mapph1"
OUT_DIR = DATA_DIR / "processed"


def build_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_lag_cols = [f"queue_length_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols = []
    for col in ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                "recent_arrival_rate", "recent_service_rate"]:
        rich_extra_cols += [f"{col}_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols += ["recent_utilisation_lag0"]

    target_cols = [f"target_qlen_{n}" for n in HORIZONS] + [f"target_congested_{n}" for n in HORIZONS]

    for rho in RHOS:
        raw = pd.read_csv(DATA_DIR / f"mapph1_timeseries_rho{rho}.csv")

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

        base_cols = ["arrival_process", "service_erlang_k", "rho", "run_id", "t",
                     "congestion_threshold", "split"]
        history_df = feat[base_cols + history_lag_cols + target_cols].copy()
        rich_df = feat[base_cols + history_lag_cols + rich_extra_cols + target_cols].copy()

        tag = f"rho{rho}"
        history_df.to_csv(OUT_DIR / f"history_only_{tag}.csv", index=False)
        rich_df.to_csv(OUT_DIR / f"rich_features_{tag}.csv", index=False)

        print(f"MAP/PH/1, rho={rho}: {len(feat)} usable rows | "
              f"split counts = {feat['split'].value_counts().to_dict()} | "
              f"threshold (train runs) = {threshold:.2f}, (all runs, old method) = {threshold_all_runs:.2f} | "
              f"medium-horizon congestion rate = {feat['target_congested_medium'].mean():.1%}")


if __name__ == "__main__":
    build_all()