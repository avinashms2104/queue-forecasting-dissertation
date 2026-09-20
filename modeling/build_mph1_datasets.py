"""
Step 2 (M/PH/1 version): Build ML-ready forecasting datasets from the
M/PH/1 (Erlang-k service) time series.

Reuses the feature construction, congestion definition and run-level
train/val/test split from build_mg1_datasets.py (which itself mirrors
build_datasets.py), so M/M/1, M/G/1 and M/PH/1 datasets are built in
exactly the same way and results stay comparable.

Congestion threshold: 80th percentile of queue length, computed per
(erlang_k, rho), same as the M/M/1 and M/G/1 pipelines.

Usage
-----
    python modeling/build_mph1_datasets.py

Produces, per (k, rho):
    data/mph1/processed/history_only_erlang{k}_rho{rho}.csv
    data/mph1/processed/rich_features_erlang{k}_rho{rho}.csv
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mph1"
OUT_DIR = DATA_DIR / "processed"

ERLANG_KS = [2]


def build_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_lag_cols = [f"queue_length_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols = []
    for col in ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                "recent_arrival_rate", "recent_service_rate"]:
        rich_extra_cols += [f"{col}_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols += ["recent_utilisation_lag0"]

    target_cols = [f"target_qlen_{n}" for n in HORIZONS] + [f"target_congested_{n}" for n in HORIZONS]

    for k in ERLANG_KS:
        for rho in RHOS:
            raw = pd.read_csv(DATA_DIR / f"mph1_timeseries_erlang{k}_rho{rho}.csv")
            threshold = congestion_threshold(raw["queue_length"])

            run_feature_dfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                run_feature_dfs.append(build_features_for_run(g, threshold))

            feat = pd.concat(run_feature_dfs, ignore_index=True)
            feat["congestion_threshold"] = threshold
            feat["split"] = feat["run_id"].apply(assign_split)

            base_cols = ["erlang_k", "rho", "run_id", "t", "congestion_threshold", "split"]
            history_df = feat[base_cols + history_lag_cols + target_cols].copy()
            rich_df = feat[base_cols + history_lag_cols + rich_extra_cols + target_cols].copy()

            history_df.to_csv(OUT_DIR / f"history_only_erlang{k}_rho{rho}.csv", index=False)
            rich_df.to_csv(OUT_DIR / f"rich_features_erlang{k}_rho{rho}.csv", index=False)

            print(f"erlang{k}, rho={rho}: {len(feat)} usable rows | "
                  f"split counts = {feat['split'].value_counts().to_dict()} | "
                  f"congestion threshold (queue_length) = {threshold:.2f} | "
                  f"medium-horizon congestion rate = {feat['target_congested_medium'].mean():.1%}")


if __name__ == "__main__":
    build_all()