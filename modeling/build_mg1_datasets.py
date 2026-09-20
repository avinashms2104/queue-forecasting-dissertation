"""
Step 2 (M/G/1 version): Build ML-ready forecasting datasets from the M/G/1
time series.

Mirrors modeling/build_datasets.py, but loops over (weibull_shape, rho)
combinations instead of just rho, and reads/writes from data/mg1/ instead
of data/. Kept as a separate file (rather than generalising
build_datasets.py) to avoid risking the already-validated M/M/1 pipeline
under time pressure -- see notes in the dissertation notebook / commit
history for the reasoning. Candidate for unification with build_datasets.py
once M/PH/1 is also in place next week.

Congestion threshold is computed separately per (shape, rho) combination,
consistent with the M/M/1 approach of defining "congested" relative to
each system's own observed distribution.

Train/validation/test split: identical run-level logic to build_datasets.py
(run_id 0-19 -> train, 20-24 -> val, 25-29 -> test), since the M/G/1
simulator also produces 30 independent runs per (shape, rho).

Usage
-----
    python build_mg1_datasets.py

Produces, per (shape, rho):
    data/mg1/processed/history_only_shape{shape}_rho{rho}.csv
    data/mg1/processed/rich_features_shape{shape}_rho{rho}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mg1"
OUT_DIR = DATA_DIR / "processed"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
WEIBULL_SHAPES = [2.0, 0.5]

LOOKBACK = 10
HORIZONS = {
    "short": 5,
    "medium": 10,
    "long": 20,
    "extra_long": 15,
}
CONGESTION_PERCENTILE = 0.80

N_TRAIN_RUNS = 20
N_VAL_RUNS = 5
# remaining runs (25-29) are test


def congestion_threshold(queue_length_series, percentile=CONGESTION_PERCENTILE):
    return queue_length_series.quantile(percentile)


def assign_split(run_id):
    if run_id < N_TRAIN_RUNS:
        return "train"
    elif run_id < N_TRAIN_RUNS + N_VAL_RUNS:
        return "val"
    else:
        return "test"


def build_features_for_run(df_run, threshold):
    """Identical logic to build_datasets.py's version."""
    df = df_run.copy().reset_index(drop=True)

    for lag in range(LOOKBACK):
        df[f"queue_length_lag{lag}"] = df["queue_length"].shift(lag)

    rich_cols = ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                 "recent_arrival_rate", "recent_service_rate"]
    for col in rich_cols:
        for lag in range(LOOKBACK):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    with np.errstate(divide="ignore", invalid="ignore"):
        recent_util = df["recent_arrival_rate"] / df["recent_service_rate"].replace(0, np.nan)
    df["recent_utilisation_lag0"] = recent_util

    for name, h in HORIZONS.items():
        df[f"target_qlen_{name}"] = df["queue_length"].shift(-h)
        df[f"target_congested_{name}"] = (df[f"target_qlen_{name}"] > threshold).astype("float")

    lag_cols = [c for c in df.columns if "_lag" in c]
    target_cols = [c for c in df.columns if c.startswith("target_qlen_")]
    df = df.dropna(subset=lag_cols + target_cols).reset_index(drop=True)

    for name in HORIZONS:
        df[f"target_congested_{name}"] = df[f"target_congested_{name}"].astype(int)

    return df


def build_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_lag_cols = [f"queue_length_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols = []
    for col in ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                "recent_arrival_rate", "recent_service_rate"]:
        rich_extra_cols += [f"{col}_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols += ["recent_utilisation_lag0"]

    target_cols = [f"target_qlen_{n}" for n in HORIZONS] + [f"target_congested_{n}" for n in HORIZONS]

    for shape in WEIBULL_SHAPES:
        for rho in RHOS:
            raw_path = DATA_DIR / f"mg1_timeseries_shape{shape}_rho{rho}.csv"
            raw = pd.read_csv(raw_path)

            threshold = congestion_threshold(raw["queue_length"])

            run_feature_dfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                feat = build_features_for_run(g, threshold)
                run_feature_dfs.append(feat)

            rho_feat = pd.concat(run_feature_dfs, ignore_index=True)
            rho_feat["congestion_threshold"] = threshold
            rho_feat["split"] = rho_feat["run_id"].apply(assign_split)

            base_cols = ["weibull_shape", "rho", "run_id", "t", "congestion_threshold", "split"]

            history_df = rho_feat[base_cols + history_lag_cols + target_cols].copy()
            rich_df = rho_feat[base_cols + history_lag_cols + rich_extra_cols + target_cols].copy()

            history_df.to_csv(OUT_DIR / f"history_only_shape{shape}_rho{rho}.csv", index=False)
            rich_df.to_csv(OUT_DIR / f"rich_features_shape{shape}_rho{rho}.csv", index=False)

            congestion_rate = rho_feat["target_congested_medium"].mean()
            split_counts = rho_feat["split"].value_counts().to_dict()
            print(f"shape={shape}, rho={rho}: {len(rho_feat)} usable rows | "
                  f"split counts = {split_counts} | "
                  f"congestion threshold (queue_length) = {threshold:.2f} | "
                  f"medium-horizon congestion rate = {congestion_rate:.1%}")


if __name__ == "__main__":
    build_all()