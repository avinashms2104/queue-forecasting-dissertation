"""
Step 2: Build ML-ready forecasting datasets from the M/M/1 time series.

This turns the raw simulation output (data/mm1_timeseries_rho*.csv) into
supervised-learning datasets for the forecasting task:

    Given what we know up to time t, predict the queue length at time t+h,
    for a few forecast horizons h (short / medium / long), and also predict
    whether the system will be CONGESTED at time t+h.

Two feature sets are built, so we can directly test Azam's question of
whether richer queueing features earn their keep over queue-length history
alone:

    Dataset A ("history-only"):
        just lagged queue_length values: Q_t, Q_{t-1}, ..., Q_{t-k+1}

    Dataset B ("rich"):
        the same lagged queue_length history, PLUS lagged versions of
        num_in_system, arrivals_in_interval, departures_in_interval,
        recent_arrival_rate, recent_service_rate, and a derived
        "recent utilisation" = recent_arrival_rate / recent_service_rate

Congestion definition (see also the writeup in Overleaf)
----------------------------------------------------------
For this first pass we use a simple, per-rho FIXED THRESHOLD definition:
a time point is "congested" if queue_length exceeds the 80th percentile of
queue_length observed for that rho (i.e., congestion = being in the top 20%
of observed queue lengths for that traffic intensity). This corresponds to
one of the four candidate definitions Azam listed ("upper 10%/20% of
observed queue lengths"), chosen because it adapts sensibly across very
different rho levels (a "congested" queue length at rho=0.3 is nowhere near
what counts as congested at rho=0.95) rather than using one fixed absolute
number for every utilisation level. The other three candidate definitions
(fixed absolute threshold, waiting-time threshold, utilisation-based
threshold) are easy to swap in later -- see `congestion_threshold()` below.

The classification TARGET is FUTURE congestion: given features known at
time t, will the system be congested at time t+h? This matches the
dissertation's framing ("use recent behaviour to predict ... whether the
system is moving towards congestion"), rather than just classifying the
current state.

Usage
-----
    python build_datasets.py

Produces, per rho and combined:
    data/processed/history_only_rho{rho}.csv
    data/processed/rich_features_rho{rho}.csv
    data/processed/history_only_all.csv
    data/processed/rich_features_all.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

LOOKBACK = 10          # how many past queue_length observations to use as input (Q_t, Q_{t-1}, ..., Q_{t-9})
HORIZONS = {            # forecast horizons, in observation steps (dt=1.0 time unit each in the simulator)
    "short": 5,
    "medium": 10,
    "long": 20,
}
CONGESTION_PERCENTILE = 0.80   # "congested" = top 20% of observed queue lengths for that rho

# The raw simulation runs are much longer at high rho (needed for the queue
# to reach steady state -- see simulation/mm1_simulator.py), which means the
# raw row count per rho ranges from ~30k (rho=0.3) to >1,000,000 (rho=0.95).
# For the ML datasets we don't need that many rows, and an unbalanced,
# huge file is both impractical (300MB+ CSVs) and would silently make the
# high-rho classes dominate any model trained on the combined data. So we
# cap rows per rho via random subsampling (fixed seed for reproducibility).
MAX_ROWS_PER_RHO = 30000
RANDOM_SEED = 42


def congestion_threshold(queue_length_series, percentile=CONGESTION_PERCENTILE):
    """
    Fixed, per-rho threshold: a queue length above this percentile counts
    as "congested". This is the "top 10%/20% of observed queue lengths"
    definition from Azam's list of candidates.
    """
    return queue_length_series.quantile(percentile)


def build_features_for_run(df_run, threshold):
    """
    df_run: rows for a single (rho, run_id), already sorted by t.
    Returns a DataFrame with lag features, forecast targets, and
    congestion labels for every horizon, with NaN boundary rows dropped.
    """
    df = df_run.copy().reset_index(drop=True)

    # --- lagged queue-length history (used by BOTH dataset A and B) ---
    for lag in range(LOOKBACK):
        df[f"queue_length_lag{lag}"] = df["queue_length"].shift(lag)
    # lag0 = current value Q_t, lag1 = Q_{t-1}, ..., lag{k-1} = Q_{t-k+1}

    # --- additional "rich" features (used only by dataset B) ---
    rich_cols = ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                 "recent_arrival_rate", "recent_service_rate"]
    for col in rich_cols:
        for lag in range(LOOKBACK):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # derived: recent utilisation estimate = recent arrival rate / recent service rate
    with np.errstate(divide="ignore", invalid="ignore"):
        recent_util = df["recent_arrival_rate"] / df["recent_service_rate"].replace(0, np.nan)
    df["recent_utilisation_lag0"] = recent_util

    # --- forecast targets and congestion labels, for each horizon ---
    for name, h in HORIZONS.items():
        df[f"target_qlen_{name}"] = df["queue_length"].shift(-h)
        df[f"target_congested_{name}"] = (df[f"target_qlen_{name}"] > threshold).astype("float")
        # keep as float for now so NaN boundary rows are preserved consistently; cast to int after dropna

    # drop rows where lag features are not fully available (start of series)
    # or where the largest-horizon target is not available (end of series)
    lag_cols = [c for c in df.columns if "_lag" in c]
    target_cols = [c for c in df.columns if c.startswith("target_qlen_")]
    df = df.dropna(subset=lag_cols + target_cols).reset_index(drop=True)

    for name in HORIZONS:
        df[f"target_congested_{name}"] = df[f"target_congested_{name}"].astype(int)

    return df


def build_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_dfs = []
    rich_dfs = []

    history_lag_cols = [f"queue_length_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols = []
    for col in ["num_in_system", "arrivals_in_interval", "departures_in_interval",
                "recent_arrival_rate", "recent_service_rate"]:
        rich_extra_cols += [f"{col}_lag{lag}" for lag in range(LOOKBACK)]
    rich_extra_cols += ["recent_utilisation_lag0"]

    target_cols = [f"target_qlen_{n}" for n in HORIZONS] + [f"target_congested_{n}" for n in HORIZONS]

    for rho in RHOS:
        raw_path = DATA_DIR / f"mm1_timeseries_rho{rho}.csv"
        raw = pd.read_csv(raw_path)

        threshold = congestion_threshold(raw["queue_length"])

        run_feature_dfs = []
        for run_id, g in raw.groupby("run_id"):
            g = g.sort_values("t")
            feat = build_features_for_run(g, threshold)
            # rho/run_id columns already present (carried over from the raw data)
            run_feature_dfs.append(feat)

        rho_feat = pd.concat(run_feature_dfs, ignore_index=True)
        rho_feat["congestion_threshold"] = threshold

        if len(rho_feat) > MAX_ROWS_PER_RHO:
            rho_feat = rho_feat.sample(n=MAX_ROWS_PER_RHO, random_state=RANDOM_SEED).sort_values(
                ["run_id", "t"]).reset_index(drop=True)

        base_cols = ["rho", "run_id", "t", "congestion_threshold"]

        history_df = rho_feat[base_cols + history_lag_cols + target_cols].copy()
        rich_df = rho_feat[base_cols + history_lag_cols + rich_extra_cols + target_cols].copy()

        history_df.to_csv(OUT_DIR / f"history_only_rho{rho}.csv", index=False)
        rich_df.to_csv(OUT_DIR / f"rich_features_rho{rho}.csv", index=False)

        congestion_rate = rho_feat["target_congested_medium"].mean()
        print(f"rho={rho}: {len(rho_feat)} usable rows | congestion threshold (queue_length) = "
              f"{threshold:.2f} | medium-horizon congestion rate = {congestion_rate:.1%}")

        history_dfs.append(history_df)
        rich_dfs.append(rich_df)

    history_all = pd.concat(history_dfs, ignore_index=True)
    rich_all = pd.concat(rich_dfs, ignore_index=True)
    history_all.to_csv(OUT_DIR / "history_only_all.csv", index=False)
    rich_all.to_csv(OUT_DIR / "rich_features_all.csv", index=False)

    print(f"\nDataset A (history-only): {history_all.shape[0]} rows, {history_all.shape[1]} columns "
          f"-> history_only_all.csv")
    print(f"Dataset B (rich features): {rich_all.shape[0]} rows, {rich_all.shape[1]} columns "
          f"-> rich_features_all.csv")

    return history_all, rich_all


if __name__ == "__main__":
    build_all()
