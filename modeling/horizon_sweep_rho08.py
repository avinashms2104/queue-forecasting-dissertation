"""
Wider forecast-horizon sweep at a single representative rho (0.8), per
Azam's suggestion: three-to-four horizon points aren't enough to say
whether MAE degradation is linear or eventually levels off.

Only touches rho=0.8 raw M/M/1 data (data/mm1_timeseries_rho0.8.csv), so
this is fast to run -- no need to redo the full (dataset x rho x horizon)
sweep for this.

Usage
-----
    python horizon_sweep_rho08.py

Produces:
    results/horizon_sweep_rho0.8.csv
    results/plots/horizon_sweep_rho0.8.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHO = 0.8
LOOKBACK = 10
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40]
RANDOM_SEED = 42
TEST_RUN_IDS = [25, 26, 27, 28, 29]
N_TRAIN_RUNS = 20
N_VAL_RUNS = 5


def assign_split(run_id):
    if run_id < N_TRAIN_RUNS:
        return "train"
    elif run_id < N_TRAIN_RUNS + N_VAL_RUNS:
        return "val"
    else:
        return "test"


def build_dataset_for_horizon(raw, h):
    run_dfs = []
    for run_id, g in raw.groupby("run_id"):
        g = g.sort_values("t").reset_index(drop=True)
        df = g.copy()
        for lag in range(LOOKBACK):
            df[f"lag{lag}"] = df["queue_length"].shift(lag)
        df["target"] = df["queue_length"].shift(-h)
        lag_cols = [f"lag{lag}" for lag in range(LOOKBACK)]
        df = df.dropna(subset=lag_cols + ["target"]).reset_index(drop=True)
        df["split"] = assign_split(run_id)
        run_dfs.append(df[["run_id", "split"] + lag_cols + ["target"]])
    return pd.concat(run_dfs, ignore_index=True)


def run_sweep():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_DIR / f"mm1_timeseries_rho{RHO}.csv")
    lag_cols = [f"lag{lag}" for lag in range(LOOKBACK)]

    rows = []
    for h in HORIZONS:
        df = build_dataset_for_horizon(raw, h)
        train_df = df[df["split"] == "train"]
        X_train, y_train = train_df[lag_cols], train_df["target"]

        model = RandomForestRegressor(n_estimators=30, max_depth=6,
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)

        run_maes = []
        for test_run_id in TEST_RUN_IDS:
            test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
            if len(test_df) == 0:
                continue
            pred = model.predict(test_df[lag_cols])
            run_maes.append(mean_absolute_error(test_df["target"], pred))

        naive_maes = []
        for test_run_id in TEST_RUN_IDS:
            test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
            if len(test_df) == 0:
                continue
            naive_pred = test_df["lag0"].values
            naive_maes.append(mean_absolute_error(test_df["target"], naive_pred))

        mean_mae = np.mean(run_maes)
        mean_naive = np.mean(naive_maes)
        rel_improvement = 1 - (mean_mae / mean_naive) if mean_naive > 0 else np.nan

        rows.append({"h": h, "RF_MAE": mean_mae, "RF_MAE_std": np.std(run_maes),
                     "naive_MAE": mean_naive,
                     "relative_improvement": rel_improvement})
        print(f"h={h}: RF MAE={mean_mae:.3f}+/-{np.std(run_maes):.3f}, "
              f"naive MAE={mean_naive:.3f}, rel.improvement={rel_improvement:+.1%}")

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(RESULTS_DIR / "horizon_sweep_rho0.8.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["h"], sweep_df["RF_MAE"], "o-", label="Random Forest")
    ax.plot(sweep_df["h"], sweep_df["naive_MAE"], "k:", label="Naive (persistence)")
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("MAE (queue length, rho=0.8)")
    ax.set_title("Wider horizon sweep at rho=0.8: does MAE degradation level off?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "horizon_sweep_rho0.8.png", dpi=130)
    plt.close(fig)

    print("\nSaved: results/horizon_sweep_rho0.8.csv")
    print("Saved: results/plots/horizon_sweep_rho0.8.png")


if __name__ == "__main__":
    run_sweep()