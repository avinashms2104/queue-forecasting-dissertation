"""
Autocorrelation analysis for M/G/1: extends autocorrelation_analysis.py
(M/M/1) to check whether the same rising-persistence-with-rho pattern
holds across both Weibull shape settings, and how the two variance
regimes compare to each other and to M/M/1.

Usage
-----
    python mg1_autocorrelation_analysis.py

Produces:
    results/plots/mg1_autocorrelation_by_rho.png (one panel per shape)
    results/mg1_autocorrelation_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mg1"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
WEIBULL_SHAPES = [2.0, 0.5]
MAX_LAG = 30
TRAIN_RUN_IDS = list(range(20))


def autocorrelation_for_run(series, max_lag):
    x = series.values.astype(float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.full(max_lag, np.nan)
    acf = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = np.dot(x[:-lag], x[lag:]) / (n * var)
    return acf


def compute_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    acf_by_shape_rho = {}

    for shape in WEIBULL_SHAPES:
        for rho in RHOS:
            raw = pd.read_csv(DATA_DIR / f"mg1_timeseries_shape{shape}_rho{rho}.csv")
            raw = raw[raw["run_id"].isin(TRAIN_RUN_IDS)]

            run_acfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                run_acfs.append(autocorrelation_for_run(g["queue_length"], MAX_LAG))

            run_acfs = np.array(run_acfs)
            mean_acf = run_acfs.mean(axis=0)
            acf_by_shape_rho[(shape, rho)] = mean_acf

            below_half = np.where(mean_acf < 0.5)[0]
            half_life = int(below_half[0]) + 1 if len(below_half) > 0 else np.nan

            summary_rows.append({"weibull_shape": shape, "rho": rho,
                                 "acf_lag1": mean_acf[0], "acf_lag5": mean_acf[4],
                                 "acf_lag10": mean_acf[9], "acf_lag20": mean_acf[19],
                                 "half_life_lag": half_life})
            print(f"shape={shape}, rho={rho}: ACF(1)={mean_acf[0]:.3f}, "
                  f"ACF(10)={mean_acf[9]:.3f}, ACF(20)={mean_acf[19]:.3f}, "
                  f"half-life = {half_life}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "mg1_autocorrelation_summary.csv", index=False)
    print("\nSaved: results/mg1_autocorrelation_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    lags = np.arange(1, MAX_LAG + 1)
    for ax, shape in zip(axes, WEIBULL_SHAPES):
        for rho in RHOS:
            ax.plot(lags, acf_by_shape_rho[(shape, rho)], marker="o", markersize=3,
                    label=f"rho={rho}")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Lag (time units)")
        ax.set_title(f"Weibull shape={shape}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Autocorrelation of queue_length")
    fig.suptitle("M/G/1 queue-length persistence vs. traffic intensity, both service-time variance settings")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mg1_autocorrelation_by_rho.png", dpi=130)
    plt.close(fig)
    print("Saved: results/plots/mg1_autocorrelation_by_rho.png")

    return summary_df


if __name__ == "__main__":
    compute_all()