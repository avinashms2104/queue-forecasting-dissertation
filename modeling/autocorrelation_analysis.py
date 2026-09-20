"""
Autocorrelation analysis: queue-length persistence vs. traffic intensity.

Per Azam's feedback (2026-09-01 email), the low classification recall at
low rho (and the improvement at high rho) is likely NOT because congestion
becomes more "typical" at high rho (it doesn't -- the congestion rate is
roughly constant, ~9-19%, by construction of the percentile-based
definition). Instead, the likely explanation is that congestion becomes
more temporally PERSISTENT at high rho: once the queue is congested, it
tends to stay that way for longer, before recovering to a non-congested
state.

This script quantifies that directly by computing the autocorrelation of
queue_length at each rho, as a function of lag, using the (corrected)
30-run simulated time series. A queue with longer persistence will show
autocorrelation that decays more slowly with lag.

Usage
-----
    python autocorrelation_analysis.py

Produces:
    results/plots/autocorrelation_by_rho.png
    results/autocorrelation_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
MAX_LAG = 30          # compute autocorrelation for lags 1..MAX_LAG (time units)
TRAIN_RUN_IDS = list(range(20))  # use only the 20 training runs, consistent
                                   # with the rest of the pipeline; avoids
                                   # touching val/test data at this stage


def autocorrelation_for_run(series, max_lag):
    """
    Standard (biased) sample autocorrelation of a single run's queue_length
    series, for lags 1..max_lag. Returns an array of length max_lag.
    """
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
    acf_by_rho = {}

    for rho in RHOS:
        raw = pd.read_csv(DATA_DIR / f"mm1_timeseries_rho{rho}.csv")
        raw = raw[raw["run_id"].isin(TRAIN_RUN_IDS)]

        run_acfs = []
        for run_id, g in raw.groupby("run_id"):
            g = g.sort_values("t")
            run_acfs.append(autocorrelation_for_run(g["queue_length"], MAX_LAG))

        run_acfs = np.array(run_acfs)  # shape (n_runs, MAX_LAG)
        mean_acf = run_acfs.mean(axis=0)
        acf_by_rho[rho] = mean_acf

        # a simple summary statistic: the lag at which autocorrelation first
        # drops below 0.5 (a rough "half-life" of persistence); NaN if it
        # never drops below 0.5 within MAX_LAG
        below_half = np.where(mean_acf < 0.5)[0]
        half_life = int(below_half[0]) + 1 if len(below_half) > 0 else np.nan

        summary_rows.append({"rho": rho, "acf_lag1": mean_acf[0], "acf_lag5": mean_acf[4],
                             "acf_lag10": mean_acf[9], "acf_lag20": mean_acf[19],
                             "half_life_lag": half_life})
        print(f"rho={rho}: ACF(1)={mean_acf[0]:.3f}, ACF(5)={mean_acf[4]:.3f}, "
              f"ACF(10)={mean_acf[9]:.3f}, ACF(20)={mean_acf[19]:.3f}, "
              f"half-life (lag where ACF<0.5) = {half_life}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "autocorrelation_summary.csv", index=False)
    print(f"\nSaved: results/autocorrelation_summary.csv")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    lags = np.arange(1, MAX_LAG + 1)
    for rho in RHOS:
        ax.plot(lags, acf_by_rho[rho], marker="o", markersize=3, label=f"rho={rho}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, label="ACF = 0.5")
    ax.set_xlabel("Lag (time units)")
    ax.set_ylabel("Autocorrelation of queue_length")
    ax.set_title("Queue-length persistence vs. traffic intensity\n"
                 "(slower decay = longer-lasting congestion episodes)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "autocorrelation_by_rho.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/plots/autocorrelation_by_rho.png")

    return summary_df


if __name__ == "__main__":
    compute_all()