"""
Autocorrelation analysis for M/PH/1 (Erlang-k service): extends
mg1_autocorrelation_analysis.py to check whether the rising-persistence-
with-rho pattern holds for Erlang-2 service, and places it alongside the
other service-time distributions run so far.

Erlang-2 has squared coefficient of variation C_s^2 = 0.5, so it sits
between the Weibull shape=2.0 case (C_s^2 ~ 0.27) and M/M/1 (C_s^2 = 1.0);
Weibull shape=0.5 has C_s^2 = 5.

Usage
-----
    python modeling/mph1_autocorrelation_analysis.py

Produces:
    results/plots/mph1_autocorrelation_by_rho.png
    results/mph1_autocorrelation_summary.csv
And prints a half-life comparison across service-time distributions, using
results/autocorrelation_summary.csv (M/M/1) and
results/mg1_autocorrelation_summary.csv (M/G/1) if they exist.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mph1"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
ERLANG_KS = [2]
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
    acf_by_k_rho = {}

    for k in ERLANG_KS:
        for rho in RHOS:
            raw = pd.read_csv(DATA_DIR / f"mph1_timeseries_erlang{k}_rho{rho}.csv")
            raw = raw[raw["run_id"].isin(TRAIN_RUN_IDS)]

            run_acfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                run_acfs.append(autocorrelation_for_run(g["queue_length"], MAX_LAG))

            mean_acf = np.nanmean(np.array(run_acfs), axis=0)
            acf_by_k_rho[(k, rho)] = mean_acf

            below_half = np.where(mean_acf < 0.5)[0]
            half_life = int(below_half[0]) + 1 if len(below_half) > 0 else np.nan

            summary_rows.append({"erlang_k": k, "rho": rho,
                                 "acf_lag1": mean_acf[0], "acf_lag5": mean_acf[4],
                                 "acf_lag10": mean_acf[9], "acf_lag20": mean_acf[19],
                                 "half_life_lag": half_life})
            print(f"erlang{k}, rho={rho}: ACF(1)={mean_acf[0]:.3f}, "
                  f"ACF(10)={mean_acf[9]:.3f}, ACF(20)={mean_acf[19]:.3f}, "
                  f"half-life = {half_life}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "mph1_autocorrelation_summary.csv", index=False)
    print("\nSaved: results/mph1_autocorrelation_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    lags = np.arange(1, MAX_LAG + 1)
    for k in ERLANG_KS:
        for rho in RHOS:
            ax.plot(lags, acf_by_k_rho[(k, rho)], marker="o", markersize=3, label=f"rho={rho}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag (time units)")
    ax.set_ylabel("Autocorrelation of queue_length")
    ax.set_title("M/PH/1 (Erlang-2) queue-length persistence vs. traffic intensity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mph1_autocorrelation_by_rho.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/plots/mph1_autocorrelation_by_rho.png")

    print_comparison(summary_df)
    return summary_df


def print_comparison(mph1_df):
    """Half-life of persistence (lags until ACF < 0.5) by service-time variance."""
    tables = {}
    p = RESULTS_DIR / "autocorrelation_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        if {"rho", "half_life_lag"} <= set(d.columns):
            tables["M/M/1  (C_s^2=1.0)"] = d.set_index("rho")["half_life_lag"]
    p = RESULTS_DIR / "mg1_autocorrelation_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        for shape, cs2 in [(2.0, "0.27"), (0.5, "5.0")]:
            sub = d[d["weibull_shape"] == shape]
            tables[f"Weibull {shape} (C_s^2={cs2})"] = sub.set_index("rho")["half_life_lag"]
    tables["Erlang-2 (C_s^2=0.5)"] = mph1_df.set_index("rho")["half_life_lag"]

    order = sorted(tables, key=lambda n: float(n.split("C_s^2=")[1].rstrip(")")))
    comp = pd.DataFrame({name: tables[name] for name in order})
    print("\n" + "=" * 90)
    print("HALF-LIFE OF QUEUE-LENGTH PERSISTENCE (lags until autocorrelation < 0.5)")
    print("columns ordered by service-time variance, lowest to highest")
    print("=" * 90)
    print(comp.to_string())


if __name__ == "__main__":
    compute_all()