"""
Autocorrelation analysis for PH/PH/1 (Erlang-2 arrivals, Erlang-2 service):
extends mph1_autocorrelation_analysis.py to check whether the rising-
persistence-with-rho pattern still holds once arrivals are also phase-type,
and compares directly against M/PH/1 (same service, Poisson arrivals) to
isolate the effect of arrival variance specifically.

PH/PH/1 (Erlang-2/Erlang-2) has C_a^2 = C_s^2 = 0.5, versus M/PH/1's
C_a^2 = 1.0 (Poisson), C_s^2 = 0.5. Comparing the two half-life columns at
matching rho shows what smoothing the ARRIVAL process alone does, holding
service variance fixed -- the arrival-side counterpart of the service-
variance comparison in compare_service_distributions.py.

Usage
-----
    python modeling/phph1_autocorrelation_analysis.py

Produces:
    results/plots/phph1_autocorrelation_by_rho.png
    results/phph1_autocorrelation_summary.csv
And prints a half-life comparison against M/PH/1 using
results/mph1_autocorrelation_summary.csv if it exists.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "phph1"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
SETTINGS = [(2, 2)]     # (arrival Erlang order, service Erlang order)
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
    acf_by_setting_rho = {}

    for ka, ks in SETTINGS:
        for rho in RHOS:
            raw = pd.read_csv(DATA_DIR / f"phph1_timeseries_arrE{ka}_srvE{ks}_rho{rho}.csv")
            raw = raw[raw["run_id"].isin(TRAIN_RUN_IDS)]

            run_acfs = []
            for run_id, g in raw.groupby("run_id"):
                g = g.sort_values("t")
                run_acfs.append(autocorrelation_for_run(g["queue_length"], MAX_LAG))

            mean_acf = np.nanmean(np.array(run_acfs), axis=0)
            acf_by_setting_rho[(ka, ks, rho)] = mean_acf

            below_half = np.where(mean_acf < 0.5)[0]
            half_life = int(below_half[0]) + 1 if len(below_half) > 0 else np.nan

            summary_rows.append({"arrival_erlang_k": ka, "service_erlang_k": ks, "rho": rho,
                                 "acf_lag1": mean_acf[0], "acf_lag5": mean_acf[4],
                                 "acf_lag10": mean_acf[9], "acf_lag20": mean_acf[19],
                                 "half_life_lag": half_life})
            print(f"arrE{ka}/srvE{ks}, rho={rho}: ACF(1)={mean_acf[0]:.3f}, "
                  f"ACF(10)={mean_acf[9]:.3f}, ACF(20)={mean_acf[19]:.3f}, "
                  f"half-life = {half_life}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "phph1_autocorrelation_summary.csv", index=False)
    print("\nSaved: results/phph1_autocorrelation_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    lags = np.arange(1, MAX_LAG + 1)
    for ka, ks in SETTINGS:
        for rho in RHOS:
            ax.plot(lags, acf_by_setting_rho[(ka, ks, rho)], marker="o", markersize=3, label=f"rho={rho}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag (time units)")
    ax.set_ylabel("Autocorrelation of queue_length")
    ax.set_title("PH/PH/1 (Erlang-2/Erlang-2) queue-length persistence vs. traffic intensity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phph1_autocorrelation_by_rho.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/plots/phph1_autocorrelation_by_rho.png")

    print_comparison(summary_df)
    return summary_df


def print_comparison(phph1_df):
    """Half-life of persistence (lags until ACF < 0.5): PH/PH/1 vs. M/PH/1, same service (Erlang-2),
    to isolate the effect of smoothing the ARRIVAL process alone (C_a^2: 1.0 -> 0.5)."""
    tables = {}
    p = RESULTS_DIR / "mph1_autocorrelation_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["erlang_k"] == 2]
        tables["M/PH/1  (C_a^2=1.0, C_s^2=0.5)"] = d.set_index("rho")["half_life_lag"]
    tables["PH/PH/1 (C_a^2=0.5, C_s^2=0.5)"] = phph1_df.set_index("rho")["half_life_lag"]

    comp = pd.DataFrame(tables)
    print("\n" + "=" * 90)
    print("HALF-LIFE OF QUEUE-LENGTH PERSISTENCE (lags until autocorrelation < 0.5)")
    print("Effect of smoothing arrivals alone (same Erlang-2 service in both columns)")
    print("=" * 90)
    print(comp.to_string())


if __name__ == "__main__":
    compute_all()