"""
Autocorrelation analysis for MAP/PH/1 (bursty MMPP-2 arrivals, Erlang-2
service): checks whether the rising-persistence-with-rho pattern still holds
once arrivals are bursty and correlated, and compares directly against
M/PH/1 (Poisson arrivals) and PH/PH/1 (Erlang-2 arrivals), which have the
same Erlang-2 service. The three columns therefore differ ONLY in the
arrival process:

    PH/PH/1   Erlang-2 renewal   C_a^2 = 0.5, no correlation   (smooth)
    M/PH/1    Poisson            C_a^2 = 1.0, no correlation   (random)
    MAP/PH/1  MMPP-2             C_a^2 = 1.68, lag-1 corr 0.165,
                                 index of dispersion 4.6       (bursty)

Bursty queues stay correlated for much longer than 30 lags, so the
autocorrelation is computed up to MAX_LAG = 500 (the earlier scripts used
30, so their half-life is blank at rho >= 0.9). The summary columns (lags
1, 5, 10, 20) and the half-life definition (first lag where ACF < 0.5) are
unchanged, so the numbers are comparable.

Usage
-----
    python modeling/mapph1_autocorrelation_analysis.py --level medium
    (levels: medium, weak, strong, cc_afternoon, cc_evening; see mapph1_simulator.py)

Produces (SFX = "" for medium, "_weak", "_strong", "_cc_afternoon", "_cc_evening"):
    results/plots/mapph1{SFX}_autocorrelation_by_rho.png
    results/mapph1{SFX}_autocorrelation_summary.csv
And prints a half-life comparison against M/PH/1 and PH/PH/1 using
results/mph1_autocorrelation_summary.csv and
results/phph1_autocorrelation_summary.csv if they exist.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulation"))
from map_utils import LEVEL_INFO

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mapph1"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
MAX_LAG = 500
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


def compute_all(level="medium"):
    label, sfx, _ = LEVEL_INFO[level]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    acf_by_rho = {}

    for rho in RHOS:
        raw = pd.read_csv(DATA_DIR / f"mapph1_timeseries{sfx}_rho{rho}.csv")
        raw = raw[raw["run_id"].isin(TRAIN_RUN_IDS)]

        run_acfs = []
        for run_id, g in raw.groupby("run_id"):
            g = g.sort_values("t")
            run_acfs.append(autocorrelation_for_run(g["queue_length"], MAX_LAG))

        mean_acf = np.nanmean(np.array(run_acfs), axis=0)
        acf_by_rho[rho] = mean_acf

        below_half = np.where(mean_acf < 0.5)[0]
        half_life = int(below_half[0]) + 1 if len(below_half) > 0 else np.nan

        summary_rows.append({"arrival_process": label, "service_erlang_k": 2, "rho": rho,
                             "acf_lag1": mean_acf[0], "acf_lag5": mean_acf[4],
                             "acf_lag10": mean_acf[9], "acf_lag20": mean_acf[19],
                             "acf_lag50": mean_acf[49], "half_life_lag": half_life})
        print(f"{label}/E2, rho={rho}: ACF(1)={mean_acf[0]:.3f}, "
              f"ACF(10)={mean_acf[9]:.3f}, ACF(20)={mean_acf[19]:.3f}, "
              f"ACF(50)={mean_acf[49]:.3f}, half-life = "
              f"{half_life if not np.isnan(half_life) else '> ' + str(MAX_LAG)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / f"mapph1{sfx}_autocorrelation_summary.csv", index=False)
    print(f"\nSaved: results/mapph1{sfx}_autocorrelation_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    lags = np.arange(1, MAX_LAG + 1)
    for rho in RHOS:
        ax.plot(lags, acf_by_rho[rho], marker="o", markersize=2, label=f"rho={rho}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lag (time units)")
    ax.set_ylabel("Autocorrelation of queue_length")
    ax.set_title(f"MAP/PH/1 ({label} / Erlang-2) queue-length persistence vs. traffic intensity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"mapph1{sfx}_autocorrelation_by_rho.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: results/plots/mapph1{sfx}_autocorrelation_by_rho.png")

    print_comparison()
    return summary_df


def print_comparison():
    """Half-life (lags until ACF < 0.5) for every arrival process with the same Erlang-2 service."""
    tables = {}
    p = RESULTS_DIR / "phph1_autocorrelation_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["arrival_erlang_k"] == 2) & (d["service_erlang_k"] == 2)]
        tables["PH/PH/1 smooth"] = d.set_index("rho")["half_life_lag"]
    p = RESULTS_DIR / "mph1_autocorrelation_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["erlang_k"] == 2]
        tables["M/PH/1 random"] = d.set_index("rho")["half_life_lag"]
    p = RESULTS_DIR / "h2ph1_autocorrelation_summary.csv"
    if p.exists():
        tables["H2 (no bursts)"] = pd.read_csv(p).set_index("rho")["half_life_lag"]
    names = {"weak": "MAP weak", "medium": "MAP medium", "strong": "MAP strong",
             "cc_afternoon": "call centre afternoon", "cc_evening": "call centre evening"}
    for lv, col in names.items():
        p = RESULTS_DIR / f"mapph1{LEVEL_INFO[lv][1]}_autocorrelation_summary.csv"
        if p.exists():
            tables[col] = pd.to_numeric(pd.read_csv(p).set_index("rho")["half_life_lag"], errors="coerce")

    comp = pd.DataFrame(tables)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("HALF-LIFE OF QUEUE-LENGTH PERSISTENCE (lags until autocorrelation < 0.5)")
    print("Effect of the arrival process alone (same Erlang-2 service in every column)")
    print("=" * 90)
    print(comp.to_string())
    print(f"(NaN = autocorrelation still above 0.5 at the last lag computed: 30 for the earlier "
          f"queues, {MAX_LAG} for MAP/PH/1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="medium", choices=list(LEVEL_INFO))
    compute_all(ap.parse_args().level)