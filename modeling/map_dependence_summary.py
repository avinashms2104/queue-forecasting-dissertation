"""
MAP/PH/1: effect of arrival DEPENDENCE alone (Azam, 12 Oct)
==========================================================

Brings together, for the four arrival processes with the SAME gap variability
(C_a^2 = 1.684) and the SAME Erlang-2 service:

    none (H2)   renewal, no dependence between gaps
    weak        MMPP-2, short bursts (5 arrivals per busy spell)
    medium      MMPP-2, 10 arrivals per busy spell (original MAP/PH/1)
    strong      MMPP-2, long bursts (40 arrivals per busy spell)

The three MAPs also have the same arrival rate and the same arrival-count
variability over a window of 10 mean gaps (VtM = 3.49; see
simulation/check_map_levels.py), so only the dependence structure differs.

Reads results already produced by the pipeline (nothing is re-run):
    results/{h2ph1,mapph1_weak,mapph1,mapph1_strong}_validation.csv
    results/{h2ph1,mapph1_weak,mapph1,mapph1_strong}_autocorrelation_summary.csv
    results/{...}_model_comparison_regression.csv / _classification.csv
    results/horizon_sweep_rich_features_map_levels.csv

Usage:  python modeling/map_dependence_summary.py
Writes  results/map_dependence_summary.csv and results/plots/map_dependence_summary.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOTS = RES / "plots"
sys.path.insert(0, str(ROOT / "simulation"))

LEVELS = {                     # label: (file prefix, name in the horizon-sweep file)
    "none (H2)": ("h2ph1", "H2 (no bursts)"),
    "weak": ("mapph1_weak", "MAP weak"),
    "medium": ("mapph1", "MAP medium"),
    "strong": ("mapph1_strong", "MAP strong"),
}
H = "medium"                   # forecast horizon used in the summary tables (h = 10)


def arrival_properties():
    """Exact count variability (window of 10 mean gaps) and long-run IDC, from map_utils."""
    from ph_utils import hyperexp2_balanced
    from map_utils import map_level, ph_renewal_map, MATCH_WINDOW
    ref = map_level("medium", 1.0)
    procs = {"none (H2)": ph_renewal_map(hyperexp2_balanced(1.0, ref.scv))}
    for lv in ("weak", "medium", "strong"):
        procs[lv] = map_level(lv, 1.0)
    return {lv: (M.scv, M.vtm(MATCH_WINDOW / M.lam), M.idc_limit()) for lv, M in procs.items()}, MATCH_WINDOW


def read(name):
    p = RES / name
    if not p.exists():
        print(f"  (missing {name}, column left empty)")
        return None
    return pd.read_csv(p)


def per_rho(df, value_col, out_name):
    if df is None:
        return None
    s = df.set_index("rho")[value_col]
    s = pd.to_numeric(s, errors="coerce")
    s.name = out_name
    return s


def reg_gain(df, dataset):
    """Random Forest relative MAE improvement over persistence (%), mean over the 5 test runs."""
    d = df[(df.dataset == dataset) & (df.horizon == H) & (df.model == "Random Forest")
           & (df.test_run_id.astype(str) == "ALL_MEAN")]
    return d.set_index("rho")["relative_improvement_over_persistence"] * 100


def f1_gap(df):
    d = df[(df.dataset == "history_only") & (df.horizon == H) & (df.test_run_id.astype(str) == "ALL_MEAN")]
    rf = d[d.model == "Random Forest"].set_index("rho")["f1_congested"]
    pe = d[d.model == "Naive (persistence)"].set_index("rho")["f1_congested"]
    return rf - pe


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    props, window = arrival_properties()

    cols = {k: {} for k in ["Lq", "half_life", "rf_gain", "f1_gap", "rich_minus_history"]}
    for lv, (pre, _) in LEVELS.items():
        v = read(f"{pre}_validation.csv")
        a = read(f"{pre}_autocorrelation_summary.csv")
        r = read(f"{pre}_model_comparison_regression.csv")
        c = read(f"{pre}_model_comparison_classification.csv")
        cols["Lq"][lv] = per_rho(v, "Lq_qbd", lv)
        cols["half_life"][lv] = per_rho(a, "half_life_lag", lv)
        if r is not None:
            g_h = reg_gain(r, "history_only")
            g_r = reg_gain(r, "rich_features")
            cols["rf_gain"][lv] = g_h
            cols["rich_minus_history"][lv] = (g_r - g_h) if len(g_r) else np.nan * g_h
        if c is not None:
            cols["f1_gap"][lv] = f1_gap(c)

    tables = {k: pd.DataFrame({lv: s for lv, s in d.items() if s is not None}) for k, d in cols.items()}
    for t in tables.values():
        t.index.name = "rho"
        t.columns.name = "dependence"

    bar = "=" * 100
    print(bar)
    print("Same C_a^2 = 1.684 and Erlang-2 service; only the dependence between arrival gaps changes")
    print(f"Count variability (VtM) over {window:g} mean gaps:  "
          + "  ".join(f"{lv} {p[1]:.2f}" for lv, p in props.items()))
    print("Long-run index of dispersion:            "
          + "  ".join(f"{lv} {p[2]:.2f}" for lv, p in props.items()))
    print("(the three MAPs share the same VtM over 10 gaps; H2 has no dependence, so its VtM stays near C_a^2)")

    titles = {
        "Lq": "Exact mean queue length L_q (QBD)",
        "half_life": "Half-life of queue-length autocorrelation (time units)",
        "rf_gain": "Random Forest MAE gain over persistence (%), h = 10, history-only",
        "f1_gap": "Congestion F1: Random Forest minus persistence, h = 10",
        "rich_minus_history": "Additional features minus history-only: change in MAE gain (points), h = 10",
    }
    fmt = {"Lq": "{:.2f}", "half_life": "{:.1f}", "rf_gain": "{:.1f}", "f1_gap": "{:.3f}",
           "rich_minus_history": "{:.2f}"}
    for k, title in titles.items():
        print(f"\n{bar}\n{title}\n{bar}")
        print(tables[k].map(lambda x: fmt[k].format(x) if pd.notna(x) else "NaN").to_string())

    sweep = read("horizon_sweep_rich_features_map_levels.csv")
    if sweep is not None:
        print(f"\n{bar}\nHorizon sweep at rho = 0.8\n{bar}")
        for lv, (_, qname) in LEVELS.items():
            s = sweep[sweep.queue == qname]
            hist = s[s.model == "RF history-only"].set_index("h")["MAE_impr_pct"]
            rich = s[s.model == "RF rich features"].set_index("h")["MAE_impr_pct"]
            if len(hist) == 0:
                continue
            print(f"{lv:10s}: RF MAE gain at h = {hist.index.max()} = {hist.iloc[-1]:6.1f}%   |   "
                  f"largest gain from additional features over h = {hist.index.min()}..{hist.index.max()} = "
                  f"{(rich - hist).max():+.2f} points")

    long = []
    for k, t in tables.items():
        m = t.reset_index().melt(id_vars="rho", var_name="dependence", value_name="value")
        m.insert(0, "quantity", k)
        long.append(m)
    for lv, p in props.items():
        long.append(pd.DataFrame([dict(quantity="gap_scv", rho=np.nan, dependence=lv, value=p[0]),
                                  dict(quantity=f"vtm_{window:g}_gaps", rho=np.nan, dependence=lv, value=p[1]),
                                  dict(quantity="idc_limit", rho=np.nan, dependence=lv, value=p[2])]))
    pd.concat(long, ignore_index=True).to_csv(RES / "map_dependence_summary.csv", index=False)
    print("\nSaved: results/map_dependence_summary.csv")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = [("Lq", "exact L_q", True), ("half_life", "half-life (time units)", True),
              ("rf_gain", "RF MAE gain over persistence (%)", False),
              ("rich_minus_history", "gain from additional features (points)", False)]
    for a, (k, ylab, logy) in zip(ax.ravel(), panels):
        for lv in tables[k].columns:
            a.plot(tables[k].index, tables[k][lv], marker="o", label=lv)
        if logy:
            a.set_yscale("log")
        else:
            a.axhline(0, color="grey", lw=0.8)
        a.set_xlabel("rho"); a.set_ylabel(ylab); a.set_title(titles[k], fontsize=9)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Same rate, gap variability and 10-gap count variability; only dependence changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "map_dependence_summary.png", dpi=150)
    print("Saved: results/plots/map_dependence_summary.png")


if __name__ == "__main__":
    main()