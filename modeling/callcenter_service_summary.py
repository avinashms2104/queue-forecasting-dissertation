"""
Call-centre case study: effect of the real service-time variability (Azam, 16 Oct)
=================================================================================

Sensitivity check: the same MAPs fitted to the call-centre arrivals, the same mean
service time (1) and the same rho, with service either Erlang-2 (C_s^2 = 0.5, the main
comparison used throughout the project) or a balanced H2 with the C_s^2 observed in the
call-centre data (1.47 afternoon, 2.20 evening; simulation/callcenter_service_fit.py).
So only the variability of service changes.

Reads results already written by the pipeline:
    results/mapph1_{cc_afternoon,cc_evening}{,_realsvc}_validation.csv
    results/mapph1_..._autocorrelation_summary.csv
    results/mapph1_..._model_comparison_{regression,classification}.csv
    results/early_warning_fair_paired_callcenter.csv  (early_warning_fair_comparison.py --callcenter)

Usage:   python modeling/callcenter_service_summary.py
Writes:  results/callcenter_service_summary.csv, results/plots/callcenter_service_summary.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOTS = RES / "plots"
H = "medium"
MAIN_C = 5

COLUMNS = {   # column: (results prefix, name in the early-warning file)
    "afternoon, Erlang-2": ("mapph1_cc_afternoon", "cc afternoon"),
    "afternoon, real service": ("mapph1_cc_afternoon_realsvc", "cc afternoon, real service"),
    "evening, Erlang-2": ("mapph1_cc_evening", "cc evening"),
    "evening, real service": ("mapph1_cc_evening_realsvc", "cc evening, real service"),
}


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    t = {k: {} for k in ["Lq", "half_life", "rf_gain", "rich_minus_history", "f1_gap",
                         "ew_rf_vs_persistence", "ew_rule_vs_persistence", "ew_rf_vs_rule", "ew_rf_runs_better"]}
    pair = read("early_warning_fair_paired_callcenter.csv")
    for col, (pre, ew_name) in COLUMNS.items():
        v = read(f"{pre}_validation.csv")
        a = read(f"{pre}_autocorrelation_summary.csv")
        r = read(f"{pre}_model_comparison_regression.csv")
        c = read(f"{pre}_model_comparison_classification.csv")
        if v is not None:
            t["Lq"][col] = v.set_index("rho")["Lq_qbd"]
        if a is not None:
            t["half_life"][col] = pd.to_numeric(a.set_index("rho")["half_life_lag"], errors="coerce")
        if r is not None:
            sel = (r.horizon == H) & (r.model == "Random Forest") & (r.test_run_id.astype(str) == "ALL_MEAN")
            g_h = r[sel & (r.dataset == "history_only")].set_index("rho")["relative_improvement_over_persistence"] * 100
            g_r = r[sel & (r.dataset == "rich_features")].set_index("rho")["relative_improvement_over_persistence"] * 100
            t["rf_gain"][col] = g_h
            if len(g_r):
                t["rich_minus_history"][col] = g_r - g_h
        if c is not None:
            d = c[(c.dataset == "history_only") & (c.horizon == H) & (c.test_run_id.astype(str) == "ALL_MEAN")]
            t["f1_gap"][col] = (d[d.model == "Random Forest"].set_index("rho")["f1_congested"]
                                - d[d.model == "Naive (persistence)"].set_index("rho")["f1_congested"])
        if pair is not None:
            p = pair[(pair.queue == ew_name) & (pair.C == MAIN_C)]
            get = lambda comp, k: p[p.comparison == comp].set_index("rho")[k]
            t["ew_rf_vs_persistence"][col] = get("RF classifier, rich features vs persistence", "mean_pct")
            t["ew_rule_vs_persistence"][col] = get("Lowered-threshold rule vs persistence", "mean_pct")
            t["ew_rf_vs_rule"][col] = get("RF classifier, rich features vs rule", "mean_pct")
            t["ew_rf_runs_better"][col] = get("RF classifier, rich features vs rule", "runs_better")
    tables = {k: pd.DataFrame(v) for k, v in t.items() if v}

    titles = {
        "Lq": ("Exact mean queue length L_q (QBD)", "{:.2f}"),
        "half_life": ("Half-life of queue-length autocorrelation (time units)", "{:.0f}"),
        "rf_gain": ("Random Forest MAE gain over persistence (%), h = 10, history-only", "{:+.1f}"),
        "rich_minus_history": ("Additional features minus history-only (points of MAE gain), h = 10", "{:+.2f}"),
        "f1_gap": ("Congestion F1: Random Forest minus persistence, h = 10", "{:+.3f}"),
        "ew_rf_vs_persistence": (f"Early warning (C = {MAIN_C}): RF (rich) total cost, % lower than persistence", "{:+.1f}"),
        "ew_rule_vs_persistence": (f"Early warning (C = {MAIN_C}): lowered-threshold rule, % lower than persistence", "{:+.1f}"),
        "ew_rf_vs_rule": (f"Early warning (C = {MAIN_C}): RF (rich) against the rule, % lower total cost", "{:+.1f}"),
        "ew_rf_runs_better": (f"Early warning (C = {MAIN_C}): test runs (of 5) where RF beats the rule", "{:.0f}"),
    }
    pd.set_option("display.width", 200)
    bar = "=" * 100
    print(f"{bar}\nSame fitted call-centre arrivals, mean service 1, same rho; service Erlang-2 (C_s^2 = 0.5) "
          f"or H2 with the observed C_s^2\n{bar}")
    long = []
    for k, (title, fmt) in titles.items():
        if k not in tables:
            continue
        tb = tables[k][[c for c in COLUMNS if c in tables[k].columns]]
        print(f"\n{bar}\n{title}\n{bar}")
        print(tb.map(lambda x: fmt.format(x) if pd.notna(x) else "-").to_string())
        m = tb.reset_index().melt(id_vars="rho", var_name="setting", value_name="value")
        m["quantity"] = k
        long.append(m)
    if long:
        pd.concat(long, ignore_index=True)[["quantity", "setting", "rho", "value"]].to_csv(
            RES / "callcenter_service_summary.csv", index=False)
        print("\nSaved: results/callcenter_service_summary.csv")

    panels = [("Lq", "exact L_q", True), ("rf_gain", "RF MAE gain over persistence (%)", False),
              ("ew_rf_vs_persistence", "early-warning cost cut vs persistence (%)", False),
              ("ew_rf_vs_rule", "RF vs rule (% lower total cost)", False)]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, (k, ylab, logy) in zip(ax.ravel(), panels):
        if k not in tables:
            a.set_visible(False); continue
        for col in COLUMNS:
            if col in tables[k].columns:
                real = "real" in col
                a.plot(tables[k].index, tables[k][col], marker="s" if real else "o", ls="-" if real else "--",
                       color="C0" if col.startswith("afternoon") else "C1", label=col)
        if logy:
            a.set_yscale("log")
        else:
            a.axhline(0, color="grey", lw=0.8)
        a.set_xlabel("rho"); a.set_ylabel(ylab); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Call centre: Erlang-2 service (dashed) against the observed service variability (solid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "callcenter_service_summary.png", dpi=150)
    print("Saved: results/plots/callcenter_service_summary.png")


if __name__ == "__main__":
    main()