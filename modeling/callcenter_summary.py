"""
Call-centre case study: do the conclusions from the controlled experiments carry over? (Azam, 12 Oct)
===================================================================================================

Azam: "We can use the fitted model as an additional case study to assess whether
the conclusions from the simulated experiments carry over to a more realistic setting."

This script puts the two MAPs fitted to the real call-centre arrivals (stable
periods 11:00-15:30 and 19:30-22:30, see simulation/callcenter_fit_map.py)
side by side with the controlled arrival processes, all with the same Erlang-2
service:
    none (H2)    renewal, no dependence, gap C_a^2 = 1.684
    weak / medium / strong   controlled MMPP-2s (same rate, C_a^2 and 10-gap VtM)
    cc afternoon / cc evening   MMPP-2s fitted to the real arrivals

Nothing is re-run; it reads the results the pipeline has already written:
    results/{h2ph1,mapph1{_weak,,_strong,_cc_afternoon,_cc_evening}}_validation.csv
    results/..._autocorrelation_summary.csv, ..._model_comparison_{regression,classification}.csv
    results/early_warning_{cost,onsets}.csv and results/early_warning_{cost,onsets}_callcenter.csv

Usage:   python modeling/callcenter_summary.py
Writes:  results/callcenter_summary.csv and results/plots/callcenter_summary.png
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

#  column: (results prefix, map_utils level or None, name in early-warning files)
COLUMNS = {
    "none (H2)": ("h2ph1", None, "none (H2)"),
    "weak": ("mapph1_weak", "weak", "weak"),
    "medium": ("mapph1", "medium", "medium"),
    "strong": ("mapph1_strong", "strong", "strong"),
    "cc afternoon": ("mapph1_cc_afternoon", "cc_afternoon", "call centre afternoon"),
    "cc evening": ("mapph1_cc_evening", "cc_evening", "call centre evening"),
}
H = "medium"          # h = 10
MAIN_C = 5            # cost ratio used for the early-warning rows


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def arrival_properties():
    from ph_utils import hyperexp2_balanced
    from map_utils import arrival_map, map_level, ph_renewal_map, MATCH_WINDOW
    out = {}
    for col, (_, level, _) in COLUMNS.items():
        try:
            if level is None:
                M = ph_renewal_map(hyperexp2_balanced(1.0, map_level("medium", 1.0).scv))
            else:
                M = arrival_map(level, 1.0)
            out[col] = dict(gap_scv=M.scv, gap_corr_1=M.lag_corr(1),
                            vtm_10=M.vtm(MATCH_WINDOW), idc_limit=M.idc_limit())
        except Exception as e:                     # e.g. the call-centre fit file is missing
            print(f"  (arrival properties for {col} not available: {e})")
    return pd.DataFrame(out)


def ml_tables():
    t = {k: {} for k in ["Lq", "half_life", "rf_gain", "rich_minus_history", "f1_gap"]}
    for col, (pre, _, _) in COLUMNS.items():
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
    return {k: pd.DataFrame(v) for k, v in t.items()}


def early_warning_tables():
    ons = [x for x in (read("early_warning_onsets.csv"), read("early_warning_onsets_callcenter.csv")) if x is not None]
    cost = [x for x in (read("early_warning_cost.csv"), read("early_warning_cost_callcenter.csv")) if x is not None]
    out = {}
    if ons:
        o = pd.concat(ons, ignore_index=True)
        o = o[o.C == MAIN_C]
        ml = o[o.method.str.startswith("RF classifier")]
        best = ml.loc[ml.groupby(["queue", "rho"])["cost"].idxmin()].set_index(["queue", "rho"])
        pers = o[o.method == "Persistence"].set_index(["queue", "rho"])["cost"]
        rule = o[o.method == "Lowered-threshold rule"].set_index(["queue", "rho"])["cost"]
        e = pd.DataFrame({"warned_5_ahead": best["early_warning_rate"],
                          "cost_cut_vs_persistence": 100 * (1 - best["cost"] / pers),
                          "rule_cost_cut_vs_persistence": 100 * (1 - rule / pers)})
        names = {v[2]: k for k, v in COLUMNS.items()}
        e = e.reset_index()
        e["queue"] = e["queue"].map(names)
        for k in ["warned_5_ahead", "cost_cut_vs_persistence", "rule_cost_cut_vs_persistence"]:
            out[k] = e.pivot(index="rho", columns="queue", values=k)
    if cost:
        c = pd.concat(cost, ignore_index=True)
        c = c[c.C == MAIN_C].set_index(["queue", "rho", "method"])["cost"].unstack("method")
        q = 100 * (1 - c["Quantile GB, rich features"] / c["Persistence + margin"])
        q = q.reset_index()
        names = {v[2]: k for k, v in COLUMNS.items()}
        q["queue"] = q["queue"].map(names)
        out["quantile_vs_margin"] = q.pivot(index="rho", columns="queue", values=0)
    return out


def show(title, t, fmt):
    cols = [c for c in COLUMNS if c in t.columns]
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")
    print(t[cols].map(lambda x: fmt.format(x) if pd.notna(x) else "-").to_string())


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    props = arrival_properties()
    show("Arrival process (exact, at rate 1)", props, "{:.3f}")
    print("(vtm_10 = variance-to-mean ratio of the number of arrivals in 10 mean gaps; 1 = Poisson)")

    ml = ml_tables()
    show("Exact mean queue length L_q (QBD)", ml["Lq"], "{:.2f}")
    show("Half-life of queue-length autocorrelation (time units)", ml["half_life"], "{:.0f}")
    show("Random Forest MAE gain over persistence (%), h = 10, history-only", ml["rf_gain"], "{:+.1f}")
    show("Additional features minus history-only (points of MAE gain), h = 10", ml["rich_minus_history"], "{:+.2f}")
    show("Congestion F1: Random Forest minus persistence, h = 10", ml["f1_gap"], "{:+.3f}")

    ew = early_warning_tables()
    titles = {
        "warned_5_ahead": f"Early warning (C = {MAIN_C}): share of congestion onsets warned >= 5 steps ahead, best RF classifier (persistence = 0)",
        "cost_cut_vs_persistence": f"Early warning (C = {MAIN_C}): per-step cost, % lower than persistence, best RF classifier",
        "rule_cost_cut_vs_persistence": f"Early warning (C = {MAIN_C}): per-step cost, % lower than persistence, lowered-threshold rule",
        "quantile_vs_margin": f"Under-prediction cost (C = {MAIN_C}): quantile GB with all features, % lower than persistence + margin",
    }
    fmts = {"warned_5_ahead": "{:.2f}", "cost_cut_vs_persistence": "{:+.1f}",
            "rule_cost_cut_vs_persistence": "{:+.1f}", "quantile_vs_margin": "{:+.1f}"}
    for k, title in titles.items():
        if k in ew:
            show(title, ew[k], fmts[k])

    long = [props.T.reset_index().melt(id_vars="index", var_name="quantity").rename(columns={"index": "arrival"})
            .assign(rho=np.nan)]
    for k, t in {**ml, **ew}.items():
        m = t.reset_index().melt(id_vars="rho", var_name="arrival", value_name="value")
        m["quantity"] = k
        long.append(m)
    pd.concat(long, ignore_index=True)[["quantity", "arrival", "rho", "value"]].to_csv(
        RES / "callcenter_summary.csv", index=False)
    print("\nSaved: results/callcenter_summary.csv")

    # ---------- plot ----------
    panels = [("rf_gain", ml, "RF MAE gain over persistence (%)"),
              ("rich_minus_history", ml, "gain from additional features (points)"),
              ("warned_5_ahead", ew, "onsets warned >= 5 steps ahead"),
              ("cost_cut_vs_persistence", ew, "early-warning cost cut vs persistence (%)")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for a, (k, src, ylab) in zip(ax.ravel(), panels):
        if k not in src:
            a.set_visible(False); continue
        t = src[k]
        for col in COLUMNS:
            if col in t.columns:
                cc = col.startswith("cc")
                a.plot(t.index, t[col], marker="s" if cc else "o", lw=2.4 if cc else 1.2,
                       ls="-" if cc else "--", label=col)
        a.axhline(0, color="grey", lw=0.8)
        a.set_xlabel("rho"); a.set_ylabel(ylab); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("Real call-centre arrivals (solid) against the controlled arrival processes (dashed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "callcenter_summary.png", dpi=150)
    print("Saved: results/plots/callcenter_summary.png")


if __name__ == "__main__":
    main()