"""
Consolidated comparison of M/M/1, M/PH/1 and PH/PH/1 (Azam, email of 5 October
2026, priority 1): the same utilisations, sampling interval and forecast
horizons for all three queues, so the results are directly comparable.

All three queues use: rho in {0.3, 0.5, 0.7, 0.8, 0.9, 0.95}; dt = 1; horizons
h = 5, 10, 15, 20; 30 runs (20 train / 5 validation / 5 test); the same models;
results averaged over the 5 test runs. M/PH/1 and PH/PH/1 share the same
Erlang-2 service, so M/PH/1 vs PH/PH/1 isolates the arrival process
(Poisson vs Erlang-2), and M/M/1 vs M/PH/1 isolates the service distribution.

MAE is the primary metric (Azam's choice); RMSE is reported alongside.
Reads the existing results CSVs only; trains nothing; runs in seconds.

Tables (printed, and written as LaTeX for Overleaf):
  1. MAE at the medium horizon (h = 10): persistence, Random Forest with
     history-only features, Random Forest with rich features
  2. Random Forest (history-only) MAE improvement over persistence, by rho and h
  3. The same for RMSE
  4. Rich minus history-only features: change in MAE improvement (percentage points)
  5. Congestion F1: Random Forest minus persistence, by rho and h

Usage:  python modeling/consolidate_comparison.py
Writes: results/consolidated_comparison.csv
        results/consolidated_tables.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
H_OF = {"short": 5, "medium": 10, "extra_long": 15, "long": 20}
HS = [5, 10, 15, 20]

QUEUES = [
    ("M/M/1", "model_comparison", lambda d: d),
    ("M/PH/1", "mph1_model_comparison", lambda d: d[d.erlang_k == 2]),
    ("PH/PH/1", "phph1_model_comparison",
     lambda d: d[(d.arrival_erlang_k == 2) & (d.service_erlang_k == 2)]),
]


def load():
    reg_parts, clf_parts = [], []
    for q, stem, flt in QUEUES:
        r = flt(pd.read_csv(RESULTS / f"{stem}_regression.csv"))
        r = r[r.test_run_id.astype(str) != "ALL_MEAN"].assign(queue=q)
        reg_parts.append(r[["queue", "dataset", "rho", "horizon", "model", "test_run_id", "MAE", "RMSE"]])
        c = flt(pd.read_csv(RESULTS / f"{stem}_classification.csv"))
        c = c[c.test_run_id.astype(str) != "ALL_MEAN"].assign(queue=q)
        clf_parts.append(c[["queue", "dataset", "rho", "horizon", "model", "test_run_id",
                            "precision_congested", "recall_congested", "f1_congested"]])
    reg = pd.concat(reg_parts, ignore_index=True)
    clf = pd.concat(clf_parts, ignore_index=True)
    reg["h"] = reg.horizon.map(H_OF)
    clf["h"] = clf.horizon.map(H_OF)
    return reg, clf


def summarise(reg, clf):
    """One row per queue x rho x h with every number the tables need (means over the 5 test runs)."""
    g = reg.groupby(["queue", "rho", "h", "dataset", "model"])[["MAE", "RMSE"]].mean().reset_index()

    def pick(dataset, model, col):
        s = g[(g.dataset == dataset) & (g.model == model)].set_index(["queue", "rho", "h"])[col]
        return s

    out = pd.DataFrame({
        "MAE_persistence": pick("history_only", "Naive (persistence)", "MAE"),
        "MAE_RF_history": pick("history_only", "Random Forest", "MAE"),
        "MAE_RF_rich": pick("rich_features", "Random Forest", "MAE"),
        "RMSE_persistence": pick("history_only", "Naive (persistence)", "RMSE"),
        "RMSE_RF_history": pick("history_only", "Random Forest", "RMSE"),
        "RMSE_RF_rich": pick("rich_features", "Random Forest", "RMSE"),
    })
    out["MAE_impr_history_pct"] = 100 * (1 - out.MAE_RF_history / out.MAE_persistence)
    out["MAE_impr_rich_pct"] = 100 * (1 - out.MAE_RF_rich / out.MAE_persistence)
    out["RMSE_impr_history_pct"] = 100 * (1 - out.RMSE_RF_history / out.RMSE_persistence)
    out["rich_minus_history_pp"] = out.MAE_impr_rich_pct - out.MAE_impr_history_pct

    c = clf[clf.dataset == "history_only"].groupby(["queue", "rho", "h", "model"])[
        ["precision_congested", "recall_congested", "f1_congested"]].mean().reset_index()
    for model, tag in [("Naive (persistence)", "persistence"), ("Random Forest", "RF")]:
        s = c[c.model == model].set_index(["queue", "rho", "h"])
        out[f"F1_{tag}"] = s["f1_congested"]
        out[f"precision_{tag}"] = s["precision_congested"]
        out[f"recall_{tag}"] = s["recall_congested"]
    out["F1_RF_minus_persistence"] = out.F1_RF - out.F1_persistence
    return out.reset_index()


# ---------------- LaTeX helpers (plain tabular, booktabs) ----------------
def latex_rho_by_h(summ, col, caption, label, fmt="{:+.1f}"):
    """rho rows; for each queue, one column per horizon."""
    qs = [q for q, _, _ in QUEUES]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + ("rrrr" * len(qs)) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{4}{c}{" + q + "}" for q in qs) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(qs))),
             r"$\rho$ & " + " & ".join(" & ".join(f"$h={h}$" for h in HS) for _ in qs) + r" \\",
             r"\midrule"]
    for rho in RHOS:
        cells = []
        for q in qs:
            for h in HS:
                v = summ[(summ.queue == q) & (summ.rho == rho) & (summ.h == h)][col]
                cells.append(fmt.format(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else "--")
        lines.append(f"{rho} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_medium_mae(summ):
    qs = [q for q, _, _ in QUEUES]
    med = summ[summ.h == 10]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Mean absolute error at the medium horizon ($h=10$), mean of 5 test runs. "
             r"P = persistence, H = Random Forest with queue-history-only features, "
             r"R = Random Forest with additional features.}",
             r"\label{tab:consolidated_mae_medium}",
             r"\begin{tabular}{l" + ("rrr" * len(qs)) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{" + q + "}" for q in qs) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(qs))),
             r"$\rho$ & " + " & ".join("P & H & R" for _ in qs) + r" \\", r"\midrule"]
    for rho in RHOS:
        cells = []
        for q in qs:
            r = med[(med.queue == q) & (med.rho == rho)].iloc[0]
            cells += [f"{r.MAE_persistence:.3f}", f"{r.MAE_RF_history:.3f}", f"{r.MAE_RF_rich:.3f}"]
        lines.append(f"{rho} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    reg, clf = load()
    summ = summarise(reg, clf)
    summ.to_csv(RESULTS / "consolidated_comparison.csv", index=False)
    qs = [q for q, _, _ in QUEUES]
    pd.set_option("display.width", 220)

    def show(col, title, digits=1):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        t = summ.pivot_table(index="rho", columns=["queue", "h"], values=col)
        print(t[qs].round(digits).to_string())

    print("=" * 110)
    print("1. MAE at the medium horizon (h = 10): persistence | RF history-only | RF rich features")
    print("=" * 110)
    med = summ[summ.h == 10].set_index(["rho", "queue"])[["MAE_persistence", "MAE_RF_history", "MAE_RF_rich"]]
    print(med.unstack("queue").swaplevel(axis=1).sort_index(axis=1)[qs].round(3).to_string())

    show("MAE_impr_history_pct", "2. Random Forest (history-only) MAE improvement over persistence (%), by rho and h")
    show("RMSE_impr_history_pct", "3. Random Forest (history-only) RMSE improvement over persistence (%), by rho and h")
    show("rich_minus_history_pp",
         "4. Rich minus history-only features: change in MAE improvement (percentage points)", 2)
    show("F1_RF_minus_persistence",
         "5. Congestion F1: Random Forest minus persistence, by rho and h (negative = persistence better)", 3)

    print("\n" + "=" * 110)
    print("Summary over all 72 settings (3 queues x 6 rho x 4 horizons)")
    print("=" * 110)
    n = len(summ)
    print(f"RF beats persistence on MAE in {int((summ.MAE_impr_history_pct > 0).sum())} of {n}; "
          f"on RMSE in {int((summ.RMSE_impr_history_pct > 0).sum())} of {n}")
    print(f"Rich features change the MAE improvement by at most "
          f"{summ.rich_minus_history_pp.abs().max():.2f} percentage points "
          f"(mean {summ.rich_minus_history_pp.mean():+.2f})")
    print(f"RF F1 >= persistence F1 in {int((summ.F1_RF_minus_persistence >= 0).sum())} of {n}; "
          f"largest RF advantage {summ.F1_RF_minus_persistence.max():+.3f}")

    tex = [
        "% Generated by modeling/consolidate_comparison.py -- needs \\usepackage{booktabs}",
        latex_medium_mae(summ),
        latex_rho_by_h(summ, "MAE_impr_history_pct",
                       r"Random Forest (queue-history-only) MAE improvement over persistence (\%), "
                       r"by utilisation and horizon. Positive = Random Forest better.",
                       "tab:consolidated_mae_impr"),
        latex_rho_by_h(summ, "RMSE_impr_history_pct",
                       r"Random Forest (queue-history-only) RMSE improvement over persistence (\%), "
                       r"by utilisation and horizon.",
                       "tab:consolidated_rmse_impr"),
        latex_rho_by_h(summ, "rich_minus_history_pp",
                       r"Additional features minus queue-history-only: change in the Random Forest "
                       r"MAE improvement over persistence (percentage points).",
                       "tab:consolidated_rich", fmt="{:+.2f}"),
        latex_rho_by_h(summ, "F1_RF_minus_persistence",
                       r"Congestion F1 of Random Forest minus persistence, by utilisation and horizon "
                       r"(negative = persistence better).",
                       "tab:consolidated_f1", fmt="{:+.3f}"),
    ]
    (RESULTS / "consolidated_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\nSaved: results/consolidated_comparison.csv")
    print("Saved: results/consolidated_tables.tex")


if __name__ == "__main__":
    main()