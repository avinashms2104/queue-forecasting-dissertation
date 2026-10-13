"""
Early warning and the cost of under-prediction (Azam, 12 Oct)
=============================================================

Azam: "The main objective remains to understand how arrival dependence and
burstiness affect congestion prediction, particularly whether ML models can
identify congestion early enough to reduce the cost of under-prediction."

So far every model was scored with symmetric measures (MAE, RMSE, F1), where
predicting 3 customers too few costs the same as predicting 3 too many. In
practice being caught out by congestion (too few staff, no warning) is worse
than a false alarm. This script asks two questions, for the four arrival
processes with the same gap variability (none = H2, weak, medium, strong MAP),
at rho = 0.5, 0.7, 0.8, 0.9 and horizon h = 10:

PART A - forecasts of the queue length, when under-prediction costs more
    cost per step = C * (shortfall) + 1 * (excess),   C = 2, 5, 10
    i.e. predicting one customer too few costs C times as much as one too many.
    For this cost the best forecast is not the mean but the C/(C+1) quantile
    (C = 5 -> the 83rd percentile), so a model trained for the mean is
    deliberately the wrong tool. Methods:
      Persistence                  Q(t+h) = Q(t)
      Persistence + margin         Q(t) + m, m = C/(C+1) quantile of Q(t+h) - Q(t)
                                   on the training runs (a fair, simple safety margin)
      RF (mean)                    the usual Random Forest (history-only), for reference
      Quantile GB, history-only    gradient boosting trained for the C/(C+1) quantile
      Quantile GB, rich features   the same with the additional arrival/departure features

PART B - early warning of congestion ONSETS
    The queue is congested at time t if Q(t) > threshold (top 20%, as before).
    An onset is a time s where the queue becomes congested after at least h
    quiet steps. A method raises an ALARM at time t when it predicts congestion
    at t + h. The LEAD TIME of an onset is how many steps the alarm had been on
    without a break just before it (0 to h), while the queue was still below
    the threshold; an alarm that comes and goes does not count. Reported: the
    share of onsets with lead >= 5 (a useful warning, at least half the horizon
    ahead), the share with lead >= 1 (any warning) and the median lead.
      Persistence                  alarm when congested now: by construction it can
                                   never warn before an onset
      Lowered-threshold rule       alarm when Q(t) > a * threshold (a simple rule
                                   any operator could use; a tuned on validation runs)
      RF classifier, history-only  alarm when P(congested at t+h) >= cut
      RF classifier, rich features
    The cut (and a) is chosen on the VALIDATION runs to minimise the per-step
    cost C * (missed congested step) + 1 * (false alarm), so every method is
    tuned for the same cost. Test runs are used only for the final numbers.

Usage:   python modeling/early_warning_analysis.py                 (H2 and the three controlled MAPs)
         python modeling/early_warning_analysis.py --callcenter    (the two MAPs fitted to the call centre)
Writes:  results/early_warning_cost{_callcenter}.csv       (part A, per method, mean/std over test runs)
         results/early_warning_onsets{_callcenter}.csv     (part B)
         results/plots/early_warning_summary{_callcenter}.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
SEED = 42

QUEUES = {
    "none (H2)": lambda r: DATA / "h2ph1" / "processed" / f"rich_features_rho{r}.csv",
    "weak": lambda r: DATA / "mapph1" / "processed" / f"rich_features_weak_rho{r}.csv",
    "medium": lambda r: DATA / "mapph1" / "processed" / f"rich_features_rho{r}.csv",
    "strong": lambda r: DATA / "mapph1" / "processed" / f"rich_features_strong_rho{r}.csv",
}
CALLCENTER_QUEUES = {
    "call centre afternoon": lambda r: DATA / "mapph1" / "processed" / f"rich_features_cc_afternoon_rho{r}.csv",
    "call centre evening": lambda r: DATA / "mapph1" / "processed" / f"rich_features_cc_evening_rho{r}.csv",
}
OUT_SFX = ""                                    # "_callcenter" with --callcenter
RHOS = [0.5, 0.7, 0.8, 0.9]
HNAME, H = "medium", 10
COSTS = [2, 5, 10]
USEFUL = 5                                      # lead >= 5 steps counts as a useful warning
MAIN_C = 5                                      # cost ratio used in the printed onset table
VAL_RUNS, TEST_RUNS = range(20, 25), range(25, 30)
CUT_GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)
A_GRID = np.round(np.arange(0.30, 1.01, 0.05), 2)


# ---------------------------------------------------------------- helpers
def load(queue, rho):
    df = pd.read_csv(QUEUES[queue](rho))
    hist = [c for c in df.columns if c.startswith("queue_length_lag")]
    target_cols = {c for c in df.columns if c.startswith("target_")}
    base = {"arrival_process", "service_erlang_k", "rho", "run_id", "t", "congestion_threshold", "split"}
    rich = [c for c in df.columns if c not in target_cols and c not in base]
    return df, hist, rich


def asym_cost(y, yhat, C):
    e = y - yhat
    return np.where(e > 0, C * e, -e).mean()


def class_cost(y, alarm, C):
    """Per-step cost: C for each congested step that was not flagged, 1 for each false alarm."""
    return (C * ((y == 1) & (alarm == 0)) + ((y == 0) & (alarm == 1))).mean()


def onsets_and_warnings(run_df, alarm, thr):
    """Lead time (0..h steps of unbroken alarm just before) of every congestion onset in one run."""
    t = run_df["t"].to_numpy().astype(int)
    q = run_df["queue_length_lag0"].to_numpy()
    t0 = t.min()
    n = t.max() - t0 + 1
    cong = np.full(n, np.nan)
    al = np.zeros(n, dtype=bool)
    cong[t - t0] = (q > thr).astype(float)
    al[t - t0] = alarm.astype(bool)
    warned, lead = [], []
    for s in range(H, n):
        if cong[s] != 1:
            continue
        before = cong[s - H:s]
        if np.isnan(before).any() or before.any():       # need h observed quiet steps before s
            continue
        L = 0                                            # alarm on continuously for the last L steps
        while L < H and al[s - 1 - L]:
            L += 1
        lead.append(L)
    lead = np.array(lead)
    return lead >= 1, lead


def choose(score_fn, grid):
    """Grid value with the lowest validation cost."""
    costs = [score_fn(g) for g in grid]
    return grid[int(np.argmin(costs))]


# ---------------------------------------------------------------- main loop
def run_setting(queue, rho):
    df, hist, rich = load(queue, rho)
    thr = float(df["congestion_threshold"].iloc[0])
    ycol, ccol = f"target_qlen_{HNAME}", f"target_congested_{HNAME}"
    tr = df[df.split == "train"]
    va = df[df.run_id.isin(VAL_RUNS)]
    te = df[df.run_id.isin(TEST_RUNS)]

    cost_rows, onset_rows = [], []

    # ---------- PART A: asymmetric cost of queue-length forecasts ----------
    rf_mean = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED, n_jobs=-1)
    rf_mean.fit(tr[hist], tr[ycol])
    diff_tr = (tr[ycol] - tr["queue_length_lag0"]).to_numpy()
    for C in COSTS:
        tau = C / (C + 1)
        margin = float(np.quantile(diff_tr, tau))
        gb = {}
        for fs, cols in [("history-only", hist), ("rich features", rich)]:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=200,
                                              learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)
            gb[fs] = (m.fit(tr[cols], tr[ycol]), cols)
        methods = {
            "Persistence": lambda d: d["queue_length_lag0"].to_numpy(),
            "Persistence + margin": lambda d: d["queue_length_lag0"].to_numpy() + margin,
            "RF (mean)": lambda d: rf_mean.predict(d[hist]),
            "Quantile GB, history-only": lambda d: gb["history-only"][0].predict(d[hist]),
            "Quantile GB, rich features": lambda d: gb["rich features"][0].predict(d[rich]),
        }
        for name, f in methods.items():
            per_run = []
            for rid in TEST_RUNS:
                d = te[te.run_id == rid]
                y, yhat = d[ycol].to_numpy(), f(d)
                c = d[ccol].to_numpy() == 1
                short = np.clip(y - yhat, 0, None)
                per_run.append((asym_cost(y, yhat, C), short[c].mean() if c.any() else np.nan,
                                (yhat[c] < y[c]).mean() if c.any() else np.nan))
            per_run = np.array(per_run)
            cost_rows.append(dict(queue=queue, rho=rho, C=C, method=name,
                                  cost=per_run[:, 0].mean(), cost_std=per_run[:, 0].std(ddof=1),
                                  shortfall_in_congestion=np.nanmean(per_run[:, 1]),
                                  share_under_in_congestion=np.nanmean(per_run[:, 2])))

    # ---------- PART B: early warning of congestion onsets ----------
    clf = {}
    for fs, cols in [("history-only", hist), ("rich features", rich)]:
        m = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
        clf[fs] = (m.fit(tr[cols], tr[ccol]), cols)
    p_va = {fs: m.predict_proba(va[cols])[:, 1] for fs, (m, cols) in clf.items()}
    p_te = {fs: m.predict_proba(te[cols])[:, 1] for fs, (m, cols) in clf.items()}
    q_va, q_te = va["queue_length_lag0"].to_numpy(), te["queue_length_lag0"].to_numpy()
    y_va, y_te = va[ccol].to_numpy(), te[ccol].to_numpy()

    for C in COSTS:
        a = choose(lambda g: class_cost(y_va, (q_va > g * thr).astype(int), C), A_GRID)
        cuts = {fs: choose(lambda g: class_cost(y_va, (p_va[fs] >= g).astype(int), C), CUT_GRID)
                for fs in clf}
        alarms = {
            "Persistence": (q_te > thr).astype(int),
            "Lowered-threshold rule": (q_te > a * thr).astype(int),
            "RF classifier, history-only": (p_te["history-only"] >= cuts["history-only"]).astype(int),
            "RF classifier, rich features": (p_te["rich features"] >= cuts["rich features"]).astype(int),
        }
        setting = {"Persistence": "", "Lowered-threshold rule": f"a={a:.2f}",
                   "RF classifier, history-only": f"cut={cuts['history-only']:.2f}",
                   "RF classifier, rich features": f"cut={cuts['rich features']:.2f}"}
        te_runs = te["run_id"].to_numpy()
        for name, al in alarms.items():
            costs, warned_all, lead_all, prec, rec = [], [], [], [], []
            for rid in TEST_RUNS:
                mask = te_runs == rid
                d, a_r, y_r = te[mask], al[mask], y_te[mask]
                costs.append(class_cost(y_r, a_r, C))
                prec.append(((a_r == 1) & (y_r == 1)).sum() / max((a_r == 1).sum(), 1))
                rec.append(((a_r == 1) & (y_r == 1)).sum() / max((y_r == 1).sum(), 1))
                w, l = onsets_and_warnings(d, a_r, thr)
                warned_all.append(w); lead_all.append(l)
            w = np.concatenate(warned_all); l = np.concatenate(lead_all)
            per_run_warn = [(x >= USEFUL).mean() for x in lead_all if len(x)]
            onset_rows.append(dict(queue=queue, rho=rho, C=C, method=name, setting=setting[name],
                                   n_onsets=len(w), any_warning_rate=w.mean() if len(w) else np.nan,
                                   early_warning_rate=(l >= USEFUL).mean() if len(l) else np.nan,
                                   median_lead=np.median(l) if len(l) else np.nan,
                                   early_warning_rate_std=np.std(per_run_warn, ddof=1) if len(per_run_warn) > 1 else np.nan,
                                   precision=np.mean(prec), recall=np.mean(rec),
                                   cost=np.mean(costs), cost_std=np.std(costs, ddof=1)))
    return cost_rows, onset_rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    cost_rows, onset_rows = [], []
    for queue in QUEUES:
        for rho in RHOS:
            if not QUEUES[queue](rho).exists():
                print(f"(skipping {queue} rho={rho}: no processed data)")
                continue
            c, o = run_setting(queue, rho)
            cost_rows += c; onset_rows += o
            print(f"done: {queue} rho={rho}", flush=True)
    cost = pd.DataFrame(cost_rows)
    ons = pd.DataFrame(onset_rows)
    cost.to_csv(RESULTS / f"early_warning_cost{OUT_SFX}.csv", index=False)
    ons.to_csv(RESULTS / f"early_warning_onsets{OUT_SFX}.csv", index=False)

    pd.set_option("display.width", 220)
    bar = "=" * 110
    order = [q for q in QUEUES if q in set(cost.queue)]

    # Part A tables: cost relative to persistence + margin (the fair simple baseline)
    base = cost[cost.method == "Persistence + margin"].set_index(["queue", "rho", "C"])["cost"]
    cost["vs_margin_pct"] = 100 * (1 - cost.cost / cost.set_index(["queue", "rho", "C"]).index.map(base))
    for C in COSTS:
        print(f"\n{bar}\nPART A, C = {C} (one customer too few costs {C}x one too many): test cost, "
              f"% lower than 'persistence + margin' (positive = better)\n{bar}")
        t = cost[(cost.C == C) & (cost.method != "Persistence + margin")].pivot_table(
            index="rho", columns=["queue", "method"], values="vs_margin_pct")
        t = t.reindex(columns=[(q, m) for q in order for m in
                               ["Persistence", "RF (mean)", "Quantile GB, history-only", "Quantile GB, rich features"]
                               if (q, m) in t.columns])
        t.columns = [f"{q} | {m.replace('Quantile GB, ', 'QGB ')}" for q, m in t.columns]
        for q in order:
            sub = t[[c for c in t.columns if c.startswith(q + " |")]]
            sub.columns = [c.split(" | ")[1] for c in sub.columns]
            print(f"\n  {q}")
            print(sub.round(1).to_string())

    print(f"\n{bar}\nPART A, C = {MAIN_C}: average shortfall during congestion (customers the forecast is short by)\n{bar}")
    t = cost[cost.C == MAIN_C].pivot_table(index=["queue", "rho"], columns="method",
                                           values="shortfall_in_congestion")
    t = t.reindex(order, level=0)
    print(t[["Persistence", "Persistence + margin", "RF (mean)", "Quantile GB, history-only",
             "Quantile GB, rich features"]].round(2).to_string())

    # Part B tables
    for C in COSTS:
        print(f"\n{bar}\nPART B, C = {C}: share of congestion onsets with a warning at least {USEFUL} steps ahead "
              f"[share with any warning] (median lead time)\n{bar}")
        o = ons[ons.C == C].copy()
        o["cell"] = o.apply(lambda r: f"{r.early_warning_rate:.2f} [{r.any_warning_rate:.2f}] ({r.median_lead:.0f})", axis=1)
        t = o.pivot_table(index=["queue", "rho"], columns="method", values="cell", aggfunc="first")
        t = t.reindex(order, level=0)
        print(t[["Persistence", "Lowered-threshold rule", "RF classifier, history-only",
                 "RF classifier, rich features"]].to_string())
    print(f"\n{bar}\nPART B, C = {MAIN_C}: per-step cost (missed congested step = {MAIN_C}, false alarm = 1), "
          f"% lower than persistence (positive = better)\n{bar}")
    o = ons[ons.C == MAIN_C].copy()
    pb = o[o.method == "Persistence"].set_index(["queue", "rho"])["cost"]
    o["vs_persistence_pct"] = 100 * (1 - o.cost / o.set_index(["queue", "rho"]).index.map(pb))
    t = o.pivot_table(index=["queue", "rho"], columns="method", values="vs_persistence_pct").reindex(order, level=0)
    print(t[["Lowered-threshold rule", "RF classifier, history-only", "RF classifier, rich features"]].round(1).to_string())
    print(f"\n{bar}\nPART B, C = {MAIN_C}: alarm precision / recall (per step) and number of onsets\n{bar}")
    o["cell"] = o.apply(lambda r: f"{r.precision:.2f} / {r.recall:.2f}", axis=1)
    t = o.pivot_table(index=["queue", "rho"], columns="method", values="cell", aggfunc="first").reindex(order, level=0)
    t["onsets"] = o.groupby(["queue", "rho"])["n_onsets"].first()
    print(t[["Persistence", "Lowered-threshold rule", "RF classifier, history-only",
             "RF classifier, rich features", "onsets"]].to_string())

    print(f"\nSaved: results/early_warning_cost{OUT_SFX}.csv, results/early_warning_onsets{OUT_SFX}.csv")

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {"Lowered-threshold rule": ":", "RF classifier, history-only": "--", "RF classifier, rich features": "-"}
    colors = dict(zip(order, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    o = ons[ons.C == MAIN_C]
    for q in order:
        for mth, ls in styles.items():
            s = o[(o.queue == q) & (o.method == mth)].sort_values("rho")
            ax[0].plot(s.rho, s.early_warning_rate, ls=ls, marker="o", color=colors[q],
                       label=f"{q}, {mth.replace('RF classifier, ', 'RF ')}")
    ax[0].set_title(f"Onsets warned at least {USEFUL} steps ahead (C = {MAIN_C}; persistence = 0)", fontsize=10)
    ax[0].set_xlabel("rho"); ax[0].set_ylabel(f"share of onsets with lead >= {USEFUL}"); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=6.5, ncol=2); ax[0].grid(alpha=0.3)
    c = cost[cost.C == MAIN_C]
    for q in order:
        for mth, ls in [("Persistence", ":"), ("Quantile GB, history-only", "--"), ("Quantile GB, rich features", "-")]:
            s = c[(c.queue == q) & (c.method == mth)].sort_values("rho")
            ax[1].plot(s.rho, s.vs_margin_pct, ls=ls, marker="o", color=colors[q],
                       label=f"{q}, {mth.replace('Quantile GB, ', 'QGB ')}")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_title(f"Asymmetric forecast cost vs persistence + margin (C = {MAIN_C})", fontsize=10)
    ax[1].set_xlabel("rho"); ax[1].set_ylabel("% lower cost (positive = better)")
    ax[1].legend(fontsize=6.5, ncol=2); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"early_warning_summary{OUT_SFX}.png", dpi=150)
    print(f"Saved: results/plots/early_warning_summary{OUT_SFX}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--callcenter", action="store_true",
                    help="run on the two MAPs fitted to the call-centre arrivals instead")
    if ap.parse_args().callcenter:
        QUEUES = CALLCENTER_QUEUES
        OUT_SFX = "_callcenter"
    main()