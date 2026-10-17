"""
Early warning: a fair, run-by-run comparison (Azam, 16 Oct)
==========================================================

Azam asked to check the early-warning comparison more carefully:
  * report the cost of MISSED congestion and the cost of FALSE ALARMS separately,
    as well as the total;
  * show whether ML keeps its advantage over the lowered-threshold rule;
  * select every threshold on the validation runs, separately for each cost ratio,
    and compare on the test runs only;
  * show the variability across the independent test runs;
  * define the horizon and the congestion onset precisely, and report the share of
    congestion episodes warned in advance, their lead times and the number of
    false alarms.

Definitions (time is measured in mean service times; one observation per time unit)
-----------------------------------------------------------------------------------
Congested        Q(t) > threshold, the 80th percentile of the queue length in the
                 20 training runs (as everywhere in the project).
Horizon          h = 10 time units. An ALARM at time t is a prediction that the queue
                 will be congested at time t + h.
Onset            a time s with Q(s) > threshold after at least h consecutive observed
                 times with Q <= threshold (so a new congestion episode, not a flicker).
Lead time        the number of consecutive time units with the alarm on immediately
                 before s (0 to h). "Warned in advance" = lead >= 1; "useful warning"
                 = lead >= 5 (at least half the horizon ahead).
False alarm      an alarm episode (a maximal run of consecutive alarm times) that
episode          starts while the queue is not congested and is not followed by any
                 congestion up to h time units after it ends.
Cost             per observation: C for each time t whose target (congested at t + h)
                 was 1 but no alarm was raised (a missed congested observation), and 1
                 for each alarm whose target was 0 (a false-alarm observation). The
                 cost ratios C = 2, 5, 10 are illustrative choices, not estimated
                 operational costs. Missed part = C * (missed observations) / n,
                 false-alarm part = (false-alarm observations) / n, total = their sum.

Methods (all tuned on the validation runs 20-24, separately for each C)
------------------------------------------------------------------------
Persistence                 alarm when Q(t) > threshold. Because it uses the congestion
                            threshold itself, it cannot raise an alarm before an onset.
Lowered-threshold rule      alarm when Q(t) > a * threshold, a chosen from 0.30 to 1.00
                            (persistence with a lower threshold; it CAN warn early).
RF classifier, history-only alarm when P(congested at t + h) >= cut, cut from 0.02 to 0.94
RF classifier, rich features
The same models and grids as early_warning_analysis.py, so the numbers agree with it.

Paired comparison: for each of the 5 test runs, the % lower total cost of the RF
classifier (with rich features) than of the lowered-threshold rule; reported as mean,
standard deviation, 95% t-interval over the 5 runs and the number of runs where RF wins.

Usage:   python modeling/early_warning_fair_comparison.py                 (every setting found)
         python modeling/early_warning_fair_comparison.py --callcenter    (only the four call-centre
                                                                           settings: Erlang-2 and real service)
Writes:  results/early_warning_fair_runs{_callcenter}.csv      (one row per queue, rho, C, method, test run)
         results/early_warning_fair_summary{_callcenter}.csv   (means and standard deviations over runs)
         results/early_warning_fair_paired{_callcenter}.csv    (RF minus rule, run by run)
         results/plots/early_warning_fair{_callcenter}.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
SEED = 42

P = DATA / "mapph1" / "processed"
QUEUES = {
    "none (H2)": lambda r: DATA / "h2ph1" / "processed" / f"rich_features_rho{r}.csv",
    "weak": lambda r: P / f"rich_features_weak_rho{r}.csv",
    "medium": lambda r: P / f"rich_features_rho{r}.csv",
    "strong": lambda r: P / f"rich_features_strong_rho{r}.csv",
    "cc afternoon": lambda r: P / f"rich_features_cc_afternoon_rho{r}.csv",
    "cc evening": lambda r: P / f"rich_features_cc_evening_rho{r}.csv",
    "cc afternoon, real service": lambda r: P / f"rich_features_cc_afternoon_realsvc_rho{r}.csv",
    "cc evening, real service": lambda r: P / f"rich_features_cc_evening_realsvc_rho{r}.csv",
}
OUT_SFX = ""      # "_callcenter" with --callcenter (only the four call-centre settings)
RHOS = [0.5, 0.7, 0.8, 0.9]
HNAME, H = "medium", 10
COSTS = [2, 5, 10]
MAIN_C = 5
USEFUL = 5
VAL_RUNS, TEST_RUNS = list(range(20, 25)), list(range(25, 30))
CUT_GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)
A_GRID = np.round(np.arange(0.30, 1.01, 0.05), 2)
METHODS = ["Persistence", "Lowered-threshold rule", "RF classifier, history-only", "RF classifier, rich features"]


def load(path):
    df = pd.read_csv(path)
    hist = [c for c in df.columns if c.startswith("queue_length_lag")]
    base = {"arrival_process", "service_erlang_k", "rho", "run_id", "t", "congestion_threshold", "split"}
    rich = [c for c in df.columns if not c.startswith("target_") and c not in base]
    return df, hist, rich


def class_cost(y, alarm, C):
    return (C * ((y == 1) & (alarm == 0)) + ((y == 0) & (alarm == 1))).mean()


def choose(score_fn, grid):
    return grid[int(np.argmin([score_fn(g) for g in grid]))]


def episode_stats(run_df, alarm, thr):
    """Onset lead times and false-alarm episodes for one run, on the full time grid."""
    t = run_df["t"].to_numpy().astype(int)
    t0 = t.min()
    n = t.max() - t0 + 1
    cong = np.full(n, np.nan)
    al = np.zeros(n, dtype=bool)
    cong[t - t0] = (run_df["queue_length_lag0"].to_numpy() > thr).astype(float)
    al[t - t0] = alarm.astype(bool)

    leads = []
    for s in range(H, n):
        if cong[s] != 1:
            continue
        before = cong[s - H:s]
        if np.isnan(before).any() or before.any():
            continue
        L = 0
        while L < H and al[s - 1 - L]:
            L += 1
        leads.append(L)

    false_eps = 0
    i = 0
    while i < n:
        if al[i]:
            j = i
            while j + 1 < n and al[j + 1]:
                j += 1
            if cong[i] == 0:                                   # starts while not congested
                window = cong[i:min(n, j + H + 1)]
                if not np.any(window == 1):
                    false_eps += 1
            i = j + 1
        else:
            i += 1
    return np.array(leads), false_eps, n


def run_setting(queue, rho):
    df, hist, rich = load(QUEUES[queue](rho))
    thr = float(df["congestion_threshold"].iloc[0])
    ccol = f"target_congested_{HNAME}"
    tr = df[df.split == "train"]
    va = df[df.run_id.isin(VAL_RUNS)]
    te = df[df.run_id.isin(TEST_RUNS)]

    clf = {}
    for fs, cols in [("history-only", hist), ("rich features", rich)]:
        m = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
        clf[fs] = (m.fit(tr[cols], tr[ccol]), cols)
    p_va = {fs: m.predict_proba(va[cols])[:, 1] for fs, (m, cols) in clf.items()}
    p_te = {fs: m.predict_proba(te[cols])[:, 1] for fs, (m, cols) in clf.items()}
    q_va, q_te = va["queue_length_lag0"].to_numpy(), te["queue_length_lag0"].to_numpy()
    y_va, y_te = va[ccol].to_numpy(), te[ccol].to_numpy()
    te_runs = te["run_id"].to_numpy()

    rows = []
    for C in COSTS:
        a = choose(lambda g: class_cost(y_va, (q_va > g * thr).astype(int), C), A_GRID)
        cuts = {fs: choose(lambda g: class_cost(y_va, (p_va[fs] >= g).astype(int), C), CUT_GRID) for fs in clf}
        alarms = {
            "Persistence": (q_te > thr).astype(int),
            "Lowered-threshold rule": (q_te > a * thr).astype(int),
            "RF classifier, history-only": (p_te["history-only"] >= cuts["history-only"]).astype(int),
            "RF classifier, rich features": (p_te["rich features"] >= cuts["rich features"]).astype(int),
        }
        setting = {"Persistence": "a=1.00", "Lowered-threshold rule": f"a={a:.2f}",
                   "RF classifier, history-only": f"cut={cuts['history-only']:.2f}",
                   "RF classifier, rich features": f"cut={cuts['rich features']:.2f}"}
        for name, al in alarms.items():
            for rid in TEST_RUNS:
                mask = te_runs == rid
                y, a_r = y_te[mask], al[mask]
                n_obs = len(y)
                fn = int(((y == 1) & (a_r == 0)).sum())
                fp = int(((y == 0) & (a_r == 1)).sum())
                leads, false_eps, n_time = episode_stats(te[mask], a_r, thr)
                rows.append(dict(
                    queue=queue, rho=rho, C=C, method=name, setting=setting[name], test_run=rid,
                    n_obs=n_obs, missed_obs=fn, false_alarm_obs=fp,
                    cost_missed=C * fn / n_obs, cost_false_alarm=fp / n_obs,
                    cost_total=(C * fn + fp) / n_obs,
                    onsets=len(leads),
                    warned_any=float((leads >= 1).mean()) if len(leads) else np.nan,
                    warned_useful=float((leads >= USEFUL).mean()) if len(leads) else np.nan,
                    median_lead_warned=float(np.median(leads[leads >= 1])) if (leads >= 1).any() else 0.0,
                    mean_lead=float(leads.mean()) if len(leads) else np.nan,
                    false_alarm_episodes=false_eps,
                    false_alarm_episodes_per_1000=1000 * false_eps / n_time))
    return rows


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in QUEUES:
        for rho in RHOS:
            if not QUEUES[q](rho).exists():
                print(f"(skipping {q} rho={rho}: no processed data)")
                continue
            rows += run_setting(q, rho)
            print(f"done: {q} rho={rho}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / f"early_warning_fair_runs{OUT_SFX}.csv", index=False)
    order = [q for q in QUEUES if q in set(runs.queue)]

    # ---------- summary over the 5 test runs ----------
    metrics = ["cost_total", "cost_missed", "cost_false_alarm", "warned_any", "warned_useful",
               "median_lead_warned", "false_alarm_episodes_per_1000", "onsets"]
    g = runs.groupby(["queue", "rho", "C", "method"])
    summ = g[metrics].mean().join(g[metrics].std(ddof=1), rsuffix="_sd").reset_index()
    summ = summ.merge(runs.groupby(["queue", "rho", "C", "method"])["setting"].first().reset_index())
    summ.to_csv(RESULTS / f"early_warning_fair_summary{OUT_SFX}.csv", index=False)

    # ---------- paired: each method vs persistence and RF vs rule, run by run ----------
    piv = runs.pivot_table(index=["queue", "rho", "C", "test_run"], columns="method", values="cost_total")
    pair_rows = []
    for (q, rho, C), d in piv.groupby(level=[0, 1, 2]):
        for rf in ["RF classifier, history-only", "RF classifier, rich features"]:
            diff = 100 * (1 - d[rf] / d["Lowered-threshold rule"])
            lo, hi = ci95(diff)
            pair_rows.append(dict(queue=q, rho=rho, C=C, comparison=f"{rf} vs rule",
                                  mean_pct=diff.mean(), sd_pct=diff.std(ddof=1), ci_low=lo, ci_high=hi,
                                  runs_better=int((diff > 0).sum()), n_runs=len(diff)))
        for m in METHODS[1:]:
            diff = 100 * (1 - d[m] / d["Persistence"])
            lo, hi = ci95(diff)
            pair_rows.append(dict(queue=q, rho=rho, C=C, comparison=f"{m} vs persistence",
                                  mean_pct=diff.mean(), sd_pct=diff.std(ddof=1), ci_low=lo, ci_high=hi,
                                  runs_better=int((diff > 0).sum()), n_runs=len(diff)))
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(RESULTS / f"early_warning_fair_paired{OUT_SFX}.csv", index=False)

    # ---------- printed tables ----------
    pd.set_option("display.width", 230)
    bar = "=" * 120
    short = {"Persistence": "persistence", "Lowered-threshold rule": "rule",
             "RF classifier, history-only": "RF hist", "RF classifier, rich features": "RF rich"}

    def cell(r, k, f):
        return f"{r[k]:{f}} ({r[k + '_sd']:{f}})"

    print(f"\n{bar}\nDefinitions: horizon h = {H} time units; onset = first congested time after >= {H} quiet times;"
          f"\nuseful warning = alarm on for >= {USEFUL} time units just before the onset; cost per observation."
          f"\nAll thresholds chosen on validation runs 20-24 for each C; numbers below are test runs 25-29,"
          f"\nmean (standard deviation over the 5 runs).\n{bar}")

    s = summ[summ.C == MAIN_C].set_index(["queue", "rho", "method"])
    print(f"\n{bar}\nC = {MAIN_C}: cost per observation = missed part + false-alarm part\n{bar}")
    tab = []
    for q in order:
        for rho in sorted(s.loc[q].index.get_level_values(0).unique()):
            for m in METHODS:
                r = s.loc[(q, rho, m)]
                tab.append(dict(queue=q, rho=rho, method=short[m], setting=r["setting"],
                                total=cell(r, "cost_total", ".4f"), missed=cell(r, "cost_missed", ".4f"),
                                false_alarm=cell(r, "cost_false_alarm", ".4f")))
    print(pd.DataFrame(tab).to_string(index=False))

    print(f"\n{bar}\nC = {MAIN_C}: congestion episodes warned in advance, lead times and false alarms\n{bar}")
    tab = []
    for q in order:
        for rho in sorted(s.loc[q].index.get_level_values(0).unique()):
            for m in METHODS:
                r = s.loc[(q, rho, m)]
                tab.append(dict(queue=q, rho=rho, method=short[m], onsets_per_run=f"{r['onsets']:.0f}",
                                warned_ahead=cell(r, "warned_any", ".2f"),
                                warned_5_ahead=cell(r, "warned_useful", ".2f"),
                                median_lead=f"{r['median_lead_warned']:.1f}",
                                false_alarm_eps_per_1000=cell(r, "false_alarm_episodes_per_1000", ".1f")))
    print(pd.DataFrame(tab).to_string(index=False))

    for C in COSTS:
        print(f"\n{bar}\nC = {C}: RF classifier (rich features) against the lowered-threshold rule, run by run: "
              f"% lower total cost\n(positive = RF better; mean, sd, 95% interval over 5 test runs, runs where RF is better)\n{bar}")
        d = pair[(pair.C == C) & (pair.comparison == "RF classifier, rich features vs rule")]
        d = d.assign(result=d.apply(lambda r: f"{r.mean_pct:+6.1f}  sd {r.sd_pct:4.1f}  "
                                              f"[{r.ci_low:+6.1f}, {r.ci_high:+6.1f}]  {r.runs_better}/{r.n_runs}", axis=1))
        t = d.pivot(index="rho", columns="queue", values="result")
        print(t[[q for q in order if q in t.columns]].to_string())

    print(f"\n{bar}\nC = {MAIN_C}: % lower total cost than persistence (mean, sd over 5 test runs)\n{bar}")
    d = pair[(pair.C == MAIN_C) & pair.comparison.str.endswith("vs persistence")].copy()
    d["m"] = d.comparison.str.replace(" vs persistence", "").map(short)
    d["result"] = d.apply(lambda r: f"{r.mean_pct:+5.1f} ({r.sd_pct:.1f})", axis=1)
    t = d.pivot_table(index=["queue", "rho"], columns="m", values="result", aggfunc="first").reindex(order, level=0)
    print(t[["rule", "RF hist", "RF rich"]].to_string())

    print(f"\nSaved: results/early_warning_fair_runs{OUT_SFX}.csv, results/early_warning_fair_summary{OUT_SFX}.csv, "
          f"results/early_warning_fair_paired{OUT_SFX}.csv")

    # ---------- plot: run-by-run, C = MAIN_C ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    d = runs[runs.C == MAIN_C]
    pers = d[d.method == "Persistence"].set_index(["queue", "rho", "test_run"])["cost_total"]
    for k, (mth, col) in enumerate([("Lowered-threshold rule", "C1"), ("RF classifier, rich features", "C0")]):
        dm = d[d.method == mth].set_index(["queue", "rho", "test_run"])
        red = (100 * (1 - dm["cost_total"] / pers)).rename("red").reset_index()
        red["x"] = red["queue"].map({q: i for i, q in enumerate(order)}) + \
                   red["rho"].map({r: j for j, r in enumerate(RHOS)}) * 0.18 - 0.27 + 0.05 * k
        ax[0].scatter(red["x"], red["red"], s=10, color=col, alpha=0.7,
                      label=mth.replace("RF classifier, ", "RF "))
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, fontsize=8)
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_ylabel("% lower total cost than persistence")
    ax[0].set_title(f"Each dot = one test run (C = {MAIN_C}); within a queue rho = 0.5, 0.7, 0.8, 0.9 left to right",
                    fontsize=9)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    dp = pair[(pair.C == MAIN_C) & (pair.comparison == "RF classifier, rich features vs rule")]
    for q in order:
        e = dp[dp.queue == q].sort_values("rho")
        ax[1].errorbar(e.rho + 0.004 * order.index(q), e.mean_pct,
                       yerr=[e.mean_pct - e.ci_low, e.ci_high - e.mean_pct], marker="o", capsize=3, label=q)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_xlabel("rho"); ax[1].set_ylabel("% lower total cost, RF (rich) vs rule")
    ax[1].set_title(f"RF against the lowered-threshold rule (C = {MAIN_C}), mean and 95% interval over 5 runs",
                    fontsize=9)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"early_warning_fair{OUT_SFX}.png", dpi=150)
    print(f"Saved: results/plots/early_warning_fair{OUT_SFX}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--callcenter", action="store_true",
                    help="only the call-centre settings (Erlang-2 and real service variability)")
    if ap.parse_args().callcenter:
        QUEUES = {k: v for k, v in QUEUES.items() if k.startswith("cc")}
        OUT_SFX = "_callcenter"
    main()