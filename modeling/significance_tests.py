"""
Paired significance tests: Random Forest vs naive persistence, regression.

Motivation (notebook Sections 10.1 and 3.8): the literature review makes
Velarde et al.'s habit of testing whether an apparent improvement is real a
central methodological point, but the results so far only compared MAE gaps
with run-to-run standard deviations by eye. This script does it properly.

Design
------
* Pairing: the SAME 5 held-out test runs are scored by both methods, so for
  each (queue, rho, horizon, metric) we form 5 paired differences
        d_run = error_persistence(run) - error_RandomForest(run)
  (positive = Random Forest better) and test whether mean(d) differs from 0.
* Test: paired t-test (4 degrees of freedom). It assumes roughly normal
  differences, which cannot be checked well with 5 points, so the number of
  runs (out of 5) where Random Forest wins is reported beside every p-value as
  a distribution-free check. Rank-based tests are NOT used as the primary
  test: with n = 5 the smallest possible two-sided p from a sign or Wilcoxon
  test is 0.0625, so they could never reach 5%.
* Multiple testing: 30 (queue x rho) tests per metric at the medium horizon,
  120 over all horizons. Uncorrected, about 5% of true-null tests would look
  "significant" by chance, so Holm-adjusted p-values are reported (they
  control the family-wise error rate; adjusted within each metric).
* Limitation: 5 test runs gives low power, so "not significant" means "not
  shown", not "no difference". Only run-to-run variation is captured, because
  each model is trained once on the 20 training runs.

Usage:  python modeling/significance_tests.py
Writes: results/regression_significance_tests.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHA = 0.05
HORIZON_ORDER = ["short", "medium", "extra_long", "long"]


def load(name, queue_label, mask=None):
    d = pd.read_csv(RESULTS / name)
    d = d[(d.dataset == "history_only") & (d.test_run_id.astype(str) != "ALL_MEAN")]
    if mask is not None:
        d = d[mask(d)]
    d = d.assign(queue=queue_label)
    return d[["queue", "rho", "horizon", "model", "test_run_id", "MAE", "RMSE"]]


parts = [
    load("model_comparison_regression.csv", "M/M/1"),
    load("mg1_model_comparison_regression.csv", "M/G/1 Weibull 2.0", lambda d: d.weibull_shape == 2.0),
    load("mg1_model_comparison_regression.csv", "M/G/1 Weibull 0.5", lambda d: d.weibull_shape == 0.5),
    load("mph1_model_comparison_regression.csv", "M/PH/1 Erlang-2", lambda d: d.erlang_k == 2),
    load("phph1_model_comparison_regression.csv", "PH/PH/1 E2/E2",
         lambda d: (d.arrival_erlang_k == 2) & (d.service_erlang_k == 2)),
]
runs = pd.concat(parts, ignore_index=True)
QUEUES = list(dict.fromkeys(runs.queue))
RHOS = sorted(runs.rho.unique())


def holm(p):
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


rows = []
for (queue, rho, horizon), grp in runs.groupby(["queue", "rho", "horizon"]):
    naive = grp[grp.model == "Naive (persistence)"].set_index("test_run_id")
    rf = grp[grp.model == "Random Forest"].set_index("test_run_id")
    common = naive.index.intersection(rf.index)
    for metric in ["MAE", "RMSE"]:
        d = (naive.loc[common, metric] - rf.loc[common, metric]).values   # positive = RF better
        sd = d.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            t, p = np.nan, 1.0
        else:
            t, p = stats.ttest_1samp(d, 0.0)
        rows.append(dict(queue=queue, rho=rho, horizon=horizon, metric=metric,
                         mean_diff=d.mean(), sd_diff=sd, t=t, df=len(d) - 1, p_raw=p,
                         rf_wins=int((d > 0).sum()), n_runs=len(d),
                         rel_improvement=d.mean() / naive.loc[common, metric].mean()))
res = pd.DataFrame(rows)

# Holm adjustment within each metric x (medium horizon | all horizons) family
res["p_holm_medium"] = np.nan
res["p_holm_all"] = np.nan
for metric in ["MAE", "RMSE"]:
    m = res.metric == metric
    mm = m & (res.horizon == "medium")
    res.loc[mm, "p_holm_medium"] = holm(res.loc[mm, "p_raw"])
    res.loc[m, "p_holm_all"] = holm(res.loc[m, "p_raw"])
res.to_csv(RESULTS / "regression_significance_tests.csv", index=False)


def cell(r):
    who = "RF" if r.mean_diff > 0 else "PS"
    return f"{who} {r.rf_wins}/{r.n_runs} p={r.p_holm_medium:.3f}"


print("Cell = who is better on average | runs won by Random Forest out of 5 | Holm-adjusted p")
print("RF = Random Forest better, PS = persistence better. Treat a difference as shown only if the adjusted p < 0.05.\n")
for metric in ["MAE", "RMSE"]:
    sub = res[(res.metric == metric) & (res.horizon == "medium")]
    tab = pd.DataFrame({q: {rho: cell(sub[(sub.queue == q) & (sub.rho == rho)].iloc[0]) for rho in RHOS}
                        for q in QUEUES})
    print("=" * 120)
    print(f"{metric}, MEDIUM horizon (30 tests, Holm-adjusted)")
    print("=" * 120)
    pd.set_option("display.width", 250)
    print(tab.to_string())
    print()

print("=" * 120)
print(f"SUMMARY: cells where the difference is significant at {ALPHA:.0%} after Holm correction")
print("=" * 120)
for metric in ["MAE", "RMSE"]:
    for label, col, hsel in [("medium horizon (30 tests)", "p_holm_medium", res.horizon == "medium"),
                             ("all four horizons (120 tests)", "p_holm_all", res.horizon.notna())]:
        s = res[(res.metric == metric) & hsel]
        sig = s[s[col] < ALPHA]
        print(f"{metric:5s} {label:30s}: RF significantly better in {(sig.mean_diff > 0).sum():3d}, "
              f"persistence significantly better in {(sig.mean_diff < 0).sum():3d}, "
              f"not significant in {len(s) - len(sig):3d} of {len(s)}")
print("\nSaved: results/regression_significance_tests.csv")