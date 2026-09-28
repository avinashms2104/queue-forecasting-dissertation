"""
Full regression summary across all queue types: MAE *and* RMSE for every model
(Naive persistence, Linear Regression, Decision Tree, Random Forest).

Why this exists: train_*_baseline_models.py compute RMSE for every model and
test run, but the ALL_MEAN summary rows only carry MAE. RMSE and the Linear
Regression / Decision Tree results were therefore never summarised anywhere.
This script averages the per-run rows (5 held-out test runs) and writes one
tidy table. Trains nothing; runs in seconds.

Restricted to history-only features (the feature set used for all headline
comparisons; rich features were compared separately and made no difference).

Usage:  python modeling/summarise_regression_models.py
Writes: results/regression_model_summary_all.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["Naive (persistence)", "Linear Regression", "Decision Tree", "Random Forest"]
SHORT = {"Naive (persistence)": "Naive", "Linear Regression": "LinReg",
         "Decision Tree": "DecTree", "Random Forest": "RandFor"}
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


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

# mean and std over the 5 test runs
g = runs.groupby(["queue", "rho", "horizon", "model"])
summ = g.agg(MAE=("MAE", "mean"), MAE_std=("MAE", "std"),
             RMSE=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
             n_runs=("MAE", "size")).reset_index()

# relative improvement over persistence, for both metrics
naive = summ[summ.model == "Naive (persistence)"][["queue", "rho", "horizon", "MAE", "RMSE"]] \
    .rename(columns={"MAE": "MAE_naive", "RMSE": "RMSE_naive"})
summ = summ.merge(naive, on=["queue", "rho", "horizon"])
summ["MAE_rel_improvement"] = 1 - summ["MAE"] / summ["MAE_naive"]
summ["RMSE_rel_improvement"] = 1 - summ["RMSE"] / summ["RMSE_naive"]
summ.drop(columns=["MAE_naive", "RMSE_naive"]).to_csv(RESULTS / "regression_model_summary_all.csv", index=False)

pd.set_option("display.width", 220)
med = summ[summ.horizon == "medium"]

print("=" * 110)
print("A. MEDIUM HORIZON, history-only: MAE and RMSE by model (mean of 5 test runs)")
print("=" * 110)
for q in QUEUES:
    sub = med[med.queue == q]
    mae = sub.pivot(index="rho", columns="model", values="MAE")[MODELS].rename(columns=SHORT).add_prefix("MAE ")
    rmse = sub.pivot(index="rho", columns="model", values="RMSE")[MODELS].rename(columns=SHORT).add_prefix("RMSE ")
    print(f"\n{q}")
    print(pd.concat([mae, rmse], axis=1).round(2).to_string())

print("\n" + "=" * 110)
print("B. Random Forest RMSE improvement over persistence (%), medium horizon")
print("   (compare with the MAE version in the service-variance table)")
print("=" * 110)
b = med[med.model == "Random Forest"].pivot(index="rho", columns="queue", values="RMSE_rel_improvement")[QUEUES]
print((b * 100).round(1).to_string())

print("\n" + "=" * 110)
print("C. MAE improvement over persistence (%), averaged over rho, medium horizon: every model")
print("=" * 110)
c = med[med.model != "Naive (persistence)"].groupby(["queue", "model"])["MAE_rel_improvement"].mean().unstack("model")
print((c.loc[QUEUES, MODELS[1:]] * 100).round(1).to_string())

print("\n" + "=" * 110)
print("D. Does RMSE tell a different story from MAE? Sign of RF improvement, all rho x horizons")
print("=" * 110)
rf = summ[summ.model == "Random Forest"]
for q in QUEUES:
    s = rf[rf.queue == q]
    disagree = ((s.MAE_rel_improvement > 0) != (s.RMSE_rel_improvement > 0)).sum()
    print(f"{q:20s}: RF beats persistence on MAE in {(s.MAE_rel_improvement > 0).sum():2d}/{len(s)} cells, "
          f"on RMSE in {(s.RMSE_rel_improvement > 0).sum():2d}/{len(s)}; MAE and RMSE disagree in {disagree}")
print("\nSaved: results/regression_model_summary_all.csv")