"""
Extra summaries for the M/PH/1 results (reads the CSVs already produced by
train_mph1_baseline_models.py; trains nothing, runs in seconds).

  1. Rich vs history-only features: relative MAE difference for every model,
     rho and horizon (Azam's request: quantify across all values and horizons).
         difference = 1 - MAE_rich / MAE_history     (positive = rich is better)
  2. Congestion classification, medium horizon, history-only: precision,
     recall and F1 for Random Forest vs the persistence baseline.

Usage:  python modeling/mph1_extra_summaries.py
Writes: results/mph1_feature_set_comparison.csv
"""

from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

reg = pd.read_csv(RESULTS / "mph1_model_comparison_regression.csv")
clf = pd.read_csv(RESULTS / "mph1_model_comparison_classification.csv")
reg = reg[reg["test_run_id"] == "ALL_MEAN"]
clf = clf[clf["test_run_id"] == "ALL_MEAN"]

# ---- 1. rich vs history-only (regression) ----
key = ["erlang_k", "rho", "horizon", "model"]
h = reg[reg.dataset == "history_only"][key + ["MAE", "MAE_std"]].rename(
    columns={"MAE": "MAE_history", "MAE_std": "MAE_std_history"})
r = reg[reg.dataset == "rich_features"][key + ["MAE", "MAE_std"]].rename(
    columns={"MAE": "MAE_rich", "MAE_std": "MAE_std_rich"})
cmp_ = h.merge(r, on=key)
cmp_["rich_improvement"] = 1 - cmp_["MAE_rich"] / cmp_["MAE_history"]
cmp_.to_csv(RESULTS / "mph1_feature_set_comparison.csv", index=False)

print("=" * 90)
print("RICH vs HISTORY-ONLY, regression MAE (positive = rich features better)")
print("=" * 90)
pd.set_option("display.width", 200)
rf = cmp_[cmp_.model == "Random Forest"].pivot(index="rho", columns="horizon",
                                                values="rich_improvement")
rf = rf[[c for c in ["short", "medium", "extra_long", "long"] if c in rf.columns]]
print("Random Forest:")
print((rf * 100).round(2).astype(str).add("%").to_string())
print("\nMean over rho and horizon, by model:")
print((cmp_.groupby("model")["rich_improvement"].mean() * 100).round(2).astype(str).add("%").to_string())

# ---- 2. classification precision / recall / F1 ----
print("\n" + "=" * 90)
print("CLASSIFICATION, medium horizon, history-only: Random Forest vs persistence")
print("=" * 90)
sel = clf[(clf.dataset == "history_only") & (clf.horizon == "medium") &
          (clf.model.isin(["Random Forest", "Naive (persistence)"]))]
for rho in sorted(sel.rho.unique()):
    print(f"rho={rho}  (congestion rate {sel[sel.rho == rho].congestion_rate.iloc[0]:.3f})")
    for _, x in sel[sel.rho == rho].iterrows():
        print(f"   {x.model:20s} precision={x.precision_congested:.3f}  "
              f"recall={x.recall_congested:.3f}  F1={x.f1_congested:.3f}")
print("\nSaved: results/mph1_feature_set_comparison.csv")