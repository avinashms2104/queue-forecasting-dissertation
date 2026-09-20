"""
Rich vs history-only feature sets for the M/G/1 results (reads the CSV already
produced by train_mg1_baseline_models.py; trains nothing, runs in seconds).

    difference = 1 - MAE_rich / MAE_history     (positive = rich features better)

Usage:  python modeling/mg1_extra_summaries.py
Writes: results/mg1_feature_set_comparison.csv
"""
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
reg = pd.read_csv(RESULTS / "mg1_model_comparison_regression.csv")
reg = reg[reg["test_run_id"].astype(str) == "ALL_MEAN"]

key = ["weibull_shape", "rho", "horizon", "model"]
h = reg[reg.dataset == "history_only"][key + ["MAE"]].rename(columns={"MAE": "MAE_history"})
r = reg[reg.dataset == "rich_features"][key + ["MAE"]].rename(columns={"MAE": "MAE_rich"})
cmp_ = h.merge(r, on=key)
cmp_["rich_improvement"] = 1 - cmp_["MAE_rich"] / cmp_["MAE_history"]
cmp_.to_csv(RESULTS / "mg1_feature_set_comparison.csv", index=False)

pd.set_option("display.width", 200)
print("=" * 90)
print("M/G/1: RICH vs HISTORY-ONLY, regression MAE (positive = rich features better)")
print("=" * 90)
for shape in sorted(cmp_.weibull_shape.unique(), reverse=True):
    sub = cmp_[(cmp_.weibull_shape == shape) & (cmp_.model == "Random Forest")]
    pv = sub.pivot(index="rho", columns="horizon", values="rich_improvement")
    pv = pv[[c for c in ["short", "medium", "extra_long", "long"] if c in pv.columns]]
    print(f"\nWeibull shape {shape}, Random Forest:")
    print((pv * 100).round(2).astype(str).add("%").to_string())
print("\nMean over rho and horizon, by shape and model:")
print((cmp_.groupby(["weibull_shape", "model"])["rich_improvement"].mean() * 100)
      .round(2).astype(str).add("%").to_string())
print("\nSaved: results/mg1_feature_set_comparison.csv")