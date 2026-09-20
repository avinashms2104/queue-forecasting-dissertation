"""
Side-by-side comparison of the service-time distributions run so far,
ordered by service-time variance (squared coefficient of variation C_s^2):

    Weibull shape=2.0   C_s^2 ~ 0.27   (M/G/1)
    Erlang-2            C_s^2 = 0.5    (M/PH/1)
    Exponential         C_s^2 = 1.0    (M/M/1)
    Weibull shape=0.5   C_s^2 = 5.0    (M/G/1)

Reads the existing results CSVs (medium horizon, history-only features,
Random Forest, mean over the 5 test runs). Trains nothing.

Usage:  python modeling/compare_service_distributions.py
Writes: results/service_variance_comparison.csv
        results/plots/service_variance_comparison.png
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load(name):
    df = pd.read_csv(RESULTS / name)
    df = df[df["test_run_id"].astype(str) == "ALL_MEAN"]
    if "dataset" in df.columns:
        df = df[df["dataset"] == "history_only"]
    return df[df["horizon"] == "medium"]


reg_mm1 = load("model_comparison_regression.csv")
reg_mg1 = load("mg1_model_comparison_regression.csv")
reg_mph1 = load("mph1_model_comparison_regression.csv")
clf_mm1 = load("model_comparison_classification.csv")
clf_mg1 = load("mg1_model_comparison_classification.csv")
clf_mph1 = load("mph1_model_comparison_classification.csv")

# (label, C_s^2, row filter, regression rows, classification rows)
SPEC = [
    ("Weibull 2.0 (C_s^2=0.27)", 0.27,
     lambda d: d[d.weibull_shape == 2.0], reg_mg1, clf_mg1),
    ("Erlang-2 (C_s^2=0.5)", 0.5,
     lambda d: d[d.erlang_k == 2], reg_mph1, clf_mph1),
    ("M/M/1 (C_s^2=1.0)", 1.0,
     lambda d: d, reg_mm1, clf_mm1),
    ("Weibull 0.5 (C_s^2=5.0)", 5.0,
     lambda d: d[d.weibull_shape == 0.5], reg_mg1, clf_mg1),
]

reg_table, f1_table = {}, {}
for label, cs2, filt, reg, clf in SPEC:
    r = filt(reg)
    r = r[r.model == "Random Forest"].set_index("rho")
    reg_table[label] = r["relative_improvement_over_persistence"] * 100

    c = filt(clf)
    rf = c[c.model == "Random Forest"].set_index("rho")["f1_congested"]
    ps = c[c.model == "Naive (persistence)"].set_index("rho")["f1_congested"]
    f1_table[label] = rf - ps

reg_df = pd.DataFrame(reg_table).sort_index()
f1_df = pd.DataFrame(f1_table).sort_index()

pd.set_option("display.width", 200)
print("=" * 100)
print("REGRESSION: Random Forest relative improvement over persistence (%), medium horizon")
print("(positive = Random Forest better; columns ordered by service-time variance)")
print("=" * 100)
print(reg_df.round(1).to_string())
print("\n" + "=" * 100)
print("CLASSIFICATION: F1(Random Forest) - F1(persistence), medium horizon")
print("(negative = persistence better)")
print("=" * 100)
print(f1_df.round(3).to_string())

out = pd.concat({"regression_rel_improvement_pct": reg_df, "f1_rf_minus_persistence": f1_df}, axis=1)
out.to_csv(RESULTS / "service_variance_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for label in reg_df.columns:
    axes[0].plot(reg_df.index, reg_df[label], "o-", label=label)
    axes[1].plot(f1_df.index, f1_df[label], "o-", label=label)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Regression: Random Forest gain over persistence (%)")
axes[1].set_title("Classification: F1(RF) - F1(persistence)")
for ax in axes:
    ax.set_xlabel("Traffic intensity (rho)")
axes[0].set_ylabel("Relative improvement (%)")
axes[1].set_ylabel("F1 difference")
axes[0].legend(fontsize=8)
fig.suptitle("Effect of service-time variance (medium horizon, history-only features)")
fig.tight_layout()
fig.savefig(PLOTS / "service_variance_comparison.png", dpi=130)
print("\nSaved: results/service_variance_comparison.csv")
print("Saved: results/plots/service_variance_comparison.png")