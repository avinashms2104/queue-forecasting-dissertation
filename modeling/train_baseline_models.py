"""
Step 3: Train baseline forecasting models and evaluate them.

This is the step that answers Azam's core questions:
    1. Do the additional queueing features (Dataset B) meaningfully improve
       on queue-length history alone (Dataset A)?
    2. How does prediction quality change as utilisation (rho) increases?
    3. How does prediction quality change as the forecast horizon lengthens?
    4. Can a model correctly identify congested periods (not just overall
       accuracy -- precision/recall on the congested class specifically)?

Per Azam's instructions, we train SIMPLE models first (no neural nets):
    - Linear Regression                  (regression baseline)
    - Decision Tree Regressor            (regression)
    - Random Forest Regressor            (regression, usually the strongest of these three)
    - Logistic Regression                (classification baseline)
    - Random Forest Classifier           (classification)

And, following the lesson from Velarde et al. (Paper 4, lit review): every
model is compared against a NAIVE baseline (predict that the queue length
simply stays the same as now) -- an "improvement" that doesn't beat naive
persistence is not a real improvement.

Usage
-----
    python train_baseline_models.py

Produces:
    results/model_comparison_regression.csv   (MAE/RMSE per model/dataset/rho/horizon)
    results/model_comparison_classification.csv (precision/recall/F1 per model/dataset/rho/horizon)
    results/plots/regression_mae_by_rho.png
    results/plots/regression_mae_by_horizon.png
    results/plots/classification_recall_by_rho.png
    results/plots/dataset_comparison_summary.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              precision_score, recall_score, f1_score, accuracy_score)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
HORIZONS = ["short", "medium", "long"]
RANDOM_SEED = 42
TEST_SIZE = 0.2

REG_MODELS = {
    "Naive (persistence)": None,  # handled specially -- predicts queue_length_lag0
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_SEED),
    "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1),
}

CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
}


def get_feature_cols(df, dataset_name):
    if dataset_name == "history_only":
        return [c for c in df.columns if c.startswith("queue_length_lag")]
    else:  # rich_features
        exclude_prefixes = ("target_", "congestion_threshold")
        exclude_exact = {"rho", "run_id", "t"}
        return [c for c in df.columns
                if c not in exclude_exact and not c.startswith(exclude_prefixes)]


def evaluate_regression(dataset_name, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_qlen_{horizon}"

    X, y = df[feat_cols], df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    rows = []
    for name, model in REG_MODELS.items():
        if name == "Naive (persistence)":
            pred = X_test["queue_length_lag0"].values
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        rows.append({
            "dataset": dataset_name, "rho": rho, "horizon": horizon,
            "model": name, "MAE": mae, "RMSE": rmse, "n_test": len(y_test),
        })
    return rows


def evaluate_classification(dataset_name, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_congested_{horizon}"

    X, y = df[feat_cols], df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    rows = []
    # naive classifier: always predict "not congested" (the majority class)
    naive_pred = np.zeros_like(y_test)
    rows.append({
        "dataset": dataset_name, "rho": rho, "horizon": horizon, "model": "Naive (always 'not congested')",
        "accuracy": accuracy_score(y_test, naive_pred),
        "precision_congested": precision_score(y_test, naive_pred, pos_label=1, zero_division=0),
        "recall_congested": recall_score(y_test, naive_pred, pos_label=1, zero_division=0),
        "f1_congested": f1_score(y_test, naive_pred, pos_label=1, zero_division=0),
        "n_test": len(y_test), "congestion_rate": y_test.mean(),
    })

    for name, model in CLF_MODELS.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "dataset": dataset_name, "rho": rho, "horizon": horizon, "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_congested": precision_score(y_test, pred, pos_label=1, zero_division=0),
            "recall_congested": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "f1_congested": f1_score(y_test, pred, pos_label=1, zero_division=0),
            "n_test": len(y_test), "congestion_rate": y_test.mean(),
        })
    return rows


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    reg_rows, clf_rows = [], []

    for dataset_name in ["history_only", "rich_features"]:
        for rho in RHOS:
            for horizon in HORIZONS:
                reg_rows += evaluate_regression(dataset_name, rho, horizon)
                clf_rows += evaluate_classification(dataset_name, rho, horizon)
                print(f"  done: {dataset_name} rho={rho} horizon={horizon}", flush=True)
        print(f"Finished dataset: {dataset_name}", flush=True)

    reg_df = pd.DataFrame(reg_rows)
    clf_df = pd.DataFrame(clf_rows)
    reg_df.to_csv(RESULTS_DIR / "model_comparison_regression.csv", index=False)
    clf_df.to_csv(RESULTS_DIR / "model_comparison_classification.csv", index=False)

    print("\nSaved: results/model_comparison_regression.csv")
    print("Saved: results/model_comparison_classification.csv")

    make_plots(reg_df, clf_df)
    print_summary(reg_df, clf_df)

    return reg_df, clf_df


def make_plots(reg_df, clf_df):
    # --- Plot 1: regression MAE vs rho, medium horizon, Random Forest, both datasets ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = reg_df[(reg_df.model == "Random Forest") & (reg_df.horizon == "medium") & (reg_df.dataset == dataset_name)]
        sub = sub.sort_values("rho")
        ax.plot(sub.rho, sub.MAE, marker, label=f"Random Forest ({dataset_name})")
    naive = reg_df[(reg_df.model == "Naive (persistence)") & (reg_df.horizon == "medium") & (reg_df.dataset == "history_only")]
    naive = naive.sort_values("rho")
    ax.plot(naive.rho, naive.MAE, "k:", label="Naive (persistence)")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("MAE (queue length, medium horizon)")
    ax.set_title("Regression accuracy vs. utilisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_mae_by_rho.png", dpi=130)
    plt.close(fig)

    # --- Plot 2: regression MAE vs horizon, rho=0.8, Random Forest, both datasets ---
    fig, ax = plt.subplots(figsize=(7, 5))
    horizon_order = ["short", "medium", "long"]
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = reg_df[(reg_df.model == "Random Forest") & (reg_df.rho == 0.8) & (reg_df.dataset == dataset_name)]
        sub = sub.set_index("horizon").loc[horizon_order].reset_index()
        ax.plot(sub.horizon, sub.MAE, marker, label=f"Random Forest ({dataset_name})")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE (queue length, rho=0.8)")
    ax.set_title("Regression accuracy vs. forecast horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_mae_by_horizon.png", dpi=130)
    plt.close(fig)

    # --- Plot 3: classification recall (congested class) vs rho, medium horizon ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = clf_df[(clf_df.model == "Random Forest") & (clf_df.horizon == "medium") & (clf_df.dataset == dataset_name)]
        sub = sub.sort_values("rho")
        ax.plot(sub.rho, sub.recall_congested, marker, label=f"Random Forest ({dataset_name})")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("Recall on 'congested' class (medium horizon)")
    ax.set_title("Congestion detection: recall vs. utilisation\n(can the model actually catch congested periods?)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classification_recall_by_rho.png", dpi=130)
    plt.close(fig)

    # --- Plot 4: dataset A vs B summary -- MAE improvement (%) per rho, medium horizon ---
    fig, ax = plt.subplots(figsize=(7, 5))
    a = reg_df[(reg_df.model == "Random Forest") & (reg_df.horizon == "medium") & (reg_df.dataset == "history_only")].set_index("rho")["MAE"]
    b = reg_df[(reg_df.model == "Random Forest") & (reg_df.horizon == "medium") & (reg_df.dataset == "rich_features")].set_index("rho")["MAE"]
    improvement_pct = 100 * (a - b) / a
    ax.bar([str(r) for r in improvement_pct.index], improvement_pct.values, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("MAE improvement of rich features over history-only (%)")
    ax.set_title("Does the richer feature set actually help?\n(Random Forest, medium horizon)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dataset_comparison_summary.png", dpi=130)
    plt.close(fig)

    print("Saved 4 plots to results/plots/")


def print_summary(reg_df, clf_df):
    print("\n" + "=" * 70)
    print("SUMMARY: Random Forest, medium horizon, history_only vs rich_features")
    print("=" * 70)
    for rho in RHOS:
        a = reg_df[(reg_df.model == "Random Forest") & (reg_df.horizon == "medium") &
                    (reg_df.dataset == "history_only") & (reg_df.rho == rho)]["MAE"].values[0]
        b = reg_df[(reg_df.model == "Random Forest") & (reg_df.horizon == "medium") &
                    (reg_df.dataset == "rich_features") & (reg_df.rho == rho)]["MAE"].values[0]
        naive = reg_df[(reg_df.model == "Naive (persistence)") & (reg_df.horizon == "medium") &
                        (reg_df.dataset == "history_only") & (reg_df.rho == rho)]["MAE"].values[0]
        pct = 100 * (a - b) / a
        print(f"rho={rho}: naive MAE={naive:.2f} | history-only MAE={a:.2f} | "
              f"rich-features MAE={b:.2f} | rich-features improvement over history-only: {pct:+.1f}%")

    print("\n" + "=" * 70)
    print("SUMMARY: Congestion classification recall (Random Forest, medium horizon)")
    print("=" * 70)
    for rho in RHOS:
        a = clf_df[(clf_df.model == "Random Forest") & (clf_df.horizon == "medium") &
                    (clf_df.dataset == "history_only") & (clf_df.rho == rho)]["recall_congested"].values[0]
        b = clf_df[(clf_df.model == "Random Forest") & (clf_df.horizon == "medium") &
                    (clf_df.dataset == "rich_features") & (clf_df.rho == rho)]["recall_congested"].values[0]
        print(f"rho={rho}: history-only recall={a:.2f} | rich-features recall={b:.2f}")


if __name__ == "__main__":
    run_all()
