"""
Step 3: Train baseline forecasting models and evaluate them.

Updated per Azam's feedback (2026-09-01 email):
  - use the run-level train/val/test split from build_datasets.py (column
    "split") instead of a random row-level split;
  - add a persistence baseline for classification ("congestion state stays
    the same as now"), in addition to the majority-class baseline;
  - report RMSE alongside MAE, and relative improvement over persistence
    (1 - MAE_model / MAE_naive);
  - add h=15, plus a wider horizon sweep at rho=0.8;
  - evaluate each of the 5 independent test runs SEPARATELY and report
    mean +/- std across them, not one pooled number;
  - standardise features before fitting Logistic Regression.

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

Usage
-----
    python train_baseline_models.py

Produces:
    results/model_comparison_regression.csv
        one row per (dataset, rho, horizon, model, test_run_id) with MAE/RMSE,
        PLUS summary rows (test_run_id="ALL_MEAN") with mean/std across the
        5 test runs and relative improvement over persistence.
    results/model_comparison_classification.csv
        same structure, with accuracy/precision/recall/F1 on the congested class.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              precision_score, recall_score, f1_score, accuracy_score)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
HORIZONS = ["short", "medium", "long", "extra_long"]  # extra_long = h=15, see build_datasets.py
RANDOM_SEED = 42

# Test runs are run_id 25-29 (5 independent runs) per build_datasets.py's
# assign_split(). We evaluate each one separately for uncertainty estimates.
TEST_RUN_IDS = [25, 26, 27, 28, 29]

REG_MODEL_BUILDERS = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=RANDOM_SEED),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=30, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1),
}

CLF_MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
}


def get_feature_cols(df, dataset_name):
    if dataset_name == "history_only":
        return [c for c in df.columns if c.startswith("queue_length_lag")]
    else:  # rich_features
        exclude_prefixes = ("target_", "congestion_threshold")
        exclude_exact = {"rho", "run_id", "t", "split"}
        return [c for c in df.columns
                if c not in exclude_exact and not c.startswith(exclude_prefixes)]


def evaluate_regression(dataset_name, rho, horizon):
    """
    Train once on the full train split, then evaluate SEPARATELY on each of
    the 5 test runs.

    Returns a list of per-test-run rows, plus one summary row
    (test_run_id="ALL_MEAN") with mean/std and relative improvement over
    the naive persistence baseline.
    """
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_qlen_{horizon}"

    train_df = df[df["split"] == "train"]
    X_train, y_train = train_df[feat_cols], train_df[target_col]

    # fit each model once on the full training split
    fitted_models = {}
    for name, builder in REG_MODEL_BUILDERS.items():
        model = builder()
        model.fit(X_train, y_train)
        fitted_models[name] = model

    rows = []
    per_run_mae = {name: [] for name in list(REG_MODEL_BUILDERS) + ["Naive (persistence)"]}

    for test_run_id in TEST_RUN_IDS:
        test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
        if len(test_df) == 0:
            continue
        X_test, y_test = test_df[feat_cols], test_df[target_col]

        # naive persistence baseline: predict Q_{t+h} = Q_t
        naive_pred = X_test["queue_length_lag0"].values
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_rmse = mean_squared_error(y_test, naive_pred) ** 0.5
        rows.append({"dataset": dataset_name, "rho": rho, "horizon": horizon,
                      "model": "Naive (persistence)", "test_run_id": test_run_id,
                      "MAE": naive_mae, "RMSE": naive_rmse, "n_test": len(y_test)})
        per_run_mae["Naive (persistence)"].append(naive_mae)

        for name, model in fitted_models.items():
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            rows.append({"dataset": dataset_name, "rho": rho, "horizon": horizon,
                         "model": name, "test_run_id": test_run_id,
                         "MAE": mae, "RMSE": rmse, "n_test": len(y_test)})
            per_run_mae[name].append(mae)

    # summary rows: mean +/- std across the 5 test runs, plus relative
    # improvement over the naive baseline's mean MAE
    naive_mean_mae = np.mean(per_run_mae["Naive (persistence)"])
    for name in list(REG_MODEL_BUILDERS) + ["Naive (persistence)"]:
        maes = per_run_mae[name]
        mean_mae = np.mean(maes)
        rel_improvement = 1 - (mean_mae / naive_mean_mae) if naive_mean_mae > 0 else np.nan
        rows.append({"dataset": dataset_name, "rho": rho, "horizon": horizon,
                     "model": name, "test_run_id": "ALL_MEAN",
                     "MAE": mean_mae, "MAE_std": np.std(maes),
                     "relative_improvement_over_persistence": rel_improvement,
                     "n_test_runs": len(maes)})

    return rows


def evaluate_classification(dataset_name, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_congested_{horizon}"
    current_congested_col = "current_congested"  # constructed below for the persistence baseline

    # "currently congested" = is queue_length_lag0 (i.e. Q_t) above the
    # per-rho congestion threshold? Needed for the persistence baseline
    # ("assume congestion state stays the same to the forecast horizon").
    df[current_congested_col] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)

    train_df = df[df["split"] == "train"]
    X_train, y_train = train_df[feat_cols], train_df[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    fitted_models = {}
    for name, builder in CLF_MODEL_BUILDERS.items():
        model = builder()
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        fitted_models[name] = model

    all_model_names = ["Naive (always 'not congested')", "Naive (persistence)"] + list(CLF_MODEL_BUILDERS)
    per_run_metrics = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "congestion_rate": []}
                       for name in all_model_names}

    rows = []
    for test_run_id in TEST_RUN_IDS:
        test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
        if len(test_df) == 0:
            continue
        X_test = test_df[feat_cols]
        y_test = test_df[target_col]
        congestion_rate = y_test.mean()

        def score_and_record(name, pred):
            row = {
                "dataset": dataset_name, "rho": rho, "horizon": horizon,
                "model": name, "test_run_id": test_run_id,
                "accuracy": accuracy_score(y_test, pred),
                "precision_congested": precision_score(y_test, pred, pos_label=1, zero_division=0),
                "recall_congested": recall_score(y_test, pred, pos_label=1, zero_division=0),
                "f1_congested": f1_score(y_test, pred, pos_label=1, zero_division=0),
                "n_test": len(y_test), "congestion_rate": congestion_rate,
            }
            rows.append(row)
            per_run_metrics[name]["accuracy"].append(row["accuracy"])
            per_run_metrics[name]["precision"].append(row["precision_congested"])
            per_run_metrics[name]["recall"].append(row["recall_congested"])
            per_run_metrics[name]["f1"].append(row["f1_congested"])
            per_run_metrics[name]["congestion_rate"].append(congestion_rate)

        # baseline 1: always predict "not congested"
        score_and_record("Naive (always 'not congested')", np.zeros(len(y_test), dtype=int))

        # baseline 2: persistence -- assume congestion state stays the same
        # as it is right now (Q_t vs threshold) through to the horizon
        score_and_record("Naive (persistence)", test_df[current_congested_col].values)

        for name, model in fitted_models.items():
            if name == "Logistic Regression":
                X_test_input = scaler.transform(X_test)
            else:
                X_test_input = X_test
            pred = model.predict(X_test_input)
            score_and_record(name, pred)

    # summary rows: mean +/- std across the 5 test runs
    for name in all_model_names:
        m = per_run_metrics[name]
        if len(m["accuracy"]) == 0:
            continue
        rows.append({
            "dataset": dataset_name, "rho": rho, "horizon": horizon,
            "model": name, "test_run_id": "ALL_MEAN",
            "accuracy": np.mean(m["accuracy"]), "accuracy_std": np.std(m["accuracy"]),
            "precision_congested": np.mean(m["precision"]), "precision_congested_std": np.std(m["precision"]),
            "recall_congested": np.mean(m["recall"]), "recall_congested_std": np.std(m["recall"]),
            "f1_congested": np.mean(m["f1"]), "f1_congested_std": np.std(m["f1"]),
            "congestion_rate": np.mean(m["congestion_rate"]),
            "n_test_runs": len(m["accuracy"]),
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


def _summary(df):
    return df[df["test_run_id"] == "ALL_MEAN"]


def make_plots(reg_df, clf_df):
    reg_summary = _summary(reg_df)
    clf_summary = _summary(clf_df)

    # --- Plot 1: regression MAE vs rho, medium horizon, Random Forest, both datasets ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                          (reg_summary.dataset == dataset_name)].sort_values("rho")
        ax.plot(sub.rho, sub.MAE, marker, label=f"Random Forest ({dataset_name})")
    naive = reg_summary[(reg_summary.model == "Naive (persistence)") & (reg_summary.horizon == "medium") &
                        (reg_summary.dataset == "history_only")].sort_values("rho")
    ax.plot(naive.rho, naive.MAE, "k:", label="Naive (persistence)")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("Mean MAE across 5 test runs (queue length, medium horizon)")
    ax.set_title("Regression accuracy vs. utilisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_mae_by_rho.png", dpi=130)
    plt.close(fig)

    # --- Plot 2: regression MAE vs horizon, rho=0.8, Random Forest, both datasets ---
    fig, ax = plt.subplots(figsize=(7, 5))
    horizon_order = ["short", "medium", "long", "extra_long"]
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.rho == 0.8) &
                          (reg_summary.dataset == dataset_name)]
        sub = sub.set_index("horizon").reindex(horizon_order).reset_index()
        ax.plot(sub.horizon, sub.MAE, marker, label=f"Random Forest ({dataset_name})")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Mean MAE across 5 test runs (rho=0.8)")
    ax.set_title("Regression accuracy vs. forecast horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_mae_by_horizon.png", dpi=130)
    plt.close(fig)

    # --- Plot 3: classification recall (congested class) vs rho, medium horizon ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset_name, marker in [("history_only", "o-"), ("rich_features", "s--")]:
        sub = clf_summary[(clf_summary.model == "Random Forest") & (clf_summary.horizon == "medium") &
                          (clf_summary.dataset == dataset_name)].sort_values("rho")
        ax.plot(sub.rho, sub.recall_congested, marker, label=f"Random Forest ({dataset_name})")
    persist = clf_summary[(clf_summary.model == "Naive (persistence)") & (clf_summary.horizon == "medium") &
                          (clf_summary.dataset == "history_only")].sort_values("rho")
    ax.plot(persist.rho, persist.recall_congested, "k:", label="Naive (persistence)")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("Mean recall on 'congested' class across 5 test runs (medium horizon)")
    ax.set_title("Congestion detection: recall vs. utilisation\n(can the model actually catch congested periods?)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classification_recall_by_rho.png", dpi=130)
    plt.close(fig)

    # --- Plot 4: dataset A vs B summary -- MAE improvement (%) per rho, medium horizon ---
    fig, ax = plt.subplots(figsize=(7, 5))
    a = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                    (reg_summary.dataset == "history_only")].set_index("rho")["MAE"]
    b = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                    (reg_summary.dataset == "rich_features")].set_index("rho")["MAE"]
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
    reg_summary = _summary(reg_df)
    clf_summary = _summary(clf_df)

    print("\n" + "=" * 90)
    print("SUMMARY: Random Forest, medium horizon, history_only vs rich_features "
          "(mean +/- std across 5 test runs)")
    print("=" * 90)
    for rho in RHOS:
        a = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                        (reg_summary.dataset == "history_only") & (reg_summary.rho == rho)]
        b = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                        (reg_summary.dataset == "rich_features") & (reg_summary.rho == rho)]
        naive = reg_summary[(reg_summary.model == "Naive (persistence)") & (reg_summary.horizon == "medium") &
                            (reg_summary.dataset == "history_only") & (reg_summary.rho == rho)]
        a_mae, a_std = a["MAE"].values[0], a["MAE_std"].values[0]
        b_mae, b_std = b["MAE"].values[0], b["MAE_std"].values[0]
        naive_mae = naive["MAE"].values[0]
        rel_imp_a = a["relative_improvement_over_persistence"].values[0]
        rel_imp_b = b["relative_improvement_over_persistence"].values[0]
        pct_ab = 100 * (a_mae - b_mae) / a_mae
        print(f"rho={rho}: naive MAE={naive_mae:.2f} | "
              f"history-only MAE={a_mae:.2f}+/-{a_std:.2f} (rel.improvement={rel_imp_a:+.1%}) | "
              f"rich-features MAE={b_mae:.2f}+/-{b_std:.2f} (rel.improvement={rel_imp_b:+.1%}) | "
              f"rich vs history-only: {pct_ab:+.1f}%")

    print("\n" + "=" * 90)
    print("SUMMARY: Congestion classification recall (Random Forest vs. persistence baseline, "
          "medium horizon)")
    print("=" * 90)
    for rho in RHOS:
        rf = clf_summary[(clf_summary.model == "Random Forest") & (clf_summary.horizon == "medium") &
                         (clf_summary.dataset == "history_only") & (clf_summary.rho == rho)]
        persist = clf_summary[(clf_summary.model == "Naive (persistence)") & (clf_summary.horizon == "medium") &
                              (clf_summary.dataset == "history_only") & (clf_summary.rho == rho)]
        rf_recall = rf["recall_congested"].values[0]
        rf_std = rf["recall_congested_std"].values[0]
        persist_recall = persist["recall_congested"].values[0]
        print(f"rho={rho}: Random Forest recall={rf_recall:.2f}+/-{rf_std:.2f} | "
              f"persistence-baseline recall={persist_recall:.2f}")


if __name__ == "__main__":
    run_all()