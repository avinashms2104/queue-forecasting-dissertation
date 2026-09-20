"""
Step 3 (M/G/1 version): Train baseline forecasting models on the M/G/1
datasets and evaluate them.

Mirrors modeling/train_baseline_models.py, adapted for the
(weibull_shape, rho) loop instead of just rho, reading from
data/mg1/processed/. Kept as a separate file for the same reason as
build_mg1_datasets.py -- see that file's docstring.

Answers the specific question from Azam's second email: how does moving
from M/M/1 to a general (non-exponential) service-time distribution
affect prediction accuracy, compared directly against the corrected
M/M/1 results in results/model_comparison_regression.csv /
model_comparison_classification.csv?

Usage
-----
    python train_mg1_baseline_models.py

Produces:
    results/mg1_model_comparison_regression.csv
    results/mg1_model_comparison_classification.csv
    results/plots/mg1_regression_mae_by_rho.png
    results/plots/mg1_classification_recall_by_rho.png
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mg1" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
WEIBULL_SHAPES = [2.0, 0.5]
HORIZONS = ["short", "medium", "long", "extra_long"]
RANDOM_SEED = 42

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
    else:
        exclude_prefixes = ("target_", "congestion_threshold")
        exclude_exact = {"weibull_shape", "rho", "run_id", "t", "split"}
        return [c for c in df.columns
                if c not in exclude_exact and not c.startswith(exclude_prefixes)]


def evaluate_regression(dataset_name, shape, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_shape{shape}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_qlen_{horizon}"

    train_df = df[df["split"] == "train"]
    X_train, y_train = train_df[feat_cols], train_df[target_col]

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

        naive_pred = X_test["queue_length_lag0"].values
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_rmse = mean_squared_error(y_test, naive_pred) ** 0.5
        rows.append({"dataset": dataset_name, "weibull_shape": shape, "rho": rho, "horizon": horizon,
                      "model": "Naive (persistence)", "test_run_id": test_run_id,
                      "MAE": naive_mae, "RMSE": naive_rmse, "n_test": len(y_test)})
        per_run_mae["Naive (persistence)"].append(naive_mae)

        for name, model in fitted_models.items():
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            rows.append({"dataset": dataset_name, "weibull_shape": shape, "rho": rho, "horizon": horizon,
                         "model": name, "test_run_id": test_run_id,
                         "MAE": mae, "RMSE": rmse, "n_test": len(y_test)})
            per_run_mae[name].append(mae)

    naive_mean_mae = np.mean(per_run_mae["Naive (persistence)"])
    for name in list(REG_MODEL_BUILDERS) + ["Naive (persistence)"]:
        maes = per_run_mae[name]
        mean_mae = np.mean(maes)
        rel_improvement = 1 - (mean_mae / naive_mean_mae) if naive_mean_mae > 0 else np.nan
        rows.append({"dataset": dataset_name, "weibull_shape": shape, "rho": rho, "horizon": horizon,
                     "model": name, "test_run_id": "ALL_MEAN",
                     "MAE": mean_mae, "MAE_std": np.std(maes),
                     "relative_improvement_over_persistence": rel_improvement,
                     "n_test_runs": len(maes)})

    return rows


def evaluate_classification(dataset_name, shape, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_shape{shape}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_congested_{horizon}"
    current_congested_col = "current_congested"

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
                "dataset": dataset_name, "weibull_shape": shape, "rho": rho, "horizon": horizon,
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

        score_and_record("Naive (always 'not congested')", np.zeros(len(y_test), dtype=int))
        score_and_record("Naive (persistence)", test_df[current_congested_col].values)

        for name, model in fitted_models.items():
            if name == "Logistic Regression":
                X_test_input = scaler.transform(X_test)
            else:
                X_test_input = X_test
            pred = model.predict(X_test_input)
            score_and_record(name, pred)

    for name in all_model_names:
        m = per_run_metrics[name]
        if len(m["accuracy"]) == 0:
            continue
        rows.append({
            "dataset": dataset_name, "weibull_shape": shape, "rho": rho, "horizon": horizon,
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
        for shape in WEIBULL_SHAPES:
            for rho in RHOS:
                for horizon in HORIZONS:
                    reg_rows += evaluate_regression(dataset_name, shape, rho, horizon)
                    clf_rows += evaluate_classification(dataset_name, shape, rho, horizon)
                    print(f"  done: {dataset_name} shape={shape} rho={rho} horizon={horizon}", flush=True)
            print(f"  finished shape={shape}", flush=True)
        print(f"Finished dataset: {dataset_name}", flush=True)

    reg_df = pd.DataFrame(reg_rows)
    clf_df = pd.DataFrame(clf_rows)
    reg_df.to_csv(RESULTS_DIR / "mg1_model_comparison_regression.csv", index=False)
    clf_df.to_csv(RESULTS_DIR / "mg1_model_comparison_classification.csv", index=False)

    print("\nSaved: results/mg1_model_comparison_regression.csv")
    print("Saved: results/mg1_model_comparison_classification.csv")

    make_plots(reg_df, clf_df)
    print_summary(reg_df, clf_df)

    return reg_df, clf_df


def _summary(df):
    return df[df["test_run_id"] == "ALL_MEAN"]


def make_plots(reg_df, clf_df):
    reg_summary = _summary(reg_df)
    clf_summary = _summary(clf_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    for shape, marker in [(2.0, "o-"), (0.5, "s--")]:
        sub = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                          (reg_summary.dataset == "history_only") & (reg_summary.weibull_shape == shape)]
        sub = sub.sort_values("rho")
        ax.plot(sub.rho, sub.MAE, marker, label=f"Weibull shape={shape}")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("Mean MAE across 5 test runs (queue length, medium horizon)")
    ax.set_title("M/G/1 regression accuracy vs. utilisation\n(Random Forest, history-only features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mg1_regression_mae_by_rho.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for shape, marker in [(2.0, "o-"), (0.5, "s--")]:
        sub = clf_summary[(clf_summary.model == "Random Forest") & (clf_summary.horizon == "medium") &
                          (clf_summary.dataset == "history_only") & (clf_summary.weibull_shape == shape)]
        sub = sub.sort_values("rho")
        ax.plot(sub.rho, sub.recall_congested, marker, label=f"Weibull shape={shape}")
    ax.set_xlabel("Traffic intensity (rho)")
    ax.set_ylabel("Mean recall on 'congested' class across 5 test runs (medium horizon)")
    ax.set_title("M/G/1 congestion detection vs. utilisation\n(Random Forest, history-only features)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mg1_classification_recall_by_rho.png", dpi=130)
    plt.close(fig)

    print("Saved 2 plots to results/plots/")


def print_summary(reg_df, clf_df):
    reg_summary = _summary(reg_df)
    clf_summary = _summary(clf_df)

    print("\n" + "=" * 100)
    print("SUMMARY (M/G/1): Random Forest, medium horizon, history-only features "
          "(mean +/- std across 5 test runs)")
    print("=" * 100)
    for shape in WEIBULL_SHAPES:
        for rho in RHOS:
            rf = reg_summary[(reg_summary.model == "Random Forest") & (reg_summary.horizon == "medium") &
                             (reg_summary.dataset == "history_only") & (reg_summary.weibull_shape == shape) &
                             (reg_summary.rho == rho)]
            naive = reg_summary[(reg_summary.model == "Naive (persistence)") & (reg_summary.horizon == "medium") &
                                (reg_summary.dataset == "history_only") & (reg_summary.weibull_shape == shape) &
                                (reg_summary.rho == rho)]
            if len(rf) == 0 or len(naive) == 0:
                continue
            rf_mae, rf_std = rf["MAE"].values[0], rf["MAE_std"].values[0]
            naive_mae = naive["MAE"].values[0]
            rel_imp = rf["relative_improvement_over_persistence"].values[0]
            print(f"shape={shape}, rho={rho}: naive MAE={naive_mae:.2f} | "
                  f"Random Forest MAE={rf_mae:.2f}+/-{rf_std:.2f} (rel.improvement={rel_imp:+.1%})")

    print("\n" + "=" * 100)
    print("SUMMARY (M/G/1): Congestion classification recall (Random Forest vs. persistence, "
          "medium horizon)")
    print("=" * 100)
    for shape in WEIBULL_SHAPES:
        for rho in RHOS:
            rf = clf_summary[(clf_summary.model == "Random Forest") & (clf_summary.horizon == "medium") &
                             (clf_summary.dataset == "history_only") & (clf_summary.weibull_shape == shape) &
                             (clf_summary.rho == rho)]
            persist = clf_summary[(clf_summary.model == "Naive (persistence)") & (clf_summary.horizon == "medium") &
                                  (clf_summary.dataset == "history_only") & (clf_summary.weibull_shape == shape) &
                                  (clf_summary.rho == rho)]
            if len(rf) == 0 or len(persist) == 0:
                continue
            rf_recall = rf["recall_congested"].values[0]
            rf_std = rf["recall_congested_std"].values[0]
            persist_recall = persist["recall_congested"].values[0]
            print(f"shape={shape}, rho={rho}: Random Forest recall={rf_recall:.2f}+/-{rf_std:.2f} | "
                  f"persistence-baseline recall={persist_recall:.2f}")


if __name__ == "__main__":
    run_all()