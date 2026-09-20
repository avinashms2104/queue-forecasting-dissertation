"""
Step 3 (M/PH/1 version): Train baseline forecasting models on the M/PH/1
(Erlang-k service) datasets and evaluate them.

Mirrors train_mg1_baseline_models.py exactly (same models, same settings,
same metrics, same persistence baselines, same run-level test evaluation),
with the (weibull_shape, rho) loop replaced by an (erlang_k, rho) loop and
data read from data/mph1/processed/, so M/PH/1 results are directly
comparable with the M/M/1 and M/G/1 results.

NOTE: get_feature_cols excludes "erlang_k" (not "weibull_shape") from the
rich feature set, so the constant k column is never used as a predictor.

Usage
-----
    python modeling/train_mph1_baseline_models.py                 # both feature sets (slow)
    python modeling/train_mph1_baseline_models.py --history-only  # faster first pass

Produces:
    results/mph1_model_comparison_regression.csv
    results/mph1_model_comparison_classification.csv
    results/plots/mph1_regression_mae_by_rho.png
    results/plots/mph1_classification_recall_by_rho.png
"""

import argparse
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mph1" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
ERLANG_KS = [2]
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
        exclude_exact = {"erlang_k", "rho", "run_id", "t", "split"}
        return [c for c in df.columns
                if c not in exclude_exact and not c.startswith(exclude_prefixes)]


def evaluate_regression(dataset_name, k, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_erlang{k}_rho{rho}.csv")
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
    base = {"dataset": dataset_name, "erlang_k": k, "rho": rho, "horizon": horizon}

    for test_run_id in TEST_RUN_IDS:
        test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
        if len(test_df) == 0:
            continue
        X_test, y_test = test_df[feat_cols], test_df[target_col]

        naive_pred = X_test["queue_length_lag0"].values
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_rmse = mean_squared_error(y_test, naive_pred) ** 0.5
        rows.append({**base, "model": "Naive (persistence)", "test_run_id": test_run_id,
                     "MAE": naive_mae, "RMSE": naive_rmse, "n_test": len(y_test)})
        per_run_mae["Naive (persistence)"].append(naive_mae)

        for name, model in fitted_models.items():
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            rows.append({**base, "model": name, "test_run_id": test_run_id,
                         "MAE": mae, "RMSE": rmse, "n_test": len(y_test)})
            per_run_mae[name].append(mae)

    naive_mean_mae = np.mean(per_run_mae["Naive (persistence)"])
    for name in list(REG_MODEL_BUILDERS) + ["Naive (persistence)"]:
        maes = per_run_mae[name]
        mean_mae = np.mean(maes)
        rel = 1 - (mean_mae / naive_mean_mae) if naive_mean_mae > 0 else np.nan
        rows.append({**base, "model": name, "test_run_id": "ALL_MEAN",
                     "MAE": mean_mae, "MAE_std": np.std(maes),
                     "relative_improvement_over_persistence": rel,
                     "n_test_runs": len(maes)})
    return rows


def evaluate_classification(dataset_name, k, rho, horizon):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}_erlang{k}_rho{rho}.csv")
    feat_cols = get_feature_cols(df, dataset_name)
    target_col = f"target_congested_{horizon}"

    df["current_congested"] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)

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
    per_run = {n: {"accuracy": [], "precision": [], "recall": [], "f1": [], "congestion_rate": []}
               for n in all_model_names}
    base = {"dataset": dataset_name, "erlang_k": k, "rho": rho, "horizon": horizon}
    rows = []

    for test_run_id in TEST_RUN_IDS:
        test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
        if len(test_df) == 0:
            continue
        X_test = test_df[feat_cols]
        y_test = test_df[target_col]
        congestion_rate = y_test.mean()

        def score_and_record(name, pred):
            row = {**base, "model": name, "test_run_id": test_run_id,
                   "accuracy": accuracy_score(y_test, pred),
                   "precision_congested": precision_score(y_test, pred, pos_label=1, zero_division=0),
                   "recall_congested": recall_score(y_test, pred, pos_label=1, zero_division=0),
                   "f1_congested": f1_score(y_test, pred, pos_label=1, zero_division=0),
                   "n_test": len(y_test), "congestion_rate": congestion_rate}
            rows.append(row)
            per_run[name]["accuracy"].append(row["accuracy"])
            per_run[name]["precision"].append(row["precision_congested"])
            per_run[name]["recall"].append(row["recall_congested"])
            per_run[name]["f1"].append(row["f1_congested"])
            per_run[name]["congestion_rate"].append(congestion_rate)

        score_and_record("Naive (always 'not congested')", np.zeros(len(y_test), dtype=int))
        score_and_record("Naive (persistence)", test_df["current_congested"].values)

        for name, model in fitted_models.items():
            X_in = scaler.transform(X_test) if name == "Logistic Regression" else X_test
            score_and_record(name, model.predict(X_in))

    for name in all_model_names:
        m = per_run[name]
        if len(m["accuracy"]) == 0:
            continue
        rows.append({**base, "model": name, "test_run_id": "ALL_MEAN",
                     "accuracy": np.mean(m["accuracy"]), "accuracy_std": np.std(m["accuracy"]),
                     "precision_congested": np.mean(m["precision"]), "precision_congested_std": np.std(m["precision"]),
                     "recall_congested": np.mean(m["recall"]), "recall_congested_std": np.std(m["recall"]),
                     "f1_congested": np.mean(m["f1"]), "f1_congested_std": np.std(m["f1"]),
                     "congestion_rate": np.mean(m["congestion_rate"]),
                     "n_test_runs": len(m["accuracy"])})
    return rows


def _summary(df):
    return df[df["test_run_id"] == "ALL_MEAN"]


def make_plots(reg_df, clf_df):
    reg_s, clf_s = _summary(reg_df), _summary(clf_df)
    for k in ERLANG_KS:
        sub = reg_s[(reg_s.model == "Random Forest") & (reg_s.horizon == "medium") &
                    (reg_s.dataset == "history_only") & (reg_s.erlang_k == k)].sort_values("rho")
        pers = reg_s[(reg_s.model == "Naive (persistence)") & (reg_s.horizon == "medium") &
                     (reg_s.dataset == "history_only") & (reg_s.erlang_k == k)].sort_values("rho")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub.rho, sub.MAE, "o-", label=f"Random Forest (Erlang-{k})")
        ax.plot(pers.rho, pers.MAE, "s--", label="Persistence")
        ax.set_xlabel("Traffic intensity (rho)")
        ax.set_ylabel("Mean MAE across 5 test runs (medium horizon)")
        ax.set_title("M/PH/1 regression accuracy vs. utilisation\n(history-only features)")
        ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "mph1_regression_mae_by_rho.png", dpi=130); plt.close(fig)

        sub = clf_s[(clf_s.model == "Random Forest") & (clf_s.horizon == "medium") &
                    (clf_s.dataset == "history_only") & (clf_s.erlang_k == k)].sort_values("rho")
        pers = clf_s[(clf_s.model == "Naive (persistence)") & (clf_s.horizon == "medium") &
                     (clf_s.dataset == "history_only") & (clf_s.erlang_k == k)].sort_values("rho")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub.rho, sub.recall_congested, "o-", label=f"Random Forest (Erlang-{k})")
        ax.plot(pers.rho, pers.recall_congested, "s--", label="Persistence")
        ax.set_xlabel("Traffic intensity (rho)")
        ax.set_ylabel("Mean recall on 'congested' class (medium horizon)")
        ax.set_title("M/PH/1 congestion detection vs. utilisation\n(history-only features)")
        ax.legend(); ax.set_ylim(0, 1.05); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "mph1_classification_recall_by_rho.png", dpi=130); plt.close(fig)
    print("Saved 2 plots to results/plots/")


def print_summary(reg_df, clf_df):
    reg_s, clf_s = _summary(reg_df), _summary(clf_df)
    print("\n" + "=" * 100)
    print("SUMMARY (M/PH/1): regression, Random Forest vs persistence, medium horizon, history-only")
    print("=" * 100)
    for k in ERLANG_KS:
        for rho in RHOS:
            sel = (reg_s.horizon == "medium") & (reg_s.dataset == "history_only") & \
                  (reg_s.erlang_k == k) & (reg_s.rho == rho)
            rf = reg_s[sel & (reg_s.model == "Random Forest")]
            nv = reg_s[sel & (reg_s.model == "Naive (persistence)")]
            if len(rf) == 0 or len(nv) == 0:
                continue
            print(f"erlang{k}, rho={rho}: naive MAE={nv['MAE'].values[0]:.2f} | "
                  f"Random Forest MAE={rf['MAE'].values[0]:.2f}+/-{rf['MAE_std'].values[0]:.2f} "
                  f"(rel.improvement={rf['relative_improvement_over_persistence'].values[0]:+.1%})")

    print("\n" + "=" * 100)
    print("SUMMARY (M/PH/1): congestion recall, Random Forest vs persistence, medium horizon")
    print("=" * 100)
    for k in ERLANG_KS:
        for rho in RHOS:
            sel = (clf_s.horizon == "medium") & (clf_s.dataset == "history_only") & \
                  (clf_s.erlang_k == k) & (clf_s.rho == rho)
            rf = clf_s[sel & (clf_s.model == "Random Forest")]
            ps = clf_s[sel & (clf_s.model == "Naive (persistence)")]
            if len(rf) == 0 or len(ps) == 0:
                continue
            print(f"erlang{k}, rho={rho}: Random Forest recall={rf['recall_congested'].values[0]:.2f}"
                  f"+/-{rf['recall_congested_std'].values[0]:.2f} | "
                  f"persistence recall={ps['recall_congested'].values[0]:.2f}")


def run_all(history_only=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["history_only"] if history_only else ["history_only", "rich_features"]
    reg_rows, clf_rows = [], []

    for dataset_name in datasets:
        for k in ERLANG_KS:
            for rho in RHOS:
                for horizon in HORIZONS:
                    reg_rows += evaluate_regression(dataset_name, k, rho, horizon)
                    clf_rows += evaluate_classification(dataset_name, k, rho, horizon)
                    print(f"  done: {dataset_name} erlang{k} rho={rho} horizon={horizon}", flush=True)
        print(f"Finished dataset: {dataset_name}", flush=True)

    reg_df, clf_df = pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)
    reg_df.to_csv(RESULTS_DIR / "mph1_model_comparison_regression.csv", index=False)
    clf_df.to_csv(RESULTS_DIR / "mph1_model_comparison_classification.csv", index=False)
    print("\nSaved: results/mph1_model_comparison_regression.csv")
    print("Saved: results/mph1_model_comparison_classification.csv")

    make_plots(reg_df, clf_df)
    print_summary(reg_df, clf_df)
    return reg_df, clf_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-only", action="store_true",
                    help="skip the rich feature set (much faster)")
    args = ap.parse_args()
    run_all(history_only=args.history_only)