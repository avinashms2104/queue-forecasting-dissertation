"""
Class-weighting and threshold-tuning for congestion classification.

Per Azam's instruction, this was deliberately deferred until after the
persistence/autocorrelation investigation was complete (see
autocorrelation_analysis.py, mg1_autocorrelation_analysis.py, and the
persistence-baseline results in train_baseline_models.py) -- now that
that's done for both M/M/1 and M/G/1, this tests two concrete fixes for
the near-zero recall problem at low rho:

  1. class_weight='balanced' -- reweight the loss so the minority
     (congested) class isn't effectively ignored during training.
  2. Threshold tuning -- instead of the default 0.5 probability cutoff
     for predicting "congested", sweep thresholds and report the one
     that maximises F1, showing the precision/recall trade-off directly.

Uses the M/M/1 history-only datasets already built (data/processed/), at
the medium horizon, across all six rho.

Usage
-----
    python class_weight_threshold_tuning.py

Produces:
    results/class_weight_threshold_tuning.csv
    results/plots/threshold_tuning_low_rho.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
RANDOM_SEED = 42
TEST_RUN_IDS = [25, 26, 27, 28, 29]
THRESHOLDS = np.arange(0.05, 0.95, 0.05)


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("queue_length_lag")]


def run_for_rho(rho):
    df = pd.read_csv(DATA_DIR / f"history_only_rho{rho}.csv")
    feat_cols = get_feature_cols(df)
    target_col = "target_congested_medium"

    train_df = df[df["split"] == "train"]
    X_train, y_train = train_df[feat_cols], train_df[target_col]

    model_default = RandomForestClassifier(n_estimators=50, max_depth=8,
                                            random_state=RANDOM_SEED, n_jobs=-1)
    model_default.fit(X_train, y_train)

    model_balanced = RandomForestClassifier(n_estimators=50, max_depth=8,
                                             random_state=RANDOM_SEED, n_jobs=-1,
                                             class_weight="balanced")
    model_balanced.fit(X_train, y_train)

    # gather predicted probabilities across all 5 test runs for both models
    all_y_test, all_proba_default, all_proba_balanced = [], [], []
    for test_run_id in TEST_RUN_IDS:
        test_df = df[(df["split"] == "test") & (df["run_id"] == test_run_id)]
        if len(test_df) == 0:
            continue
        X_test, y_test = test_df[feat_cols], test_df[target_col]
        all_y_test.append(y_test.values)
        all_proba_default.append(model_default.predict_proba(X_test)[:, 1])
        all_proba_balanced.append(model_balanced.predict_proba(X_test)[:, 1])

    y_test_all = np.concatenate(all_y_test)
    proba_default_all = np.concatenate(all_proba_default)
    proba_balanced_all = np.concatenate(all_proba_balanced)

    def eval_at_threshold(y_true, proba, thresh):
        pred = (proba >= thresh).astype(int)
        return (precision_score(y_true, pred, zero_division=0),
                recall_score(y_true, pred, zero_division=0),
                f1_score(y_true, pred, zero_division=0))

    # default threshold (0.5) for both models
    p_def, r_def, f_def = eval_at_threshold(y_test_all, proba_default_all, 0.5)
    p_bal, r_bal, f_bal = eval_at_threshold(y_test_all, proba_balanced_all, 0.5)

    # best F1 threshold for the balanced model (threshold tuning)
    best_f1, best_thresh, best_p, best_r = -1, 0.5, 0, 0
    threshold_curve = []
    for t in THRESHOLDS:
        p, r, f = eval_at_threshold(y_test_all, proba_balanced_all, t)
        threshold_curve.append((t, p, r, f))
        if f > best_f1:
            best_f1, best_thresh, best_p, best_r = f, t, p, r

    print(f"rho={rho}: RF default (thresh=0.5) P={p_def:.3f} R={r_def:.3f} F1={f_def:.3f} | "
          f"RF balanced (thresh=0.5) P={p_bal:.3f} R={r_bal:.3f} F1={f_bal:.3f} | "
          f"RF balanced (best thresh={best_thresh:.2f}) P={best_p:.3f} R={best_r:.3f} F1={best_f1:.3f}")

    return {
        "rho": rho,
        "RF_default_precision": p_def, "RF_default_recall": r_def, "RF_default_f1": f_def,
        "RF_balanced_precision": p_bal, "RF_balanced_recall": r_bal, "RF_balanced_f1": f_bal,
        "RF_balanced_tuned_threshold": best_thresh,
        "RF_balanced_tuned_precision": best_p, "RF_balanced_tuned_recall": best_r,
        "RF_balanced_tuned_f1": best_f1,
    }, threshold_curve


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    for rho in RHOS:
        row, curve = run_for_rho(rho)
        rows.append(row)
        curves[rho] = curve

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "class_weight_threshold_tuning.csv", index=False)
    print("\nSaved: results/class_weight_threshold_tuning.csv")

    # plot precision/recall vs threshold for the two lowest rho (where the
    # original problem was worst)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rho in zip(axes, [0.3, 0.5]):
        curve = np.array(curves[rho])
        ax.plot(curve[:, 0], curve[:, 1], "o-", label="Precision")
        ax.plot(curve[:, 0], curve[:, 2], "s-", label="Recall")
        ax.plot(curve[:, 0], curve[:, 3], "^-", label="F1")
        ax.set_xlabel("Classification threshold")
        ax.set_title(f"rho={rho} (class-weighted RF)")
        ax.legend()
    fig.suptitle("Precision/recall/F1 vs. classification threshold, class-weighted Random Forest")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "threshold_tuning_low_rho.png", dpi=130)
    plt.close(fig)
    print("Saved: results/plots/threshold_tuning_low_rho.png")

    return result_df


if __name__ == "__main__":
    run_all()