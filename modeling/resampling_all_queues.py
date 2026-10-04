"""
Resampling experiment for congestion classification, repeated for ALL seven queue
types (the original resampling_experiment.py covered M/M/1 only; open question 3
to Azam). Medium horizon, history-only features.

Methods (resampling is applied to the 20 TRAINING runs only; validation and test
rows are never touched):
    Persistence           "congested now -> congested later" (the baseline)
    Always congested      flags every time point as congested; uses no input
    RF oversample         Random Forest after duplicating minority rows to balance
    RF undersample        Random Forest after dropping majority rows to balance

"Always congested" is a reference, not a model. Its F1 is fixed by the
congestion rate alone (recall 1, precision = congestion rate). It is added
because class_weight_all_queues.py showed that at low rho the tuned model's
apparent wins over persistence came from flagging almost everything: any
method whose F1 is close to this reference has learned little.

Class weighting and validation-tuned thresholds are in class_weight_all_queues.py,
so they are not repeated here. SMOTE is not used, as in the original M/M/1 run.
Scores are computed on each of the 5 test runs separately and averaged.

Usage:  python modeling/resampling_all_queues.py
Writes: results/resampling_all_queues.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from class_weight_all_queues import QUEUES, RHOS, HORIZON, SEED, TEST_RUN_IDS

RESULTS = Path(__file__).resolve().parent.parent / "results"
METHODS = ["Persistence", "Always congested", "RF oversample", "RF undersample"]


def balance(X, y, rng, mode):
    """Random over- or undersampling to a 50/50 class split (training rows only)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if mode == "over":
        extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
        keep = np.concatenate([majority, minority, extra])
    else:
        keep = np.concatenate([minority, rng.choice(majority, size=len(minority), replace=False)])
    return X[keep], y[keep]


def make_rf():
    return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)


def run_setting(queue, rho, path, rng):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c.startswith("queue_length_lag")]
    target = f"target_congested_{HORIZON}"
    df["current_congested"] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)

    train = df[df["split"] == "train"]
    X_tr, y_tr = train[feat].to_numpy(), train[target].to_numpy()
    rf_over = make_rf().fit(*balance(X_tr, y_tr, rng, "over"))
    rf_under = make_rf().fit(*balance(X_tr, y_tr, rng, "under"))

    rows = []
    for rid in TEST_RUN_IDS:
        te = df[(df["split"] == "test") & (df["run_id"] == rid)]
        if len(te) == 0:
            continue
        y = te[target].to_numpy()
        X = te[feat].to_numpy()
        preds = {
            "Persistence": te["current_congested"].to_numpy(),
            "Always congested": np.ones(len(te), dtype=int),
            "RF oversample": rf_over.predict(X),
            "RF undersample": rf_under.predict(X),
        }
        for method, pred in preds.items():
            rows.append(dict(queue=queue, rho=rho, horizon=HORIZON, method=method, test_run_id=rid,
                             precision=precision_score(y, pred, zero_division=0),
                             recall=recall_score(y, pred, zero_division=0),
                             f1=f1_score(y, pred, zero_division=0),
                             congestion_rate=float(y.mean())))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for queue, path_fn in QUEUES:
        if not path_fn(RHOS[0]).exists():
            print(f"(skipping {queue}: no processed data found)")
            continue
        for rho in RHOS:
            rows += run_setting(queue, rho, path_fn(rho), rng)
            print(f"done: {queue} rho={rho}", flush=True)

    per_run = pd.DataFrame(rows)
    summ = (per_run.groupby(["queue", "rho", "method"])
            .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                 f1=("f1", "mean"), f1_std=("f1", "std"),
                 congestion_rate=("congestion_rate", "mean"))
            .reset_index())
    summ.to_csv(RESULTS / "resampling_all_queues.csv", index=False)

    qlist = [q for q, _ in QUEUES if q in set(summ.queue)]
    pd.set_option("display.width", 220)
    for metric in ["recall", "precision", "f1"]:
        print("\n" + "=" * 110)
        print(f"{metric.upper()} on the congested class, medium horizon (mean of 5 test runs)")
        print("=" * 110)
        for q in qlist:
            t = summ[summ.queue == q].pivot(index="rho", columns="method", values=metric)[METHODS]
            print(f"\n{q}")
            print(t.round(3).to_string())

    f1 = summ.pivot_table(index=["queue", "rho"], columns="method", values="f1")
    best_rs = f1[["RF oversample", "RF undersample"]].max(axis=1)
    gap = (best_rs - f1["Persistence"]).unstack("queue")[qlist]
    print("\n" + "=" * 110)
    print("Best resampled Random Forest minus persistence, F1 (positive = resampling beats persistence)")
    print("=" * 110)
    print(gap.round(3).to_string())
    print(f"\nBetter than persistence by more than 0.01 in {int((gap > 0.01).sum().sum())} of {gap.size}; "
          f"within 0.01 in {int((gap.abs() <= 0.01).sum().sum())}; "
          f"worse by more than 0.01 in {int((gap < -0.01).sum().sum())}.")

    always = f1["Always congested"].unstack("queue")[qlist]
    pers = f1["Persistence"].unstack("queue")[qlist]
    print("\n" + "=" * 110)
    print("Persistence minus 'Always congested', F1 (negative = flagging everything beats persistence)")
    print("=" * 110)
    print((pers - always).round(3).to_string())
    print("\nSaved: results/resampling_all_queues.csv")


if __name__ == "__main__":
    main()