"""
Class weighting and threshold tuning for congestion classification, repeated
for ALL seven queue types (the original class_weight_threshold_tuning.py covered
M/M/1 only; open question 3 to Azam).

Four ways of predicting congestion at the medium horizon (history-only features):
    Persistence             "congested now -> congested later" (the baseline)
    RF default              Random Forest, 0.5 probability cut-off
    RF balanced             Random Forest with class_weight="balanced", 0.5 cut-off
    RF balanced + tuned     the balanced model with the cut-off chosen to maximise
                            F1 on the VALIDATION runs (20-24), then applied unchanged
                            to the TEST runs (25-29)

Difference from the original M/M/1 script: there the best cut-off was chosen on
the test runs themselves, which flatters the tuned result slightly. Here the
cut-off is chosen on validation runs only, so the test score is a fair estimate.
The original script and its results are left unchanged.

Scores are computed on each of the 5 test runs separately and averaged (mean and
std), exactly as in train_*_baseline_models.py.

Usage:  python modeling/class_weight_all_queues.py
Writes: results/class_weight_all_queues.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
HORIZON = "medium"
SEED = 42
TEST_RUN_IDS = [25, 26, 27, 28, 29]
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.05), 2)

# (label, processed history-only file for a given rho)
QUEUES = [
    ("M/M/1", lambda r: DATA / "processed" / f"history_only_rho{r}.csv"),
    ("M/G/1 Weibull 2.0", lambda r: DATA / "mg1" / "processed" / f"history_only_shape2.0_rho{r}.csv"),
    ("M/G/1 Weibull 0.5", lambda r: DATA / "mg1" / "processed" / f"history_only_shape0.5_rho{r}.csv"),
    ("M/PH/1 Erlang-2", lambda r: DATA / "mph1" / "processed" / f"history_only_erlang2_rho{r}.csv"),
    ("PH/PH/1 E2/E2", lambda r: DATA / "phph1" / "processed" / f"history_only_arrE2_srvE2_rho{r}.csv"),
    ("H2/PH/1 H2/E2", lambda r: DATA / "h2ph1" / "processed" / f"history_only_rho{r}.csv"),
    ("MAP/PH/1 MMPP2/E2", lambda r: DATA / "mapph1" / "processed" / f"history_only_rho{r}.csv"),
]
METHODS = ["Persistence", "RF default", "RF balanced", "RF balanced + tuned"]


def scores(y, pred):
    return (precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def run_setting(queue, rho, path):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c.startswith("queue_length_lag")]
    target = f"target_congested_{HORIZON}"
    df["current_congested"] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    rf_def = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_bal = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1,
                                    class_weight="balanced")
    rf_def.fit(train[feat], train[target])
    rf_bal.fit(train[feat], train[target])

    # choose the cut-off on the validation runs only
    p_val = rf_bal.predict_proba(val[feat])[:, 1]
    val_f1 = [f1_score(val[target], (p_val >= c).astype(int), zero_division=0) for c in CUTOFFS]
    best_cut = float(CUTOFFS[int(np.argmax(val_f1))])

    rows = []
    for rid in TEST_RUN_IDS:
        te = df[(df["split"] == "test") & (df["run_id"] == rid)]
        if len(te) == 0:
            continue
        y = te[target]
        p_bal = rf_bal.predict_proba(te[feat])[:, 1]
        preds = {
            "Persistence": te["current_congested"].values,
            "RF default": rf_def.predict(te[feat]),
            "RF balanced": (p_bal >= 0.5).astype(int),
            "RF balanced + tuned": (p_bal >= best_cut).astype(int),
        }
        for method, pred in preds.items():
            p, r, f = scores(y, pred)
            rows.append(dict(queue=queue, rho=rho, horizon=HORIZON, method=method, test_run_id=rid,
                             precision=p, recall=r, f1=f, cutoff=best_cut if "tuned" in method else 0.5,
                             congestion_rate=float(y.mean())))
    return rows


def main():
    rows = []
    for queue, path_fn in QUEUES:
        if not path_fn(RHOS[0]).exists():
            print(f"(skipping {queue}: no processed data found)")
            continue
        for rho in RHOS:
            rows += run_setting(queue, rho, path_fn(rho))
            print(f"done: {queue} rho={rho}", flush=True)

    per_run = pd.DataFrame(rows)
    summ = (per_run.groupby(["queue", "rho", "method"])
            .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                 f1=("f1", "mean"), f1_std=("f1", "std"), cutoff=("cutoff", "first"),
                 congestion_rate=("congestion_rate", "mean"))
            .reset_index())
    summ.to_csv(RESULTS / "class_weight_all_queues.csv", index=False)

    qlist = [q for q, _ in QUEUES if q in set(summ.queue)]
    pd.set_option("display.width", 220)

    for metric in ["recall", "f1"]:
        print("\n" + "=" * 110)
        print(f"{metric.upper()} on the congested class, medium horizon (mean of 5 test runs)")
        print("=" * 110)
        for q in qlist:
            t = summ[summ.queue == q].pivot(index="rho", columns="method", values=metric)[METHODS]
            print(f"\n{q}")
            print(t.round(3).to_string())

    print("\n" + "=" * 110)
    print("Cut-off chosen on the validation runs (RF balanced + tuned)")
    print("=" * 110)
    cut = summ[summ.method == "RF balanced + tuned"].pivot(index="rho", columns="queue", values="cutoff")[qlist]
    print(cut.to_string())

    print("\n" + "=" * 110)
    print("Best of the three Random Forest variants minus persistence, F1 (positive = a model beats persistence)")
    print("=" * 110)
    f1 = summ.pivot_table(index=["queue", "rho"], columns="method", values="f1")
    best_rf = f1[["RF default", "RF balanced", "RF balanced + tuned"]].max(axis=1)
    gap = (best_rf - f1["Persistence"]).unstack("queue")[qlist]
    print(gap.round(3).to_string())
    n = gap.size
    print(f"\nBest Random Forest variant beats persistence on F1 by more than 0.01 in "
          f"{int((gap > 0.01).sum().sum())} of {n} settings; within 0.01 in "
          f"{int((gap.abs() <= 0.01).sum().sum())}; worse by more than 0.01 in {int((gap < -0.01).sum().sum())}.")
    print("\nSaved: results/class_weight_all_queues.csv")


if __name__ == "__main__":
    main()