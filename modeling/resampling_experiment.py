"""
Resampling experiment for congestion classification (M/M/1, history-only features).

Compares, at each rho, on the 5 held-out TEST runs:
    - persistence       : predict "congested at t+h" iff current queue is congested now
    - RF_default        : plain Random Forest
    - RF_class_weight   : Random Forest with class_weight="balanced"
    - RF_class_weight_valtuned : same, threshold picked on the 5 VALIDATION runs
    - RF_oversample     : random oversampling of the minority class (train runs only)
    - RF_undersample    : random undersampling of the majority class (train runs only)
    - RF_smote          : SMOTE (only if imbalanced-learn is installed)

Resampling is applied ONLY to the training runs. Validation and test rows are never touched.

Run from the project folder:
    python modeling/resampling_experiment.py

Output: results/resampling_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
HORIZON = "medium"          # same horizon as the class-weighting experiment (h = 10)
SEED = 42

ROOT = Path(__file__).resolve().parent.parent


def random_oversample(X, y, rng):
    """Duplicate minority rows (with replacement) until classes are balanced."""
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return X, y
    if len(idx_pos) < len(idx_neg):
        minority, majority = idx_pos, idx_neg
    else:
        minority, majority = idx_neg, idx_pos
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    keep = np.concatenate([majority, minority, extra])
    return X[keep], y[keep]


def random_undersample(X, y, rng):
    """Drop majority rows (without replacement) until classes are balanced."""
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return X, y
    if len(idx_pos) < len(idx_neg):
        minority, majority = idx_pos, idx_neg
    else:
        minority, majority = idx_neg, idx_pos
    kept_majority = rng.choice(majority, size=len(minority), replace=False)
    keep = np.concatenate([minority, kept_majority])
    return X[keep], y[keep]


def smote_resample(X, y):
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=SEED).fit_resample(X, y)


def metrics(y_true, y_pred):
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )


def make_rf(**kw):
    return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1, **kw)


def main(data_dir, out_path):
    try:
        import imblearn  # noqa: F401
        have_smote = True
    except ImportError:
        have_smote = False
        print("imbalanced-learn not installed -> skipping SMOTE "
              "(install with: pip install imbalanced-learn)")

    rng = np.random.default_rng(SEED)
    target = f"target_congested_{HORIZON}"
    rows = []

    for rho in RHOS:
        df = pd.read_csv(data_dir / f"history_only_rho{rho}.csv")
        lag_cols = [c for c in df.columns if c.startswith("queue_length_lag")]

        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]

        X_tr, y_tr = train[lag_cols].to_numpy(), train[target].to_numpy()
        X_te, y_te = test[lag_cols].to_numpy(), test[target].to_numpy()
        base_rate = float(y_te.mean())

        results = {}

        # persistence: is the queue congested RIGHT NOW? carry that forward
        y_pers = (test["queue_length_lag0"] > test["congestion_threshold"]).astype(int).to_numpy()
        results["persistence"] = metrics(y_te, y_pers)

        # default RF
        results["RF_default"] = metrics(y_te, make_rf().fit(X_tr, y_tr).predict(X_te))

        # class weighting
        rf_bal = make_rf(class_weight="balanced").fit(X_tr, y_tr)
        results["RF_class_weight"] = metrics(y_te, rf_bal.predict(X_te))

        # class weighting + threshold chosen on the VALIDATION runs (not the test runs)
        val = df[df["split"] == "val"]
        p_val = rf_bal.predict_proba(val[lag_cols].to_numpy())[:, 1]
        best_t = max(np.arange(0.05, 0.95, 0.05),
                     key=lambda t: f1_score(val[target].to_numpy(), (p_val >= t).astype(int),
                                            zero_division=0))
        p_te = rf_bal.predict_proba(X_te)[:, 1]
        results["RF_class_weight_valtuned"] = metrics(y_te, (p_te >= best_t).astype(int))

        # random oversampling
        Xo, yo = random_oversample(X_tr, y_tr, rng)
        results["RF_oversample"] = metrics(y_te, make_rf().fit(Xo, yo).predict(X_te))

        # random undersampling
        Xu, yu = random_undersample(X_tr, y_tr, rng)
        results["RF_undersample"] = metrics(y_te, make_rf().fit(Xu, yu).predict(X_te))

        # SMOTE (optional)
        if have_smote:
            Xs, ys = smote_resample(X_tr, y_tr)
            results["RF_smote"] = metrics(y_te, make_rf().fit(Xs, ys).predict(X_te))

        for name, (p, r, f) in results.items():
            rows.append({"rho": rho, "method": name, "precision": p, "recall": r,
                         "f1": f, "test_congestion_rate": base_rate})

        print(f"rho={rho}  base rate={base_rate:.3f}")
        for name, (p, r, f) in results.items():
            print(f"   {name:16s} precision={p:.3f} recall={r:.3f} f1={f:.3f}")

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "resampling_results.csv")
    args = ap.parse_args()
    main(args.data_dir, args.out)