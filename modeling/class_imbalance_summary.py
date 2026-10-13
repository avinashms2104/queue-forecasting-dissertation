"""
Small class-imbalance experiment in the form Azam asked for (email of 5 October
2026, priority 3): compare the UNWEIGHTED model with the most promising
imbalance approach from M/M/1, at one representative low and one high
utilisation, and report congestion precision, recall and F1.

Step 1 - choose the approach (validation runs only):
    Candidates, all Random Forest classifiers (50 trees, depth 8) with the
    standard 0.5 cut-off, history-only features, medium horizon (h = 10):
        class weighting   class_weight="balanced"
        oversampling      duplicate congested training rows to a 50/50 split
        undersampling     drop non-congested training rows to a 50/50 split
    Each is trained on the 20 training runs of M/M/1 and scored on the 5
    VALIDATION runs (20-24) at the same low and high rho used in step 2. The
    approach with the highest mean validation F1 is chosen. The test runs play no part in this choice.
    (A cut-off tuned on the validation runs is left out here, because it would
    be chosen and scored on the same runs; it is reported separately in
    class_weight_all_queues.py.)

Step 2 - report (test runs only):
    For M/M/1, M/PH/1 and PH/PH/1 at rho = 0.5 (low) and rho = 0.9 (high):
    persistence, the unweighted Random Forest and the chosen approach, scored on
    the 5 TEST runs (25-29): precision, recall and F1 for the congested class,
    mean and standard deviation over the test runs.
    rho = 0.3 is not used as the "low" setting because its congestion threshold
    is 0 for these queues, so "congested" would simply mean "anyone waiting".

Resampling is applied to the training rows only; validation and test runs keep
their original class proportions.

--map-levels reports the same comparison for H2/PH/1 and MAP/PH/1 with weak,
medium and strong bursts (the approach is still chosen on M/M/1 validation runs).

Usage:  python modeling/class_imbalance_summary.py
        python modeling/class_imbalance_summary.py --map-levels
Writes: results/class_imbalance_selection.csv          (validation scores used for the choice)
        results/class_imbalance_summary{_map_levels}.csv  (test scores at low and high rho)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from class_weight_all_queues import QUEUES as ALL_QUEUES, SEED
from resampling_all_queues import balance

RESULTS = Path(__file__).resolve().parent.parent / "results"
HORIZON = "medium"
VAL_RUN_IDS = [20, 21, 22, 23, 24]
TEST_RUN_IDS = [25, 26, 27, 28, 29]
LOW_RHO, HIGH_RHO = 0.5, 0.9
SELECT_QUEUE = "M/M/1"
REPORT_QUEUES = ["M/M/1", "M/PH/1 Erlang-2", "PH/PH/1 E2/E2"]
CANDIDATES = ["Class weighting", "Oversampling", "Undersampling"]
PATHS = dict(ALL_QUEUES)
DATA = Path(__file__).resolve().parent.parent / "data"
MAP_LEVEL_PATHS = {
    "H2 (no bursts)": lambda r: DATA / "h2ph1" / "processed" / f"history_only_rho{r}.csv",
    "MAP weak": lambda r: DATA / "mapph1" / "processed" / f"history_only_weak_rho{r}.csv",
    "MAP medium": lambda r: DATA / "mapph1" / "processed" / f"history_only_rho{r}.csv",
    "MAP strong": lambda r: DATA / "mapph1" / "processed" / f"history_only_strong_rho{r}.csv",
}
PATHS.update(MAP_LEVEL_PATHS)


def fit(method, X, y, rng):
    kw = dict(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
    if method == "Unweighted":
        return RandomForestClassifier(**kw).fit(X, y)
    if method == "Class weighting":
        return RandomForestClassifier(class_weight="balanced", **kw).fit(X, y)
    mode = "over" if method == "Oversampling" else "under"
    return RandomForestClassifier(**kw).fit(*balance(X, y, rng, mode))


def per_run_scores(df, run_ids, pred_fn):
    out = []
    for rid in run_ids:
        te = df[df.run_id == rid]
        if len(te) == 0:
            continue
        y = te[f"target_congested_{HORIZON}"].to_numpy()
        p = pred_fn(te)
        out.append((precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0),
                    f1_score(y, p, zero_division=0)))
    return np.array(out)


def load(queue, rho):
    df = pd.read_csv(PATHS[queue](rho))
    feat = [c for c in df.columns if c.startswith("queue_length_lag")]
    df["current_congested"] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)
    tr = df[df.split == "train"]
    return df, feat, tr[feat].to_numpy(), tr[f"target_congested_{HORIZON}"].to_numpy()


def main(map_levels=False):
    report_queues = list(MAP_LEVEL_PATHS) if map_levels else REPORT_QUEUES
    tag = "_map_levels" if map_levels else ""
    rng = np.random.default_rng(SEED)

    # ---------- step 1: choose on M/M/1 validation runs ----------
    sel_rows = []
    for rho in [LOW_RHO, HIGH_RHO]:
        df, feat, X, y = load(SELECT_QUEUE, rho)
        for method in ["Unweighted"] + CANDIDATES:
            m = fit(method, X, y, rng)
            s = per_run_scores(df, VAL_RUN_IDS, lambda te: m.predict(te[feat].to_numpy()))
            sel_rows.append(dict(queue=SELECT_QUEUE, rho=rho, method=method,
                                 precision=s[:, 0].mean(), recall=s[:, 1].mean(), f1=s[:, 2].mean()))
        print(f"selection: {SELECT_QUEUE} rho={rho} done", flush=True)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(RESULTS / "class_imbalance_selection.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"STEP 1: validation F1 on {SELECT_QUEUE} (runs 20-24), medium horizon, 0.5 cut-off")
    print("=" * 100)
    print(sel.pivot(index="rho", columns="method", values="f1")[["Unweighted"] + CANDIDATES].round(3).to_string())
    mean_f1 = sel[sel.method.isin(CANDIDATES)].groupby("method")["f1"].mean()
    chosen = mean_f1.idxmax()
    print(f"\nMean validation F1 over rho = {LOW_RHO} and {HIGH_RHO}: " + ", ".join(f"{k} {v:.3f}" for k, v in mean_f1.items()))
    print(f"Chosen approach: {chosen}")

    # ---------- step 2: report on test runs at low and high rho ----------
    rows = []
    for queue in report_queues:
        if not PATHS[queue](LOW_RHO).exists():
            print(f"(skipping {queue}: no processed data found)")
            continue
        for rho in [LOW_RHO, HIGH_RHO]:
            df, feat, X, y = load(queue, rho)
            models = {"Unweighted RF": fit("Unweighted", X, y, rng), f"{chosen} RF": fit(chosen, X, y, rng)}
            preds = {"Persistence": lambda te: te["current_congested"].to_numpy()}
            for name, m in models.items():
                preds[name] = (lambda mm: (lambda te: mm.predict(te[feat].to_numpy())))(m)
            for name, fn in preds.items():
                s = per_run_scores(df, TEST_RUN_IDS, fn)
                rows.append(dict(queue=queue, rho=rho, level="low" if rho == LOW_RHO else "high",
                                 method=name,
                                 precision=s[:, 0].mean(), precision_std=s[:, 0].std(ddof=1),
                                 recall=s[:, 1].mean(), recall_std=s[:, 1].std(ddof=1),
                                 f1=s[:, 2].mean(), f1_std=s[:, 2].std(ddof=1)))
            print(f"report: {queue} rho={rho} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"class_imbalance_summary{tag}.csv", index=False)

    print("\n" + "=" * 100)
    print(f"STEP 2: test runs (25-29), medium horizon. Congestion precision / recall / F1, mean (std) over 5 runs")
    print("=" * 100)
    for queue in res.queue.unique():
        for rho in [LOW_RHO, HIGH_RHO]:
            print(f"\n{queue}, rho = {rho}")
            for _, r in res[(res.queue == queue) & (res.rho == rho)].iterrows():
                print(f"   {r.method:28s} precision {r.precision:.3f} ({r.precision_std:.3f})   "
                      f"recall {r.recall:.3f} ({r.recall_std:.3f})   F1 {r.f1:.3f} ({r.f1_std:.3f})")

    print("\n" + "=" * 100)
    print(f"{chosen} RF minus unweighted RF (test runs): change in precision, recall, F1")
    print("=" * 100)
    for queue in res.queue.unique():
        for rho in [LOW_RHO, HIGH_RHO]:
            s = res[(res.queue == queue) & (res.rho == rho)].set_index("method")[["precision", "recall", "f1"]]
            d = s.loc[f"{chosen} RF"] - s.loc["Unweighted RF"]
            p = s.loc[f"{chosen} RF"] - s.loc["Persistence"]
            print(f"{queue:18s} rho={rho}: precision {d.precision:+.3f}, recall {d.recall:+.3f}, "
                  f"F1 {d.f1:+.3f}   |  vs persistence: F1 {p.f1:+.3f}")
    print(f"\nSaved: results/class_imbalance_selection.csv, results/class_imbalance_summary{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-levels", action="store_true")
    main(ap.parse_args().map_levels)