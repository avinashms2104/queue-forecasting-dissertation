"""
XGBoost baseline (history-only features) for all five queue types, evaluated
exactly like the Random Forest baselines: trained once on the 20 training runs,
scored separately on each of the 5 held-out test runs, against naive persistence.

Why: XGBoost is on Azam's list of methods (notebook Section 3.8) but had never
been run. It is a gradient-boosted TREE model, i.e. still a "simple" classical
model in the sense of Meeting 2 (no LSTM / GRU / Transformer). This script adds
a column to the existing comparison; it does not change any existing result.

Two regression variants are fitted, to test a hypothesis raised by the RMSE
summary (summarise_regression_models.py): Random Forest beats persistence
clearly on RMSE but only slightly on MAE. Squared-error models predict the
conditional MEAN, which is what RMSE rewards; MAE is minimised by the
conditional MEDIAN, which for a queue that is often empty can sit at or near
the current value. So:
    XGBoost (L2): objective = squared error   (predicts the mean)
    XGBoost (L1): objective = absolute error  (predicts the median)
If the L1 version beats persistence on MAE where L2 and Random Forest did not,
that supports the explanation; if not, the hypothesis is wrong.

A third reference, "Constant (train median)", always predicts the median of the
training-run targets and uses no input at all. It exists because persistence is
a WEAK MAE baseline at low rho (the queue is usually empty, so copying the
current value carries transient spikes forward): at low rho even this trivial
constant beats persistence by 30-50% on MAE. "Beats persistence" is therefore
not enough evidence of skill; a model should also beat the constant.

Classification: XGBClassifier with the default 0.5 threshold and no class
weighting, i.e. the same "plain" setup as the Random Forest classifier.

Usage
-----
    python modeling/train_xgboost_models.py                  # medium horizon only (fast, default)
    python modeling/train_xgboost_models.py --all-horizons   # all four horizons

Writes: results/xgboost_results_per_run.csv, results/xgboost_summary.csv
Needs:  the processed history-only datasets from the build_*_datasets.py scripts,
        and results/*_model_comparison_*.csv for the Random Forest columns.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, f1_score)
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
TEST_RUN_IDS = [25, 26, 27, 28, 29]
SEED = 42

XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                  tree_method="hist", n_jobs=-1, random_state=SEED)

# (label, path of the processed history-only file for a given rho)
QUEUES = [
    ("M/M/1", lambda r: DATA / "processed" / f"history_only_rho{r}.csv"),
    ("M/G/1 Weibull 2.0", lambda r: DATA / "mg1" / "processed" / f"history_only_shape2.0_rho{r}.csv"),
    ("M/G/1 Weibull 0.5", lambda r: DATA / "mg1" / "processed" / f"history_only_shape0.5_rho{r}.csv"),
    ("M/PH/1 Erlang-2", lambda r: DATA / "mph1" / "processed" / f"history_only_erlang2_rho{r}.csv"),
    ("PH/PH/1 E2/E2", lambda r: DATA / "phph1" / "processed" / f"history_only_arrE2_srvE2_rho{r}.csv"),
]


def run_setting(queue, rho, path, horizons):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c.startswith("queue_length_lag")]
    train = df[df["split"] == "train"]
    df["current_congested"] = (df["queue_length_lag0"] > df["congestion_threshold"]).astype(int)
    rows = []
    for h in horizons:
        yq, yc = f"target_qlen_{h}", f"target_congested_{h}"
        m_l2 = XGBRegressor(objective="reg:squarederror", **XGB_PARAMS).fit(train[feat], train[yq])
        m_l1 = XGBRegressor(objective="reg:absoluteerror", **XGB_PARAMS).fit(train[feat], train[yq])
        m_c = XGBClassifier(**XGB_PARAMS).fit(train[feat], train[yc])
        train_median = float(train[yq].median())
        for rid in TEST_RUN_IDS:
            te = df[(df["split"] == "test") & (df["run_id"] == rid)]
            if len(te) == 0:
                continue
            base = dict(queue=queue, rho=rho, horizon=h, test_run_id=rid)
            y = te[yq]
            for name, pred in [("Naive (persistence)", te["queue_length_lag0"].values),
                               ("XGBoost (L2)", m_l2.predict(te[feat])),
                               ("XGBoost (L1)", m_l1.predict(te[feat])),
                               ("Constant (train median)", np.full(len(te), train_median))]:
                rows.append({**base, "task": "regression", "model": name,
                             "MAE": mean_absolute_error(y, pred),
                             "RMSE": mean_squared_error(y, pred) ** 0.5})
            yb = te[yc]
            for name, pred in [("Naive (persistence)", te["current_congested"].values),
                               ("XGBoost", m_c.predict(te[feat]))]:
                rows.append({**base, "task": "classification", "model": name,
                             "precision": precision_score(yb, pred, zero_division=0),
                             "recall": recall_score(yb, pred, zero_division=0),
                             "F1": f1_score(yb, pred, zero_division=0)})
    return rows


def load_rf_reference():
    """Existing Random Forest results (mean over the 5 test runs), history-only, from the results CSVs."""
    specs = [
        ("M/M/1", "model_comparison", lambda d: d),
        ("M/G/1 Weibull 2.0", "mg1_model_comparison", lambda d: d[d.weibull_shape == 2.0]),
        ("M/G/1 Weibull 0.5", "mg1_model_comparison", lambda d: d[d.weibull_shape == 0.5]),
        ("M/PH/1 Erlang-2", "mph1_model_comparison", lambda d: d[d.erlang_k == 2]),
        ("PH/PH/1 E2/E2", "phph1_model_comparison",
         lambda d: d[(d.arrival_erlang_k == 2) & (d.service_erlang_k == 2)]),
    ]
    reg, clf = [], []
    for q, stem, flt in specs:
        r = pd.read_csv(RESULTS / f"{stem}_regression.csv")
        r = flt(r[(r.dataset == "history_only") & (r.test_run_id.astype(str) != "ALL_MEAN") & (r.model == "Random Forest")])
        reg.append(r.assign(queue=q)[["queue", "rho", "horizon", "test_run_id", "MAE", "RMSE"]])
        c = pd.read_csv(RESULTS / f"{stem}_classification.csv")
        c = flt(c[(c.dataset == "history_only") & (c.test_run_id.astype(str) != "ALL_MEAN") & (c.model == "Random Forest")])
        clf.append(c.assign(queue=q)[["queue", "rho", "horizon", "test_run_id", "f1_congested"]])
    return pd.concat(reg), pd.concat(clf)


def main(all_horizons):
    horizons = ["short", "medium", "extra_long", "long"] if all_horizons else ["medium"]
    rows = []
    for queue, path_fn in QUEUES:
        for rho in RHOS:
            rows += run_setting(queue, rho, path_fn(rho), horizons)
            print(f"done: {queue} rho={rho}", flush=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(RESULTS / "xgboost_results_per_run.csv", index=False)

    rf_reg, rf_clf = load_rf_reference()
    key = ["queue", "rho", "horizon"]
    reg = per_run[per_run.task == "regression"]
    mean_reg = reg.groupby(key + ["model"])[["MAE", "RMSE"]].mean().reset_index()
    rf_mean = rf_reg.groupby(key)[["MAE", "RMSE"]].mean().reset_index().assign(model="Random Forest")
    mean_reg = pd.concat([mean_reg, rf_mean[mean_reg.columns]], ignore_index=True)
    naive = mean_reg[mean_reg.model == "Naive (persistence)"].rename(
        columns={"MAE": "MAE_n", "RMSE": "RMSE_n"})[key + ["MAE_n", "RMSE_n"]]
    mean_reg = mean_reg.merge(naive, on=key)
    mean_reg["MAE_impr_pct"] = 100 * (1 - mean_reg["MAE"] / mean_reg["MAE_n"])
    mean_reg["RMSE_impr_pct"] = 100 * (1 - mean_reg["RMSE"] / mean_reg["RMSE_n"])
    mean_reg.to_csv(RESULTS / "xgboost_summary.csv", index=False)

    med = mean_reg[mean_reg.horizon == "medium"]
    order = ["Random Forest", "XGBoost (L2)", "XGBoost (L1)", "Constant (train median)"]
    qlist = [q for q, _ in QUEUES]
    pd.set_option("display.width", 200)
    for metric, col in [("MAE", "MAE_impr_pct"), ("RMSE", "RMSE_impr_pct")]:
        print("\n" + "=" * 100)
        print(f"{metric} improvement over persistence (%), MEDIUM horizon, mean over rho (positive = better than persistence)")
        print("=" * 100)
        t = med[med.model.isin(order)].groupby(["queue", "model"])[col].mean().unstack("model")
        print(t.loc[qlist, order].round(1).to_string())

    print("\n" + "=" * 100)
    print("XGBoost (L1) MAE improvement over persistence (%), medium horizon, by rho")
    print("=" * 100)
    t = med[med.model == "XGBoost (L1)"].pivot(index="rho", columns="queue", values="MAE_impr_pct")[qlist]
    print(t.round(1).to_string())

    print("\n" + "=" * 100)
    print("Constant (train median) MAE improvement over persistence (%), medium horizon, by rho")
    print("(a model that ignores its input; positive = even this beats persistence)")
    print("=" * 100)
    t = med[med.model == "Constant (train median)"].pivot(index="rho", columns="queue", values="MAE_impr_pct")[qlist]
    print(t.round(1).to_string())

    print("\n" + "=" * 100)
    print("XGBoost (L1) MAE improvement over the CONSTANT baseline (%), medium horizon, by rho")
    print("(positive = the model learned something beyond 'always predict the typical value')")
    print("=" * 100)
    l1 = med[med.model == "XGBoost (L1)"].set_index(key)["MAE"]
    cc = med[med.model == "Constant (train median)"].set_index(key)["MAE"]
    vs_const = (100 * (1 - l1 / cc)).reset_index(name="pct")
    print(vs_const.pivot(index="rho", columns="queue", values="pct")[qlist].round(1).to_string())
    pers = med[med.model == "Naive (persistence)"].set_index(key)["MAE"]
    print(f"\nOf {len(l1)} settings (medium horizon): XGBoost (L1) has lower MAE than persistence in "
          f"{int((l1 < pers).sum())}, lower than the constant baseline in {int((l1 < cc).sum())}; "
          f"the constant baseline is lower than persistence in {int((cc < pers).sum())}.")

    clf = per_run[per_run.task == "classification"]
    mc = clf.groupby(key + ["model"])["F1"].mean().unstack("model").reset_index()
    rf_f1 = rf_clf.groupby(key)["f1_congested"].mean().reset_index().rename(columns={"f1_congested": "RF"})
    mc = mc.merge(rf_f1, on=key)
    mc["XGB_minus_persistence"] = mc["XGBoost"] - mc["Naive (persistence)"]
    mc["RF_minus_persistence"] = mc["RF"] - mc["Naive (persistence)"]
    mm = mc[mc.horizon == "medium"]
    print("\n" + "=" * 100)
    print("Classification, MEDIUM horizon: F1(model) - F1(persistence)  (negative = persistence better)")
    print("=" * 100)
    for lab, col in [("XGBoost", "XGB_minus_persistence"), ("Random Forest (existing)", "RF_minus_persistence")]:
        print(f"\n{lab}")
        print(mm.pivot(index="rho", columns="queue", values=col)[qlist].round(3).to_string())
    print(f"\nXGBoost F1 >= persistence in {(mm.XGB_minus_persistence >= 0).sum()} of {len(mm)} settings "
          f"(Random Forest: {(mm.RF_minus_persistence >= 0).sum()} of {len(mm)})")
    print("\nSaved: results/xgboost_results_per_run.csv, results/xgboost_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-horizons", action="store_true")
    main(ap.parse_args().all_horizons)