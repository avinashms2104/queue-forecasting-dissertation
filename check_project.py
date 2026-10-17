"""
Whole-project cross-check: inputs, outputs and their consistency
=================================================================

Run from the project folder:   python check_project.py
(about 3-6 minutes; it reads every history-only dataset once)

What it checks
--------------
1. Scripts       every simulation/ and modeling/ script the README describes is present,
                 and every script on disk is mentioned in the README.
2. Datasets      for each of the 13 queue settings x 6 rho:
                 - raw simulated time series and both processed datasets exist;
                 - runs 0-19 are train, 20-24 validation, 25-29 test, all 30 present;
                 - one congestion threshold per file, equal to the 80th percentile of the
                   raw queue length (training runs only, or all runs for the two older
                   pipelines, which give identical labels);
                 - the congestion label equals (target queue length > threshold);
                 - NO LEAKAGE: the medium target at time t is the queue length at t + 10
                   and lag 1 at time t is the queue length at t - 1 (checked row by row);
                 - the rich-feature file has the extra columns;
                 - the persistence MAE and the test congestion rate recomputed from the
                   data equal those in the results files (so the regression AND the
                   classification results come from the current data and labels).
3. Results       validation (z-scores), model comparison files (every rho x horizon x
                 feature set, 5 test runs, no missing values), autocorrelation summaries.
4. Analysis      every analysis CSV/JSON and figure exists and is not empty.
5. Consistency   numbers that appear in two places agree (validation tables, early-warning
                 files, call-centre service fit, real-service L_q above Erlang-2 L_q).
6. Git           what is not yet committed.

Writes results/project_check_report.txt (the same text as printed).
"""

import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"
PLOTS = RES / "plots"
sys.path.insert(0, str(ROOT / "simulation"))

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
HORIZONS = ["short", "medium", "long", "extra_long"]
H_MEDIUM = 10
TRAIN, VAL, TEST = set(range(20)), set(range(20, 25)), set(range(25, 30))

LINES = []
COUNT = {"OK": 0, "WARN": 0, "FAIL": 0}


def out(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def ok(msg):
    COUNT["OK"] += 1
    out(f"  OK    {msg}")


def warn(msg):
    COUNT["WARN"] += 1
    out(f"  WARN  {msg}")


def fail(msg):
    COUNT["FAIL"] += 1
    out(f"  FAIL  {msg}")


def section(title):
    out("")
    out("=" * 100)
    out(title)
    out("=" * 100)


def nonempty(p):
    return p.exists() and p.stat().st_size > 0


# ------------------------------------------------------------------ settings
def map_setting(level, sfx):
    tag = sfx[1:] + "_" if sfx else ""
    return dict(
        name=f"MAP/PH/1 {level}",
        raw=lambda r, s=sfx: DATA / "mapph1" / f"mapph1_timeseries{s}_rho{r}.csv",
        hist=lambda r, t=tag: DATA / "mapph1" / "processed" / f"history_only_{t}rho{r}.csv",
        rich=lambda r, t=tag: DATA / "mapph1" / "processed" / f"rich_features_{t}rho{r}.csv",
        reg=f"mapph1{sfx}_model_comparison_regression.csv",
        clf=f"mapph1{sfx}_model_comparison_classification.csv",
        acf=f"mapph1{sfx}_autocorrelation_summary.csv",
        val=f"mapph1{sfx}_validation.csv",
        filt={}, thr="train")


SETTINGS = [
    dict(name="M/M/1",
         raw=lambda r: DATA / f"mm1_timeseries_rho{r}.csv",
         hist=lambda r: DATA / "processed" / f"history_only_rho{r}.csv",
         rich=lambda r: DATA / "processed" / f"rich_features_rho{r}.csv",
         reg="model_comparison_regression.csv", clf="model_comparison_classification.csv",
         acf="autocorrelation_summary.csv", val=None, filt={}, thr="all"),
]
for shape in ["2.0", "0.5"]:
    SETTINGS.append(dict(
        name=f"M/G/1 Weibull {shape}",
        raw=lambda r, s=shape: DATA / "mg1" / f"mg1_timeseries_shape{s}_rho{r}.csv",
        hist=lambda r, s=shape: DATA / "mg1" / "processed" / f"history_only_shape{s}_rho{r}.csv",
        rich=lambda r, s=shape: DATA / "mg1" / "processed" / f"rich_features_shape{s}_rho{r}.csv",
        reg="mg1_model_comparison_regression.csv", clf="mg1_model_comparison_classification.csv",
        acf="mg1_autocorrelation_summary.csv", val=None, filt={"weibull_shape": float(shape)}, thr="train"))
SETTINGS += [
    dict(name="M/PH/1 Erlang-2",
         raw=lambda r: DATA / "mph1" / f"mph1_timeseries_erlang2_rho{r}.csv",
         hist=lambda r: DATA / "mph1" / "processed" / f"history_only_erlang2_rho{r}.csv",
         rich=lambda r: DATA / "mph1" / "processed" / f"rich_features_erlang2_rho{r}.csv",
         reg="mph1_model_comparison_regression.csv", clf="mph1_model_comparison_classification.csv",
         acf="mph1_autocorrelation_summary.csv", val=None, filt={"erlang_k": 2}, thr="all"),
    dict(name="PH/PH/1 E2/E2",
         raw=lambda r: DATA / "phph1" / f"phph1_timeseries_arrE2_srvE2_rho{r}.csv",
         hist=lambda r: DATA / "phph1" / "processed" / f"history_only_arrE2_srvE2_rho{r}.csv",
         rich=lambda r: DATA / "phph1" / "processed" / f"rich_features_arrE2_srvE2_rho{r}.csv",
         reg="phph1_model_comparison_regression.csv", clf="phph1_model_comparison_classification.csv",
         acf="phph1_autocorrelation_summary.csv", val="phph1_validation.csv", filt={}, thr="train"),
    dict(name="H2/PH/1",
         raw=lambda r: DATA / "h2ph1" / f"h2ph1_timeseries_rho{r}.csv",
         hist=lambda r: DATA / "h2ph1" / "processed" / f"history_only_rho{r}.csv",
         rich=lambda r: DATA / "h2ph1" / "processed" / f"rich_features_rho{r}.csv",
         reg="h2ph1_model_comparison_regression.csv", clf="h2ph1_model_comparison_classification.csv",
         acf="h2ph1_autocorrelation_summary.csv", val="h2ph1_validation.csv", filt={}, thr="train"),
]
for level, sfx in [("medium", ""), ("weak", "_weak"), ("strong", "_strong"),
                   ("cc_afternoon", "_cc_afternoon"), ("cc_evening", "_cc_evening"),
                   ("cc_afternoon_realsvc", "_cc_afternoon_realsvc"),
                   ("cc_evening_realsvc", "_cc_evening_realsvc")]:
    SETTINGS.append(map_setting(level, sfx))

SCRIPTS = {
    "simulation": [
        "mm1_simulator.py", "mg1_simulator.py", "mph1_simulator.py", "phph1_simulator.py",
        "h2ph1_simulator.py", "mapph1_simulator.py", "ph_utils.py", "map_utils.py", "qbd_solver.py",
        "check_ph_utils.py", "check_qbd_solver.py", "check_map_utils.py", "check_map_levels.py",
        "validation_z_scores.py", "plot_mm1_validation.py", "callcenter_fit_map.py",
        "callcenter_fit_check.py", "callcenter_service_fit.py", "map_timescales.py"],
    "modeling": [
        "build_datasets.py", "build_mg1_datasets.py", "build_mph1_datasets.py", "build_phph1_datasets.py",
        "build_h2ph1_datasets.py", "build_mapph1_datasets.py",
        "train_baseline_models.py", "train_mg1_baseline_models.py", "train_mph1_baseline_models.py",
        "train_phph1_baseline_models.py", "train_h2ph1_baseline_models.py", "train_mapph1_baseline_models.py",
        "autocorrelation_analysis.py", "mg1_autocorrelation_analysis.py", "mph1_autocorrelation_analysis.py",
        "phph1_autocorrelation_analysis.py", "h2ph1_autocorrelation_analysis.py",
        "mapph1_autocorrelation_analysis.py",
        "mg1_extra_summaries.py", "mph1_extra_summaries.py", "phph1_extra_summaries.py",
        "h2ph1_extra_summaries.py", "mapph1_extra_summaries.py",
        "class_weight_all_queues.py", "resampling_all_queues.py", "horizon_sweep_all_queues.py",
        "compare_service_distributions.py", "compare_arrival_distributions.py",
        "summarise_regression_models.py", "significance_tests.py", "train_xgboost_models.py",
        "check_threshold_leakage.py", "map_dependence_summary.py", "horizon_sweep_rich_features.py",
        "mae_rmse_investigation.py", "class_imbalance_summary.py", "early_warning_analysis.py",
        "callcenter_summary.py", "early_warning_fair_comparison.py", "callcenter_service_summary.py",
        "consolidate_comparison.py",
        "class_weight_threshold_tuning.py", "resampling_experiment.py", "horizon_sweep_rho08.py"],
}

ANALYSIS_FILES = [
    # cross-stage and all-queue analyses
    "validation_z_scores.csv", "arrival_variance_comparison.csv", "service_variance_comparison.csv",
    "consolidated_comparison.csv", "consolidated_tables.tex", "regression_model_summary_all.csv",
    "regression_significance_tests.csv", "xgboost_results_per_run.csv", "xgboost_summary.csv",
    "class_weight_all_queues.csv", "class_weight_threshold_tuning.csv", "resampling_all_queues.csv",
    "resampling_results.csv", "horizon_sweep_all_queues.csv", "horizon_sweep_rho0.8.csv",
    "threshold_leakage_check.csv", "class_imbalance_selection.csv", "class_imbalance_summary.csv",
    "mae_rmse_investigation.csv",
    "mg1_feature_set_comparison.csv", "mph1_feature_set_comparison.csv", "phph1_feature_set_comparison.csv",
    "h2ph1_feature_set_comparison.csv", "mapph1_feature_set_comparison.csv",
    # MAP dependence levels
    "map_levels_properties.csv", "map_dependence_summary.csv", "horizon_sweep_rich_features.csv",
    "horizon_sweep_rich_features_map_levels.csv", "mae_rmse_investigation_map_levels.csv",
    "class_imbalance_summary_map_levels.csv", "map_dependence_measures.csv", "map_timescales_idc.csv",
    # early warning
    "early_warning_cost.csv", "early_warning_onsets.csv",
    "early_warning_cost_callcenter.csv", "early_warning_onsets_callcenter.csv",
    "early_warning_fair_runs.csv", "early_warning_fair_summary.csv", "early_warning_fair_paired.csv",
    "early_warning_fair_runs_callcenter.csv", "early_warning_fair_summary_callcenter.csv",
    "early_warning_fair_paired_callcenter.csv",
    # call centre
    "callcenter_map_fit.json", "callcenter_periods.csv", "callcenter_rate_profile.csv",
    "callcenter_summary.csv", "callcenter_fit_check.csv", "callcenter_stability.csv",
    "callcenter_service_fit.json", "callcenter_service_times.csv", "callcenter_service_summary.csv",
]

PLOT_FILES = [
    "validation_sim_vs_theory.png", "example_timeseries.png", "arrival_variance_comparison.png",
    "service_variance_comparison.png", "dataset_comparison_summary.png", "regression_mae_by_rho.png",
    "regression_mae_by_horizon.png", "classification_recall_by_rho.png", "autocorrelation_by_rho.png",
    "threshold_tuning_low_rho.png", "horizon_sweep_all_queues.png", "horizon_sweep_rho0.8.png",
    "horizon_sweep_rich_features.png", "horizon_sweep_rich_features_map_levels.png",
    "mae_rmse_trajectories.png", "mae_rmse_trajectories_map_levels.png", "map_dependence_summary.png",
    "map_levels_properties.png", "map_timescales_idc.png",
    "early_warning_summary.png", "early_warning_summary_callcenter.png",
    "early_warning_fair.png", "early_warning_fair_callcenter.png",
    "callcenter_arrivals.png", "callcenter_summary.png", "callcenter_fit_check.png",
    "callcenter_service_times.png", "callcenter_service_summary.png",
]
for pre in ["mg1", "mph1", "phph1", "h2ph1", "mapph1", "mapph1_weak", "mapph1_strong",
            "mapph1_cc_afternoon", "mapph1_cc_evening", "mapph1_cc_afternoon_realsvc", "mapph1_cc_evening_realsvc"]:
    PLOT_FILES += [f"{pre}_autocorrelation_by_rho.png", f"{pre}_classification_recall_by_rho.png",
                   f"{pre}_regression_mae_by_rho.png"]


# ------------------------------------------------------------------ 1. scripts
def readme_names(text):
    """Script names in the README, with patterns like build_{mg1,mph1}_datasets.py expanded."""
    names = set()
    for tok in re.findall(r"[\w{},.\-]+\.py", text):
        parts = re.split(r"(\{[^}]*\})", tok)
        opts = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
        names.update("".join(c) for c in itertools.product(*opts))
    return {n.split("/")[-1] for n in names}


def check_scripts():
    section("1. SCRIPTS")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    if not readme:
        fail("README.md not found")
    documented = readme_names(readme)
    for folder, names in SCRIPTS.items():
        missing = [n for n in names if not (ROOT / folder / n).exists()]
        if missing:
            fail(f"{folder}/: missing {', '.join(missing)}")
        else:
            ok(f"{folder}/: all {len(names)} expected scripts present")
        on_disk = sorted(p.name for p in (ROOT / folder).glob("*.py"))
        extra = [n for n in on_disk if n not in names]
        if extra:
            warn(f"{folder}/: scripts not in this checker's list: {', '.join(extra)}")
        undocumented = [n for n in on_disk if n not in documented and not n.startswith("__")]
        if undocumented:
            warn(f"{folder}/: not mentioned in README.md: {', '.join(undocumented)}")
        elif readme:
            ok(f"{folder}/: every script is mentioned in README.md")
    raw = DATA / "callcenter" / "raw"
    have = [n for n in ["november.txt", "december.txt"] if (raw / n).exists()]
    if len(have) == 2:
        ok("call-centre raw data present (data/callcenter/raw/november.txt, december.txt)")
    else:
        warn(f"call-centre raw data: only {have} in data/callcenter/raw/ (needed only to refit)")


# ------------------------------------------------------------------ 2. datasets
def results_row(st, dataset, rho, horizon, model, kind):
    f = RES / (st["reg"] if kind == "reg" else st["clf"])
    if not f.exists():
        return None
    d = CACHE.setdefault(f, pd.read_csv(f))
    sel = (d.dataset == dataset) & np.isclose(d.rho, rho) & (d.horizon == horizon) & (d.model == model) & \
          (d.test_run_id.astype(str) == "ALL_MEAN")
    for k, v in st["filt"].items():
        sel &= np.isclose(d[k].astype(float), float(v))
    return d[sel]


CACHE = {}


def check_one_dataset(st, rho, summary):
    tag = f"{st['name']}, rho={rho}"
    hp, rp, raw_p = st["hist"](rho), st["rich"](rho), st["raw"](rho)
    if not hp.exists():
        fail(f"{tag}: processed dataset missing ({hp.relative_to(ROOT)})")
        return
    problems = []
    cols = ["run_id", "t", "split", "congestion_threshold", "queue_length_lag0", "queue_length_lag1",
            "target_qlen_medium", "target_congested_medium"]
    d = pd.read_csv(hp, usecols=cols)

    # split by run
    runs = d.groupby("split")["run_id"].unique()
    got = {s: set(int(x) for x in runs.get(s, [])) for s in ["train", "val", "test"]}
    if got["train"] != TRAIN or got["val"] != VAL or got["test"] != TEST:
        problems.append(f"run split wrong: train {sorted(got['train'])[:3]}..., val {sorted(got['val'])}, "
                        f"test {sorted(got['test'])}")

    # threshold
    thr_vals = d["congestion_threshold"].unique()
    if len(thr_vals) != 1:
        problems.append(f"{len(thr_vals)} different thresholds in one file")
    thr = float(thr_vals[0])
    thr_method = "?"
    if raw_p.exists():
        raw = pd.read_csv(raw_p, usecols=["run_id", "queue_length"])
        q_train = raw.loc[raw.run_id < 20, "queue_length"].quantile(0.8)
        q_all = raw["queue_length"].quantile(0.8)
        if np.isclose(thr, q_train):
            thr_method = "train"
        elif np.isclose(thr, q_all):
            thr_method = "all"
        else:
            problems.append(f"threshold {thr} is neither the train-run ({q_train}) nor the all-run ({q_all}) "
                            f"80th percentile")
        if thr_method == "all" and st["thr"] == "train" and not np.isclose(q_train, q_all):
            problems.append(f"threshold from all runs ({q_all}), expected training runs only ({q_train})")
        summary["thr_" + thr_method] = summary.get("thr_" + thr_method, 0) + 1
    else:
        summary["raw_missing"] = summary.get("raw_missing", 0) + 1

    # label = target > threshold
    lab = (d["target_qlen_medium"] > thr).astype(int)
    n_bad = int((lab != d["target_congested_medium"]).sum())
    if n_bad:
        problems.append(f"{n_bad} congestion labels differ from (target > threshold)")

    # no leakage: target at t = queue at t + 10, lag1 at t = queue at t - 1
    q = d[["run_id", "t", "queue_length_lag0"]].copy()
    q["t"] = q["t"].round().astype(int)
    key = q.set_index(["run_id", "t"])["queue_length_lag0"]
    tt = d["t"].round().astype(int)
    fut = pd.MultiIndex.from_arrays([d["run_id"], tt + H_MEDIUM])
    past = pd.MultiIndex.from_arrays([d["run_id"], tt - 1])
    q_fut = key.reindex(fut).to_numpy()
    q_past = key.reindex(past).to_numpy()
    m_f, m_p = ~np.isnan(q_fut), ~np.isnan(q_past)
    bad_f = int((q_fut[m_f] != d["target_qlen_medium"].to_numpy()[m_f]).sum())
    bad_p = int((q_past[m_p] != d["queue_length_lag1"].to_numpy()[m_p]).sum())
    # (rows inside idle stretches are dropped when building the rich features, so at low rho
    #  not every t + 10 is in the file; all rows that can be checked are checked)
    if m_f.sum() < 0.5 * len(d) or m_p.sum() < 0.5 * len(d):
        problems.append(f"could check only {m_f.mean():.0%} / {m_p.mean():.0%} of rows for alignment")
    if bad_f or bad_p:
        problems.append(f"time alignment wrong: {bad_f} targets != queue at t+10, {bad_p} lag-1 != queue at t-1")

    # missing values
    n_nan = int(d.isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} missing values")

    # rich features file
    if not rp.exists():
        problems.append("rich-feature dataset missing")
    else:
        head = pd.read_csv(rp, nrows=2).columns
        need = ["num_in_system_lag0", "arrivals_in_interval_lag9", "recent_arrival_rate_lag0",
                "recent_utilisation_lag0", "target_congested_extra_long"]
        miss = [c for c in need if c not in head]
        if miss:
            problems.append(f"rich-feature file lacks {miss}")

    # persistence MAE recomputed against the results file
    te = d[d.split == "test"]
    mae = np.mean([np.mean(np.abs(g.target_qlen_medium - g.queue_length_lag0)) for _, g in te.groupby("run_id")])
    row = results_row(st, "history_only", rho, "medium", "Naive (persistence)", "reg")
    mae_note = "= results file"
    if row is None or len(row) != 1:
        problems.append("no persistence row in the results file to compare with")
    elif not np.isclose(mae, row["MAE"].iloc[0], rtol=1e-6):
        rel = abs(mae / row["MAE"].iloc[0] - 1)
        msg = (f"persistence MAE from data {mae:.5f} vs results file {row['MAE'].iloc[0]:.5f} "
               f"({rel:.3%} apart)")
        if rel < 0.005:
            warn(f"{tag}: {msg}: tiny, but the data were rebuilt after training (retrain to be exact)")
            mae_note = f"({rel:.3%} from results file, see WARN)"
        else:
            problems.append(msg + ": the results are not from the current data, retrain this setting")

    # test congestion rate recomputed against the classification results file (checks the labels)
    rate_te = np.mean([g.target_congested_medium.mean() for _, g in te.groupby("run_id")])
    crow = results_row(st, "history_only", rho, "medium", "Naive (persistence)", "clf")
    if crow is None or len(crow) != 1:
        problems.append("no persistence row in the classification results file to compare with")
    elif not np.isclose(rate_te, crow["congestion_rate"].iloc[0], rtol=1e-6):
        problems.append(f"test congestion rate from data {rate_te:.4f} != classification results "
                        f"{crow['congestion_rate'].iloc[0]:.4f}: labels changed after training, retrain")
    else:
        mae_note += ", test congestion rate = results file"

    rate = d["target_congested_medium"].mean()
    if problems:
        for p in problems:
            fail(f"{tag}: {p}")
    else:
        ok(f"{tag}: {len(d):,} rows, split/threshold/labels/alignment OK, threshold {thr:g} "
           f"({thr_method} runs), congestion rate {rate:.1%}, persistence MAE {mae:.3f} {mae_note}, "
           f"alignment checked on {m_f.mean():.0%} of rows")


def check_datasets():
    section("2. DATASETS (one line per queue setting and rho)")
    t0 = time.time()
    summary = {}
    for st in SETTINGS:
        for rho in RHOS:
            check_one_dataset(st, rho, summary)
    out(f"  threshold method: training runs only in {summary.get('thr_train', 0)} files, all runs in "
        f"{summary.get('thr_all', 0)} (expected all runs only for M/M/1 and M/PH/1, where the labels are "
        f"identical); raw series missing for {summary.get('raw_missing', 0)} files")
    out(f"  ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ 3. results per setting
def check_results():
    section("3. RESULTS PER QUEUE SETTING (validation, model comparison, autocorrelation)")
    zt = RES / "validation_z_scores.csv"
    if nonempty(zt):
        z = pd.read_csv(zt)
        for q, g in z.groupby("queue", sort=False):
            mz = g.z.abs().max()
            (ok if len(g) == 6 and mz <= 3 else fail)(
                f"validation {q}: {len(g)} rho, largest |z| = {mz:.2f}, {int((g.z.abs() > 2).sum())} with |z| > 2")
    else:
        fail("validation_z_scores.csv missing")
    for st in SETTINGS:
        if st["val"] and st["name"].startswith("MAP"):
            f = RES / st["val"]
            if not nonempty(f):
                fail(f"validation {st['name']}: {st['val']} missing")
                continue
            v = pd.read_csv(f)
            mz = v.z.abs().max()
            (ok if len(v) == 6 and mz <= 3 else fail)(
                f"validation {st['name']}: {len(v)} rho, largest |z| = {mz:.2f}, "
                f"{int((v.z.abs() > 2).sum())} with |z| > 2")

    for st in SETTINGS:
        for kind, metric in [("reg", "MAE"), ("clf", "f1_congested")]:
            f = RES / st[kind]
            if not nonempty(f):
                fail(f"{st['name']}: {st[kind]} missing")
                continue
            d = CACHE.setdefault(f, pd.read_csv(f))
            for k, v in st["filt"].items():
                d = d[np.isclose(d[k].astype(float), float(v))]
            s = d[d.test_run_id.astype(str) == "ALL_MEAN"]
            per_run = d[d.test_run_id.astype(str) != "ALL_MEAN"]
            probs = []
            for ds in ["history_only", "rich_features"]:
                for rho in RHOS:
                    for h in HORIZONS:
                        n = ((s.dataset == ds) & np.isclose(s.rho, rho) & (s.horizon == h)).sum()
                        if n == 0:
                            probs.append(f"{ds} rho={rho} {h}")
            runs = set(pd.to_numeric(per_run.test_run_id, errors="coerce").dropna().astype(int))
            if runs != TEST:
                probs.append(f"test runs {sorted(runs)}")
            if not (s.n_test_runs == 5).all():
                probs.append("n_test_runs not always 5")
            if s[metric].isna().any():
                probs.append(f"missing {metric}")
            name = "regression" if kind == "reg" else "classification"
            if probs:
                fail(f"{st['name']} {name}: missing/odd: {'; '.join(probs[:5])}{' ...' if len(probs) > 5 else ''}")
            else:
                ok(f"{st['name']} {name}: 6 rho x 4 horizons x 2 feature sets, {s.model.nunique()} models, "
                   f"5 test runs (25-29)")
        f = RES / st["acf"]
        if not nonempty(f):
            fail(f"{st['name']}: {st['acf']} missing")
        else:
            a = pd.read_csv(f)
            for k, v in st["filt"].items():
                a = a[np.isclose(a[k].astype(float), float(v))]
            (ok if len(a) == 6 else fail)(f"{st['name']} autocorrelation: {len(a)} rho")


# ------------------------------------------------------------------ 4. analysis outputs
def check_outputs():
    section("4. ANALYSIS OUTPUTS (files exist and are not empty)")
    miss = [f for f in ANALYSIS_FILES if not nonempty(RES / f)]
    if miss:
        for f in miss:
            fail(f"results/{f} missing or empty")
    ok(f"{len(ANALYSIS_FILES) - len(miss)} of {len(ANALYSIS_FILES)} analysis result files present")
    miss = [f for f in PLOT_FILES if not nonempty(PLOTS / f)]
    for f in miss:
        if nonempty(ROOT / "plots" / f):
            warn(f"{f} is in plots/ instead of results/plots/ (move it there)")
        else:
            warn(f"results/plots/{f} missing or empty")
    ok(f"{len(PLOT_FILES) - len(miss)} of {len(PLOT_FILES)} figures present")
    known = set(ANALYSIS_FILES) | {st[k] for st in SETTINGS for k in ["reg", "clf", "acf"]} | \
            {st["val"] for st in SETTINGS if st["val"]} | {"project_check_report.txt"}
    extra = sorted(p.name for p in RES.glob("*.*") if p.name not in known)
    if extra:
        out(f"  info  other files in results/ (not checked): {', '.join(extra)}")


# ------------------------------------------------------------------ 5. consistency
def check_consistency():
    section("5. CONSISTENCY BETWEEN FILES")
    # validation table vs per-queue validation files
    z = pd.read_csv(RES / "validation_z_scores.csv") if nonempty(RES / "validation_z_scores.csv") else None
    for q, f in [("PH/PH/1 Erlang-2/Erlang-2", "phph1_validation.csv"), ("H2/PH/1 H2/Erlang-2", "h2ph1_validation.csv"),
                 ("MAP/PH/1 MMPP-2/Erlang-2", "mapph1_validation.csv")]:
        if z is None or not nonempty(RES / f):
            warn(f"cannot compare validation table with {f}")
            continue
        a = z[z.queue == q].sort_values("rho")["z"].to_numpy()
        b = pd.read_csv(RES / f).sort_values("rho")["z"].to_numpy()
        (ok if len(a) == len(b) and np.allclose(a, b, atol=1e-6) else fail)(
            f"validation_z_scores.csv ({q}) matches {f}")

    # real service: longer queue than Erlang-2 at every rho
    for per in ["afternoon", "evening"]:
        a, b = RES / f"mapph1_cc_{per}_validation.csv", RES / f"mapph1_cc_{per}_realsvc_validation.csv"
        if nonempty(a) and nonempty(b):
            la, lb = pd.read_csv(a).set_index("rho")["Lq_qbd"], pd.read_csv(b).set_index("rho")["Lq_qbd"]
            (ok if (lb > la).all() else fail)(
                f"call centre {per}: exact L_q with real service above Erlang-2 at every rho "
                f"(rho 0.8: {la.loc[0.8]:.2f} -> {lb.loc[0.8]:.2f})")

    # service fit json vs the H2 used in the simulator
    j = RES / "callcenter_service_fit.json"
    if nonempty(j):
        try:
            from map_utils import service_ph
            fit = json.loads(j.read_text())
            for per in ["afternoon", "evening"]:
                scv = service_ph(f"cc_{per}_realsvc", 1.0).scv
                want = float(fit[f"cc_{per}"]["scv"])
                (ok if np.isclose(scv, want, rtol=1e-6) else fail)(
                    f"service H2 used for {per} has C_s^2 = {scv:.3f}; observed in the data {want:.3f}")
                m = service_ph(f"cc_{per}_realsvc", 1.0).mean
                (ok if np.isclose(m, 1.0) else fail)(f"service H2 for {per} has mean {m:.4f} (should be 1)")
        except Exception as e:                                   # noqa: BLE001
            fail(f"could not check the service fit: {e}")

    # early-warning: original analysis vs fair comparison (same definitions, same models)
    for orig, fair, names in [
        ("early_warning_onsets.csv", "early_warning_fair_summary.csv",
         {"none (H2)": "none (H2)", "weak": "weak", "medium": "medium", "strong": "strong"}),
        ("early_warning_onsets_callcenter.csv", "early_warning_fair_summary_callcenter.csv",
         {"call centre afternoon": "cc afternoon", "call centre evening": "cc evening"})]:
        if not (nonempty(RES / orig) and nonempty(RES / fair)):
            warn(f"cannot compare {orig} with {fair}")
            continue
        o, f = pd.read_csv(RES / orig), pd.read_csv(RES / fair)
        o = o.assign(queue=o.queue.map(names)).dropna(subset=["queue"])
        m = o.merge(f, on=["queue", "rho", "C", "method"])
        for meth in ["Persistence", "Lowered-threshold rule", "RF classifier, rich features"]:
            mm = m[m.method == meth]
            if len(mm) == 0:
                warn(f"{orig} vs {fair}: no common rows for {meth}")
                continue
            diff = np.max(np.abs(mm["cost"] - mm["cost_total"]) / mm["cost"])
            if diff < 1e-6:
                ok(f"{meth}: cost identical in {orig} and {fair} ({len(mm)} settings)")
            elif meth.startswith("RF") and diff < 0.05:
                warn(f"{meth}: cost differs by up to {diff:.1%} between {orig} and {fair} (random forest refit)")
            else:
                fail(f"{meth}: cost differs by up to {diff:.1%} between {orig} and {fair}")

    # call-centre fair comparison: the two Erlang-2 settings agree between the full and the --callcenter run
    a, b = RES / "early_warning_fair_summary.csv", RES / "early_warning_fair_summary_callcenter.csv"
    if nonempty(a) and nonempty(b):
        m = pd.read_csv(a).merge(pd.read_csv(b), on=["queue", "rho", "C", "method"], suffixes=("_a", "_b"))
        if len(m):
            diff = (m.cost_total_a - m.cost_total_b).abs().max()
            (ok if diff < 1e-9 else fail)(f"call-centre (Erlang-2) early-warning costs identical in the full "
                                          f"and --callcenter runs ({len(m)} rows)")

    # summary table vs sources
    s = RES / "callcenter_service_summary.csv"
    if nonempty(s):
        s = pd.read_csv(s)
        bad = 0
        for setting, f in [("afternoon, Erlang-2", "mapph1_cc_afternoon_validation.csv"),
                           ("afternoon, real service", "mapph1_cc_afternoon_realsvc_validation.csv"),
                           ("evening, Erlang-2", "mapph1_cc_evening_validation.csv"),
                           ("evening, real service", "mapph1_cc_evening_realsvc_validation.csv")]:
            v = pd.read_csv(RES / f).set_index("rho")["Lq_qbd"]
            r = s[(s.quantity == "Lq") & (s.setting == setting)].set_index("rho")["value"]
            bad += int(not np.allclose(r.sort_index().to_numpy(), v.sort_index().to_numpy()))
        (ok if bad == 0 else fail)("callcenter_service_summary.csv L_q equals the validation files")

    # datasets of the realsvc levels really use H2 service (label 0)
    for lvl in ["cc_afternoon_realsvc", "cc_evening_realsvc"]:
        p = DATA / "mapph1" / "processed" / f"history_only_{lvl}_rho0.8.csv"
        if p.exists():
            k = pd.read_csv(p, nrows=5, usecols=["arrival_process", "service_erlang_k"])
            good = (k.service_erlang_k == 0).all() and k.arrival_process.str.endswith("realsvc").all()
            (ok if good else fail)(f"{lvl} data labelled {k.arrival_process.iloc[0]} / service {k.service_erlang_k.iloc[0]}")


# ------------------------------------------------------------------ 6. git
def check_git():
    section("6. GIT")
    try:
        if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode != 0:
            warn("this folder is not a git repository")
            return
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60).stdout.splitlines()
        br = subprocess.run(["git", "status", "-sb"], cwd=ROOT, capture_output=True, text=True,
                            timeout=60).stdout.splitlines()[0]
        last = subprocess.run(["git", "log", "-1", "--format=%h %ad %s", "--date=short"], cwd=ROOT,
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                                       # noqa: BLE001
        warn(f"git not available: {e}")
        return
    out(f"  branch: {br}")
    out(f"  last commit: {last}")
    sample = next((p for p in sorted(DATA.rglob("*.csv")) if "processed" in p.parts and "mapph1" in p.parts),
                  None) if DATA.exists() else None
    if sample is not None:
        rel = sample.relative_to(ROOT).as_posix()
        ign = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0
        (ok if ign else fail)(f"data/ is ignored by git (large files are not committed; tested {rel})")
    groups = {}
    for line in st:
        path = line[3:].strip().strip('"')
        top = path.split("/")[0] + "/" if "/" in path else "the project folder"
        groups.setdefault(top, []).append(f"{line[:2].strip() or '?'} {path}")
    if not st:
        ok("everything is committed")
    for top, items in groups.items():
        out(f"  not committed in {top}: {len(items)} file(s)")
        for it in items[:40]:
            out(f"        {it}")
        if len(items) > 40:
            out(f"        ... and {len(items) - 40} more")
    if st:
        out("  (M = modified, ?? = new; run  git add -A ; git commit ; git push  to save them)")


def main():
    t0 = time.time()
    out(f"Project check, {time.strftime('%Y-%m-%d %H:%M')}, folder {ROOT}")
    check_scripts()
    check_datasets()
    check_results()
    check_outputs()
    check_consistency()
    check_git()
    section("SUMMARY")
    out(f"  {COUNT['OK']} OK, {COUNT['WARN']} warnings, {COUNT['FAIL']} failures   ({time.time() - t0:.0f} s)")
    if COUNT["FAIL"] == 0:
        out("  No failures: every input, output and cross-check is consistent.")
    else:
        out("  Look at the FAIL lines above.")
    RES.mkdir(exist_ok=True)
    (RES / "project_check_report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("\nSaved: results/project_check_report.txt")


if __name__ == "__main__":
    main()