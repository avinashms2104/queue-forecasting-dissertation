"""
Horizon sweep with BOTH feature sets: queue-history-only vs additional features
(Azam, email of 5 October 2026: "repeat the horizon sweep for M/PH/1 and PH/PH/1,
and include the persistence baseline and the comparison between
queue-history-only and additional-feature models").

Same design as horizon_sweep_all_queues.py: rho = 0.8, horizons h = 5, 10, ..., 40
time units, sampling interval dt = 1, Random Forest (30 trees, depth 6) trained on
the 20 training runs and scored on each of the 5 test runs, against persistence.
MAE is the primary metric (Azam's choice); RMSE is reported alongside.

Feature sets (identical to the build_*_datasets.py scripts):
    history-only    the last 10 queue lengths
    rich            the same, plus the last 10 values of number in system,
                    arrivals and departures per interval, recent arrival and
                    service rates, and the current recent-utilisation ratio
Both models are trained and scored on exactly the same rows (rows where every
rich feature is defined), so any difference is due to the features alone.

--map-levels runs the arrival-dependence comparison instead: H2/PH/1 (no
dependence) and MAP/PH/1 with weak, medium and strong bursts, all with the same
C_a^2 = 1.684 and the same Erlang-2 service. The rich features include recent
arrival counts and arrival rates, so this answers whether arrival-history
features help more when arrivals are correlated.

Usage
-----
    python modeling/horizon_sweep_rich_features.py               # M/M/1, M/PH/1, PH/PH/1
    python modeling/horizon_sweep_rich_features.py --all-queues  # all seven queue types
    python modeling/horizon_sweep_rich_features.py --map-levels  # H2 + MAP weak/medium/strong

Writes: results/horizon_sweep_rich_features{_map_levels}.csv
        results/plots/horizon_sweep_rich_features{_map_levels}.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

RHO = 0.8
LOOKBACK = 10
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40]
SEED = 42
N_TRAIN_RUNS = 20
TEST_RUN_IDS = [25, 26, 27, 28, 29]
RICH_BASE = ["num_in_system", "arrivals_in_interval", "departures_in_interval",
             "recent_arrival_rate", "recent_service_rate"]

MAIN_QUEUES = [
    ("M/M/1", DATA / f"mm1_timeseries_rho{RHO}.csv"),
    ("M/PH/1 Erlang-2", DATA / "mph1" / f"mph1_timeseries_erlang2_rho{RHO}.csv"),
    ("PH/PH/1 E2/E2", DATA / "phph1" / f"phph1_timeseries_arrE2_srvE2_rho{RHO}.csv"),
]
EXTRA_QUEUES = [
    ("M/G/1 Weibull 2.0", DATA / "mg1" / f"mg1_timeseries_shape2.0_rho{RHO}.csv"),
    ("M/G/1 Weibull 0.5", DATA / "mg1" / f"mg1_timeseries_shape0.5_rho{RHO}.csv"),
    ("H2/PH/1 H2/E2", DATA / "h2ph1" / f"h2ph1_timeseries_rho{RHO}.csv"),
    ("MAP/PH/1 MMPP2/E2", DATA / "mapph1" / f"mapph1_timeseries_rho{RHO}.csv"),
]

MAP_LEVEL_QUEUES = [
    ("H2 (no bursts)", DATA / "h2ph1" / f"h2ph1_timeseries_rho{RHO}.csv"),
    ("MAP weak", DATA / "mapph1" / f"mapph1_timeseries_weak_rho{RHO}.csv"),
    ("MAP medium", DATA / "mapph1" / f"mapph1_timeseries_rho{RHO}.csv"),
    ("MAP strong", DATA / "mapph1" / f"mapph1_timeseries_strong_rho{RHO}.csv"),
]

HIST_COLS = [f"queue_length_lag{k}" for k in range(LOOKBACK)]
RICH_COLS = HIST_COLS + [f"{c}_lag{k}" for c in RICH_BASE for k in range(LOOKBACK)] + ["recent_utilisation_lag0"]


def build_lags(raw):
    """Lagged features for every run (targets are added per horizon later)."""
    parts = []
    for run_id, g in raw.groupby("run_id"):
        g = g.sort_values("t").reset_index(drop=True)
        d = {"run_id": g["run_id"], "t": g["t"], "queue_length": g["queue_length"]}
        for k in range(LOOKBACK):
            d[f"queue_length_lag{k}"] = g["queue_length"].shift(k)
            for c in RICH_BASE:
                d[f"{c}_lag{k}"] = g[c].shift(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            d["recent_utilisation_lag0"] = g["recent_arrival_rate"] / g["recent_service_rate"].replace(0, np.nan)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def main(all_queues, map_levels=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    queues = MAP_LEVEL_QUEUES if map_levels else MAIN_QUEUES + (EXTRA_QUEUES if all_queues else [])
    tag = "_map_levels" if map_levels else ""
    rows = []
    for queue, path in queues:
        if not path.exists():
            print(f"(skipping {queue}: {path.name} not found)")
            continue
        raw = pd.read_csv(path, usecols=["run_id", "t", "queue_length"] + RICH_BASE)
        lags = build_lags(raw)
        for h in HORIZONS:
            df = lags.copy()
            df["target"] = df.groupby("run_id")["queue_length"].shift(-h)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=RICH_COLS + ["target"])
            tr = df[df.run_id < N_TRAIN_RUNS]
            models = {}
            for name, cols in [("RF history-only", HIST_COLS), ("RF rich features", RICH_COLS)]:
                m = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED, n_jobs=-1)
                m.fit(tr[cols], tr["target"])
                models[name] = (m, cols)
            for rid in TEST_RUN_IDS:
                te = df[df.run_id == rid]
                if len(te) == 0:
                    continue
                y = te["target"]
                preds = [("Persistence", te["queue_length_lag0"].values)]
                preds += [(name, m.predict(te[cols])) for name, (m, cols) in models.items()]
                for model, pred in preds:
                    rows.append(dict(queue=queue, rho=RHO, h=h, model=model, test_run_id=rid,
                                     MAE=mean_absolute_error(y, pred),
                                     RMSE=mean_squared_error(y, pred) ** 0.5))
            print(f"done: {queue} h={h}", flush=True)

    per_run = pd.DataFrame(rows)
    summ = (per_run.groupby(["queue", "h", "model"])
            .agg(MAE=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE=("RMSE", "mean"))
            .reset_index())
    pers = summ[summ.model == "Persistence"].set_index(["queue", "h"])[["MAE", "RMSE"]]
    summ = summ.join(pers, on=["queue", "h"], rsuffix="_persistence")
    summ["MAE_impr_pct"] = 100 * (1 - summ["MAE"] / summ["MAE_persistence"])
    summ["RMSE_impr_pct"] = 100 * (1 - summ["RMSE"] / summ["RMSE_persistence"])
    summ.insert(1, "rho", RHO)
    summ.to_csv(RESULTS / f"horizon_sweep_rich_features{tag}.csv", index=False)

    qlist = [q for q, _ in queues if q in set(summ.queue)]
    pd.set_option("display.width", 220)
    for metric in ["MAE", "RMSE"]:
        print("\n" + "=" * 110)
        print(f"{metric} improvement over persistence (%), rho = {RHO}: history-only vs rich features")
        print("=" * 110)
        t = summ[summ.model != "Persistence"].pivot_table(index="h", columns=["queue", "model"],
                                                           values=f"{metric}_impr_pct")
        print(t[qlist].round(1).to_string())

    print("\n" + "=" * 110)
    print("Rich minus history-only, MAE improvement over persistence (percentage points; positive = rich helps)")
    print("=" * 110)
    w = summ[summ.model != "Persistence"].pivot_table(index="h", columns=["queue", "model"], values="MAE_impr_pct")
    diff = pd.DataFrame({q: w[(q, "RF rich features")] - w[(q, "RF history-only")] for q in qlist})
    print(diff.round(2).to_string())

    fig, axes = plt.subplots(1, len(qlist), figsize=(4.6 * len(qlist), 4.2), sharey=True, squeeze=False)
    for ax, q in zip(axes[0], qlist):
        for model, style in [("RF history-only", "o-"), ("RF rich features", "s--")]:
            s = summ[(summ.queue == q) & (summ.model == model)].sort_values("h")
            ax.plot(s.h, s.MAE_impr_pct, style, label=model)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(q, fontsize=10)
        ax.set_xlabel("Forecast horizon h (time units)")
    axes[0, 0].set_ylabel("MAE improvement over persistence (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Horizon sweep at rho = {RHO}: queue-history-only vs additional features")
    fig.tight_layout()
    fig.savefig(PLOTS / f"horizon_sweep_rich_features{tag}.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: results/horizon_sweep_rich_features{tag}.csv")
    print(f"Saved: results/plots/horizon_sweep_rich_features{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-queues", action="store_true")
    ap.add_argument("--map-levels", action="store_true")
    a = ap.parse_args()
    main(a.all_queues, a.map_levels)