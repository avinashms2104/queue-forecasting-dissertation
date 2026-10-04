"""
Wider forecast-horizon sweep at rho = 0.8, repeated for ALL seven queue types
(the original horizon_sweep_rho08.py covered M/M/1 only; open question 3 to Azam).

Same design as the original: horizons h = 5, 10, ..., 40 time units, 10 lagged
queue lengths as inputs, Random Forest (30 trees, depth 6) trained on the 20
training runs, scored on each of the 5 test runs, against persistence. RMSE is
reported next to MAE because the two metrics disagree elsewhere in the project.

The question: as the horizon grows, does the model's advantage over persistence
keep growing, level off, or disappear, and is the answer the same for every
queue type? Each queue's raw time series at rho = 0.8 is read once and the
features are rebuilt for every horizon (so the targets are not limited to the
four horizons stored in the processed datasets).

Usage:  python modeling/horizon_sweep_all_queues.py
Writes: results/horizon_sweep_all_queues.csv
        results/plots/horizon_sweep_all_queues.png
"""

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
N_TRAIN_RUNS, N_VAL_RUNS = 20, 5
TEST_RUN_IDS = [25, 26, 27, 28, 29]

# (label, raw time-series file at rho = 0.8)
QUEUES = [
    ("M/M/1", DATA / f"mm1_timeseries_rho{RHO}.csv"),
    ("M/G/1 Weibull 2.0", DATA / "mg1" / f"mg1_timeseries_shape2.0_rho{RHO}.csv"),
    ("M/G/1 Weibull 0.5", DATA / "mg1" / f"mg1_timeseries_shape0.5_rho{RHO}.csv"),
    ("M/PH/1 Erlang-2", DATA / "mph1" / f"mph1_timeseries_erlang2_rho{RHO}.csv"),
    ("PH/PH/1 E2/E2", DATA / "phph1" / f"phph1_timeseries_arrE2_srvE2_rho{RHO}.csv"),
    ("H2/PH/1 H2/E2", DATA / "h2ph1" / f"h2ph1_timeseries_rho{RHO}.csv"),
    ("MAP/PH/1 MMPP2/E2", DATA / "mapph1" / f"mapph1_timeseries_rho{RHO}.csv"),
]
LAG_COLS = [f"lag{k}" for k in range(LOOKBACK)]


def split_of(run_id):
    if run_id < N_TRAIN_RUNS:
        return "train"
    return "val" if run_id < N_TRAIN_RUNS + N_VAL_RUNS else "test"


def build(raw, h):
    parts = []
    for run_id, g in raw.groupby("run_id"):
        g = g.sort_values("t").reset_index(drop=True)
        d = pd.DataFrame({f"lag{k}": g["queue_length"].shift(k) for k in range(LOOKBACK)})
        d["target"] = g["queue_length"].shift(-h)
        d = d.dropna().reset_index(drop=True)
        d["run_id"], d["split"] = run_id, split_of(run_id)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for queue, path in QUEUES:
        if not path.exists():
            print(f"(skipping {queue}: {path.name} not found)")
            continue
        raw = pd.read_csv(path, usecols=["run_id", "t", "queue_length"])
        for h in HORIZONS:
            df = build(raw, h)
            tr = df[df.split == "train"]
            rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED, n_jobs=-1)
            rf.fit(tr[LAG_COLS], tr["target"])
            for rid in TEST_RUN_IDS:
                te = df[(df.split == "test") & (df.run_id == rid)]
                if len(te) == 0:
                    continue
                y = te["target"]
                for model, pred in [("Random Forest", rf.predict(te[LAG_COLS])),
                                    ("Persistence", te["lag0"].values)]:
                    rows.append(dict(queue=queue, rho=RHO, h=h, model=model, test_run_id=rid,
                                     MAE=mean_absolute_error(y, pred),
                                     RMSE=mean_squared_error(y, pred) ** 0.5))
            print(f"done: {queue} h={h}", flush=True)

    per_run = pd.DataFrame(rows)
    summ = (per_run.groupby(["queue", "h", "model"])
            .agg(MAE=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE=("RMSE", "mean"))
            .reset_index())
    pers = summ[summ.model == "Persistence"].set_index(["queue", "h"])
    rf = summ[summ.model == "Random Forest"].set_index(["queue", "h"])
    out = pd.DataFrame({
        "RF_MAE": rf["MAE"], "RF_MAE_std": rf["MAE_std"], "persistence_MAE": pers["MAE"],
        "RF_RMSE": rf["RMSE"], "persistence_RMSE": pers["RMSE"],
    })
    out["MAE_rel_improvement"] = 1 - out["RF_MAE"] / out["persistence_MAE"]
    out["RMSE_rel_improvement"] = 1 - out["RF_RMSE"] / out["persistence_RMSE"]
    out = out.reset_index()
    out.insert(1, "rho", RHO)
    out.to_csv(RESULTS / "horizon_sweep_all_queues.csv", index=False)

    qlist = [q for q, _ in QUEUES if q in set(out.queue)]
    pd.set_option("display.width", 220)
    for metric in ["MAE", "RMSE"]:
        print("\n" + "=" * 110)
        print(f"Random Forest {metric} improvement over persistence (%), rho = {RHO}, by horizon h")
        print("=" * 110)
        t = out.pivot(index="h", columns="queue", values=f"{metric}_rel_improvement")[qlist]
        print((t * 100).round(1).to_string())

    print("\n" + "=" * 110)
    print(f"Persistence MAE by horizon (how fast the queue 'forgets'), rho = {RHO}")
    print("=" * 110)
    print(out.pivot(index="h", columns="queue", values="persistence_MAE")[qlist].round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for q in qlist:
        s = out[out.queue == q].sort_values("h")
        axes[0].plot(s.h, 100 * s.MAE_rel_improvement, "o-", label=q)
        axes[1].plot(s.h, 100 * s.RMSE_rel_improvement, "o-", label=q)
    for ax, m in zip(axes, ["MAE", "RMSE"]):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Forecast horizon h (time units)")
        ax.set_ylabel(f"{m} improvement over persistence (%)")
        ax.set_title(f"Random Forest vs persistence, {m}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Horizon sweep at rho = {RHO}, all queue types (history-only features)")
    fig.tight_layout()
    fig.savefig(PLOTS / "horizon_sweep_all_queues.png", dpi=130)
    plt.close(fig)
    print("\nSaved: results/horizon_sweep_all_queues.csv")
    print("Saved: results/plots/horizon_sweep_all_queues.png")


if __name__ == "__main__":
    main()