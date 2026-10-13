"""
Where and why do MAE and RMSE disagree? (Azam, email of 5 October 2026)

Across the queues, Random Forest beats persistence clearly on RMSE but only
slightly (or not at all) on MAE. This script looks inside the errors to see
whether the difference comes from occasional large errors, whether those
happen during congestion, and whether they are under- or over-predictions.

Setting: medium horizon (h = 10), history-only features, Random Forest
(30 trees, depth 6, as in train_*_baseline_models.py) vs naive persistence.

Runs used: the 5 VALIDATION runs (20-24). This is a diagnostic, and Azam asked
that the final test runs stay separate from any decision about next steps, so
the test runs are not touched here.

For each queue and rho, for each method, it reports:
    MAE, RMSE
    MSE and MAE split ADDITIVELY into the part from congested target times
        (actual queue above the congestion threshold) and the part from the rest,
        so the persistence - RF gap can be traced to one or the other
    share of the squared error coming from over-predictions (prediction > actual)
    share of the squared error coming from the largest 5% of errors
    during congestion: mean signed error and fraction of under-predictions
    fraction of predictions within 0.5 of the actual value
and writes trajectory plots (actual vs both predictions) for one validation run
at one low and one high rho, choosing the window with the largest persistence
error so that the plot shows where the two metrics disagree.

Usage
-----
    python modeling/mae_rmse_investigation.py               # M/M/1, M/PH/1, PH/PH/1
    python modeling/mae_rmse_investigation.py --all-queues  # all seven queue types
    python modeling/mae_rmse_investigation.py --map-levels  # H2 + MAP weak/medium/strong

Writes: results/mae_rmse_investigation{_map_levels}.csv
        results/plots/mae_rmse_trajectories{_map_levels}.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
PLOT_RHOS = [0.5, 0.9]                  # one low, one high utilisation for the trajectories
PLOT_RUN = 20                           # first validation run
WINDOW = 200                            # time units shown in each trajectory panel
HORIZON = "medium"                      # h = 10
SEED = 42
VAL_RUN_IDS = [20, 21, 22, 23, 24]

MAIN_QUEUES = [
    ("M/M/1", lambda r: DATA / "processed" / f"history_only_rho{r}.csv"),
    ("M/PH/1 Erlang-2", lambda r: DATA / "mph1" / "processed" / f"history_only_erlang2_rho{r}.csv"),
    ("PH/PH/1 E2/E2", lambda r: DATA / "phph1" / "processed" / f"history_only_arrE2_srvE2_rho{r}.csv"),
]
EXTRA_QUEUES = [
    ("M/G/1 Weibull 2.0", lambda r: DATA / "mg1" / "processed" / f"history_only_shape2.0_rho{r}.csv"),
    ("M/G/1 Weibull 0.5", lambda r: DATA / "mg1" / "processed" / f"history_only_shape0.5_rho{r}.csv"),
    ("H2/PH/1 H2/E2", lambda r: DATA / "h2ph1" / "processed" / f"history_only_rho{r}.csv"),
    ("MAP/PH/1 MMPP2/E2", lambda r: DATA / "mapph1" / "processed" / f"history_only_rho{r}.csv"),
]


MAP_LEVEL_QUEUES = [
    ("H2 (no bursts)", lambda r: DATA / "h2ph1" / "processed" / f"history_only_rho{r}.csv"),
    ("MAP weak", lambda r: DATA / "mapph1" / "processed" / f"history_only_weak_rho{r}.csv"),
    ("MAP medium", lambda r: DATA / "mapph1" / "processed" / f"history_only_rho{r}.csv"),
    ("MAP strong", lambda r: DATA / "mapph1" / "processed" / f"history_only_strong_rho{r}.csv"),
]


def error_profile(err, congested):
    """Summary of one method's errors (err = prediction - actual) over the validation rows."""
    a = np.abs(err)
    sq = err ** 2
    total_sq = sq.sum()
    order = np.sort(sq)[::-1]
    n = len(err)
    top1 = order[: max(1, int(0.01 * n))].sum() / total_sq if total_sq > 0 else np.nan
    top5 = order[: max(1, int(0.05 * n))].sum() / total_sq if total_sq > 0 else np.nan
    c, nc = congested, ~congested
    return dict(
        MAE=a.mean(), RMSE=np.sqrt(sq.mean()), MSE=sq.mean(),
        MSE_from_congested=sq[c].sum() / n, MSE_from_not_congested=sq[nc].sum() / n,
        MAE_from_congested=a[c].sum() / n, MAE_from_not_congested=a[nc].sum() / n,
        sq_share_over=sq[err > 0].sum() / total_sq if total_sq > 0 else np.nan,
        sq_share_top1pct=top1, sq_share_top5pct=top5,
        mean_signed_error=err.mean(),
        mean_signed_error_congested=err[c].mean() if c.any() else np.nan,
        frac_under_congested=(err[c] < 0).mean() if c.any() else np.nan,
        frac_under_not_congested=(err[nc] < 0).mean() if nc.any() else np.nan,
        frac_exact=(a < 0.5).mean(),
    )


def run_setting(queue, rho, path):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c.startswith("queue_length_lag")]
    target = f"target_qlen_{HORIZON}"
    train = df[df["split"] == "train"]
    val = df[df["run_id"].isin(VAL_RUN_IDS)].sort_values(["run_id", "t"]).copy()

    rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED, n_jobs=-1)
    rf.fit(train[feat], train[target])
    val["pred_rf"] = rf.predict(val[feat])
    val["pred_persistence"] = val["queue_length_lag0"]
    y = val[target].to_numpy()
    congested = y > val["congestion_threshold"].to_numpy()

    rows = []
    for method, col in [("Persistence", "pred_persistence"), ("Random Forest", "pred_rf")]:
        prof = error_profile(val[col].to_numpy() - y, congested)
        rows.append(dict(queue=queue, rho=rho, method=method,
                         congestion_rate=congested.mean(), n=len(y), **prof))
    return rows, val


def plot_trajectories(store, queues, tag=""):
    fig, axes = plt.subplots(len(queues), len(PLOT_RHOS), figsize=(13, 3.2 * len(queues)), squeeze=False)
    for i, q in enumerate(queues):
        for j, rho in enumerate(PLOT_RHOS):
            ax = axes[i, j]
            if (q, rho) not in store:
                ax.set_visible(False)
                continue
            v = store[(q, rho)]
            v = v[v["run_id"] == PLOT_RUN].reset_index(drop=True)
            y = v[f"target_qlen_{HORIZON}"].to_numpy()
            # window centred on the largest persistence error, to show where the metrics disagree
            k = int(np.argmax(np.abs(v["pred_persistence"].to_numpy() - y)))
            lo = max(0, min(k - WINDOW // 2, len(v) - WINDOW))
            w = v.iloc[lo:lo + WINDOW]
            t_target = w["t"].to_numpy() + 10        # the time being predicted
            ax.plot(t_target, w[f"target_qlen_{HORIZON}"], color="black", linewidth=1.4, label="Actual")
            ax.plot(t_target, w["pred_persistence"], color="tab:orange", linewidth=1, label="Persistence")
            ax.plot(t_target, w["pred_rf"], color="tab:blue", linewidth=1, label="Random Forest")
            ax.axhline(w["congestion_threshold"].iloc[0], color="grey", linestyle=":", linewidth=0.9,
                       label="Congestion threshold")
            ax.set_title(f"{q}, rho = {rho} (validation run {PLOT_RUN})", fontsize=9)
            ax.set_xlabel("Time being predicted")
            ax.set_ylabel("Queue length")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Actual vs predicted queue length, 10 time units ahead (history-only features)")
    fig.tight_layout()
    fig.savefig(PLOTS / f"mae_rmse_trajectories{tag}.png", dpi=130)
    plt.close(fig)


def main(all_queues, map_levels=False):
    PLOTS.mkdir(parents=True, exist_ok=True)
    queues = MAP_LEVEL_QUEUES if map_levels else MAIN_QUEUES + (EXTRA_QUEUES if all_queues else [])
    tag = "_map_levels" if map_levels else ""
    rows, store = [], {}
    for queue, path_fn in queues:
        if not path_fn(RHOS[0]).exists():
            print(f"(skipping {queue}: no processed data found)")
            continue
        for rho in RHOS:
            r, val = run_setting(queue, rho, path_fn(rho))
            rows += r
            if rho in PLOT_RHOS:
                store[(queue, rho)] = val
            print(f"done: {queue} rho={rho}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"mae_rmse_investigation{tag}.csv", index=False)
    qlist = [q for q, _ in queues if q in set(res.queue)]
    pd.set_option("display.width", 220)

    def table(col, fmt=3, scale=1.0, title=""):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        t = res.pivot_table(index="rho", columns=["queue", "method"], values=col) * scale
        print(t[qlist].round(fmt).to_string())

    table("MAE", title="MAE on the validation runs (lower is better)")
    table("RMSE", title="RMSE on the validation runs (lower is better)")
    table("MSE_from_congested", 2,
          title="MSE contribution from CONGESTED target times (adds up with the next table to the total MSE)")
    table("MSE_from_not_congested", 2,
          title="MSE contribution from NON-congested target times")
    table("MAE_from_congested", 3, title="MAE contribution from CONGESTED target times")
    table("MAE_from_not_congested", 3, title="MAE contribution from NON-congested target times")
    table("sq_share_over", 1, 100,
          "Share of squared error from OVER-predictions (%) (prediction above the actual value)")
    table("sq_share_top5pct", 1, 100,
          "Share of squared error from the largest 5% of errors (%)")
    table("frac_under_congested", 2,
          title="During congestion: fraction of predictions BELOW the actual value (under-prediction)")
    table("mean_signed_error_congested", 2,
          title="During congestion: mean signed error (prediction - actual; negative = under-predicts)")
    table("frac_exact", 2, title="Fraction of predictions within 0.5 of the actual value")

    # one-line verdict per setting
    print("\n" + "=" * 110)
    print("Where does Random Forest's RMSE advantage come from? (persistence MSE - RF MSE, split)")
    print("=" * 110)
    for q in qlist:
        for rho in RHOS:
            p = res[(res.queue == q) & (res.rho == rho)].set_index("method")
            if len(p) < 2:
                continue
            P, R = p.loc["Persistence"], p.loc["Random Forest"]
            gap = P.MSE - R.MSE
            g_c = P.MSE_from_congested - R.MSE_from_congested
            g_n = P.MSE_from_not_congested - R.MSE_from_not_congested
            mae_gap = P.MAE - R.MAE
            print(f"{q:18s} rho={rho:<4}: MSE gap {gap:+7.2f} = congested {g_c:+7.2f} + not congested {g_n:+7.2f} | "
                  f"MAE gap {mae_gap:+.3f} | persistence over-prediction share {P.sq_share_over:.0%}, "
                  f"RF {R.sq_share_over:.0%}")
    print("(positive gap = Random Forest better)")

    plot_trajectories(store, qlist, tag)
    print(f"\nSaved: results/mae_rmse_investigation{tag}.csv")
    print(f"Saved: results/plots/mae_rmse_trajectories{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-queues", action="store_true")
    ap.add_argument("--map-levels", action="store_true")
    a = ap.parse_args()
    main(a.all_queues, a.map_levels)