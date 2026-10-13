"""
Check the three controlled MAP burst levels (Azam, 12 Oct)
==========================================================

Azam asked that the three MAP models have
    (1) approximately the same mean arrival rate,
    (2) the same arrival-count variability over a specified time interval,
    (3) but different dependence structures,
and that we report the autocorrelations and verify all of this with simulated
arrival sequences.

What this script does
---------------------
For H2 (no dependence, reference) and the weak / medium / strong MAPs it
computes the EXACT values from the matrices (map_utils.py) and the SIMULATED
values from 30 independent arrival sequences of 200,000 arrivals each, and
gives z = (simulated - exact) / standard error over the 30 sequences.

    rate         arrivals per mean interarrival time (1 by construction)
    gap_scv      variability of a single gap (C_a^2)
    vtm_W        variance-to-mean ratio of the number of arrivals in a window
                 of W mean gaps (W = 1, 5, 10, 20, 50). W = 10 is the matching
                 window: all three MAPs have the same value there.
    corr_j       correlation between gap X_0 and gap X_j (j = 1, 2, 5, 10, 20, 50)
    count_corr_k correlation between arrival counts in two windows of 10 mean
                 gaps, k windows apart (k = 1, 2, 3, 5)
    idc_limit    VtM for a very long window (exact only)

All rates scale with lambda, so these shape numbers are the same at every rho
when windows are measured in mean gaps (10 mean gaps = 10/rho time units).

Usage
-----
    python simulation/check_map_levels.py

Writes ../results/map_levels_properties.csv and ../plots/map_levels_properties.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ph_utils import hyperexp2_balanced
from map_utils import map_level, ph_renewal_map, MATCH_WINDOW, MAP_LEVELS

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results"
PLOT_DIR = ROOT / "plots"

N_SEQ = 30
N_GAPS = 200_000
WINDOWS = [1, 5, 10, 20, 50]
GAP_LAGS = [1, 2, 5, 10, 20, 50]
COUNT_LAGS = [1, 2, 3, 5]


def make_levels():
    ref = map_level("medium", 1.0)
    h2 = ph_renewal_map(hyperexp2_balanced(1.0, ref.scv))
    h2.name = "H2 renewal (no dependence)"
    out = {"none (H2)": h2}
    for lv in MAP_LEVELS:
        out[lv] = map_level(lv, 1.0)
    return out


def exact_props(M):
    d = {"rate": M.lam, "gap_scv": M.scv}
    for w in WINDOWS:
        d[f"vtm_{w}"] = M.vtm(w / M.lam)
    for j in GAP_LAGS:
        d[f"corr_{j}"] = M.lag_corr(j)
    for k in COUNT_LAGS:
        d[f"count_corr_{k}"] = M.count_corr(MATCH_WINDOW / M.lam, k)
    return d


def acf(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).mean() / x.var())


def simulated_props(M, seed):
    rng = np.random.default_rng(seed)
    g, _ = M.sample(N_GAPS, rng)
    t = np.cumsum(g)
    T = t[-1]
    d = {"rate": N_GAPS / T, "gap_scv": g.var() / g.mean() ** 2}
    for w in WINDOWS:
        width = w / M.lam
        c = np.bincount((t // width).astype(int))[:-1]          # drop the last, incomplete window
        d[f"vtm_{w}"] = c.var() / c.mean()
        if w == MATCH_WINDOW:
            counts10 = c
    for j in GAP_LAGS:
        d[f"corr_{j}"] = acf(g, j)
    for k in COUNT_LAGS:
        d[f"count_corr_{k}"] = acf(counts10.astype(float), k)
    return d


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    levels = make_levels()
    rows = []
    print(f"Matching window: {MATCH_WINDOW:g} mean gaps | {N_SEQ} sequences x {N_GAPS:,} arrivals each\n")
    for i, (lv, M) in enumerate(levels.items()):
        print(f"{lv:10s} {M.name}")
        ex = exact_props(M)
        sims = pd.DataFrame([simulated_props(M, 1000 * i + s) for s in range(N_SEQ)])
        for key, e in ex.items():
            s_mean = sims[key].mean()
            se = sims[key].std(ddof=1) / np.sqrt(N_SEQ)
            rows.append(dict(level=lv, quantity=key, exact=e, simulated=s_mean, se=se,
                             z=(s_mean - e) / se if se > 0 else np.nan))
        rows.append(dict(level=lv, quantity="idc_limit", exact=M.idc_limit(),
                         simulated=np.nan, se=np.nan, z=np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "map_levels_properties.csv", index=False)

    # ---------- printed tables ----------
    order = list(levels)
    def table(keys, title, digits=3):
        print(f"\n{title}  (exact / simulated)")
        print(f"{'':16s}" + "".join(f"{lv:>20s}" for lv in order))
        for k in keys:
            line = f"{k:16s}"
            for lv in order:
                r = res[(res.level == lv) & (res.quantity == k)].iloc[0]
                sim = "" if np.isnan(r.simulated) else f" / {r.simulated:.{digits}f}"
                line += f"{f'{r.exact:.{digits}f}{sim}':>20s}"
            print(line)
    table(["rate", "gap_scv"] + [f"vtm_{w}" for w in WINDOWS] + ["idc_limit"],
          "1) SAME rate, gap variability and count variability at the matching window")
    table([f"corr_{j}" for j in GAP_LAGS], "2) DIFFERENT dependence: correlation between gaps j apart")
    table([f"count_corr_{k}" for k in COUNT_LAGS],
          f"3) DIFFERENT dependence: correlation of counts in windows of {MATCH_WINDOW:g} mean gaps, k windows apart")

    z = res.z.dropna()
    print(f"\nSimulation vs exact: |z| > 2 in {int((z.abs() > 2).sum())} of {len(z)} checks "
          f"(largest |z| = {z.abs().max():.2f}; about 5% are expected above 2 by chance)")
    print("Saved -> results/map_levels_properties.csv")

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ws = np.logspace(-1, np.log10(500), 80)
    lags = np.arange(1, 81)
    for lv, M in levels.items():
        line, = ax[0].plot(ws, [M.vtm(w / M.lam) for w in ws], label=lv)
        sub = res[res.level == lv].set_index("quantity")
        ax[0].scatter(WINDOWS, [sub.loc[f"vtm_{w}", "simulated"] for w in WINDOWS],
                      color=line.get_color(), s=18, zorder=3)
        ax[1].plot(lags, [M.lag_corr(j) for j in lags], color=line.get_color(), label=lv)
        ax[1].scatter(GAP_LAGS, [sub.loc[f"corr_{j}", "simulated"] for j in GAP_LAGS],
                      color=line.get_color(), s=18, zorder=3)
    ax[0].axvline(MATCH_WINDOW, color="grey", ls="--", lw=1)
    ax[0].text(MATCH_WINDOW * 1.08, 1.2, "matching window", color="grey", fontsize=9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("window length (mean interarrival times)")
    ax[0].set_ylabel("variance / mean of arrival count")
    ax[0].set_title("Count variability: equal at the matching window")
    ax[1].set_xlabel("lag j (arrivals apart)")
    ax[1].set_ylabel("correlation between gaps")
    ax[1].set_title("Dependence: how long the memory lasts")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.text(0.5, -0.02, "lines = exact, dots = simulated (30 x 200,000 arrivals)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "map_levels_properties.png", dpi=150, bbox_inches="tight")
    print("Saved -> plots/map_levels_properties.png")


if __name__ == "__main__":
    main()