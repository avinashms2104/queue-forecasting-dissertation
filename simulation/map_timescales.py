"""
Count variability at many timescales, and what exactly differs between the MAPs (Azam, 16 Oct)
=============================================================================================

Azam: "matching at one window does not establish matching at other timescales. Please
add a plot of the index of dispersion for counts across several window lengths,
including those relevant to the feature lookback and prediction horizon ... how are you
defining weak, medium and strong dependence ... We should describe precisely what
differs between the three processes."

Part 1 - index of dispersion for counts, IDC(t) = Var N(t) / E N(t)
    Exact (map_utils.MAP.vtm), for windows from 0.1 to 1,000 time units, at the arrival
    rate lambda = rho of each queue (rho = 0.5, 0.8, 0.95). Windows are in TIME UNITS
    (mean service times), the unit of the forecasting set-up, and marked:
        lookback of the features   10 time units (10 lagged observations, dt = 1)
        forecast horizons          5, 10, 15, 20 time units (and 40, end of the sweep)
    The matching window of 10 mean interarrival times equals 10 / rho time units
    (12.5 at rho = 0.8). Dots: IDC from simulated arrivals (30 sequences x 100,000
    arrivals), as a check on the exact curve.

Part 2 - what differs between the processes
    Every MMPP-2 has gap autocorrelations of the form
        corr(X_0, X_j) = corr_1 * gamma^(j - 1),   j >= 1,
    where gamma is the second eigenvalue of the phase-to-phase matrix P = (-D0)^{-1} D1.
    So the dependence is described by two numbers:
        corr_1                 how strongly neighbouring gaps are related
        gamma                  how slowly that relation dies out with the lag
    and the summaries
        memory length          -1 / ln(gamma) arrivals (lags for corr to fall by e)
        total gap correlation  sum_j corr_j = corr_1 / (1 - gamma) = (IDC_inf / C_a^2 - 1) / 2
        modulation time        1 / (r12 + r21), the relaxation time of the busy/quiet
                               switching, in mean interarrival times.
    The three controlled MAPs have similar corr_1 (0.15-0.19) but different gamma: the
    levels differ in HOW LONG the dependence lasts, not in how strong it is at lag 1.

Usage:   python simulation/map_timescales.py
Writes:  results/map_timescales_idc.csv        (exact IDC at the marked windows, every rho)
         results/map_dependence_measures.csv   (corr_1, gamma, memory length, ...)
         results/plots/map_timescales_idc.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ph_utils import hyperexp2_balanced
from map_utils import arrival_map, map_level, ph_renewal_map, MATCH_WINDOW

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOTS = RES / "plots"

PLOT_RHOS = [0.5, 0.8, 0.95]
TABLE_RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
LOOKBACK = 10
HORIZONS = [5, 10, 15, 20, 40]
MARKED = sorted(set([1, LOOKBACK] + HORIZONS + [100]))
N_SEQ, N_GAPS = 30, 100_000
LEVELS = ["none (H2)", "weak", "medium", "strong", "cc afternoon", "cc evening"]


def process(level, lam):
    if level == "none (H2)":
        ref = map_level("medium", 1.0)
        return ph_renewal_map(hyperexp2_balanced(1.0 / lam, ref.scv))
    key = {"cc afternoon": "cc_afternoon", "cc evening": "cc_evening"}.get(level, level)
    return arrival_map(key, lam)


def available():
    out = []
    for lv in LEVELS:
        try:
            process(lv, 1.0)
            out.append(lv)
        except Exception as e:
            print(f"  (skipping {lv}: {e})")
    return out


def dependence_measures(levels):
    rows = []
    for lv in levels:
        M = process(lv, 1.0)
        c1 = M.lag_corr(1)
        if lv == "none (H2)":
            gamma, mod_time = 0.0, 0.0
        else:
            ev = np.sort(np.abs(np.linalg.eigvals(M.P)))[::-1]
            gamma = float(ev[1])
            evD = np.sort(np.real(np.linalg.eigvals(M.D0 + M.D1)))
            mod_time = float(-1.0 / evD[0])
        rows.append(dict(
            process=lv, gap_scv=M.scv, corr_1=c1, gamma=gamma,
            memory_length_arrivals=(-1.0 / np.log(gamma)) if 0 < gamma < 1 else 0.0,
            total_gap_correlation=(M.idc_limit() / M.scv - 1) / 2,
            modulation_time_mean_gaps=mod_time,
            corr_5=M.lag_corr(5), corr_20=M.lag_corr(20),
            vtm_10_gaps=M.vtm(MATCH_WINDOW), idc_limit=M.idc_limit()))
    return pd.DataFrame(rows)


def simulated_idc(M, rng, windows):
    vals = []
    for _ in range(N_SEQ):
        g, _ = M.sample(N_GAPS, rng)
        t = np.cumsum(g)
        row = []
        for w in windows:
            c = np.bincount((t // w).astype(int))[:-1]
            row.append(c.var() / c.mean())
        vals.append(row)
    return np.mean(vals, axis=0)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    levels = available()

    # ---------- part 2: dependence measures ----------
    dm = dependence_measures(levels)
    dm.to_csv(RES / "map_dependence_measures.csv", index=False)
    pd.set_option("display.width", 220)
    bar = "=" * 120
    print(f"{bar}\nWhat differs between the arrival processes (exact; gaps in units of the mean gap)\n{bar}")
    print(dm.set_index("process").T.map(lambda x: f"{x:.3f}").to_string())
    print("corr_j = corr_1 * gamma^(j-1); memory length = -1/ln(gamma) arrivals; "
          "total gap correlation = sum_j corr_j;\nmodulation time = mean time for the busy/quiet "
          "switching to forget its state (mean gaps). H2 is renewal: no dependence.")

    # ---------- part 1: IDC at the marked windows (time units), every rho ----------
    rows = []
    for lv in levels:
        for rho in TABLE_RHOS:
            M = process(lv, rho)
            for w in MARKED:
                rows.append(dict(process=lv, rho=rho, kind="marked", window_time_units=w,
                                 window_mean_gaps=w * rho, idc=M.vtm(w)))
            rows.append(dict(process=lv, rho=rho, kind="matching", window_time_units=MATCH_WINDOW / rho,
                             window_mean_gaps=MATCH_WINDOW, idc=M.vtm(MATCH_WINDOW / rho)))
    idc = pd.DataFrame(rows)
    idc.to_csv(RES / "map_timescales_idc.csv", index=False)

    for rho in [0.5, 0.8, 0.95]:
        t = idc[(idc.rho == rho) & (idc.kind == "marked")].pivot(
            index="window_time_units", columns="process", values="idc")[levels]
        t.index = [f"{w:g}" + (" (lookback)" if w == LOOKBACK else "") +
                   (" (horizon)" if w in HORIZONS else "") for w in t.index]
        print(f"\n{bar}\nIDC = Var N(t) / E N(t) for windows of t time units, rho = {rho} "
              f"(matching window = {MATCH_WINDOW / rho:.1f} time units)\n{bar}")
        print(t.map(lambda x: f"{x:.2f}").to_string())

    # ---------- plot ----------
    rng = np.random.default_rng(2026)
    ws = np.logspace(-1, 3, 120)
    sim_w = [1, 5, 10, 20, 40, 100]
    fig, ax = plt.subplots(1, len(PLOT_RHOS), figsize=(17, 5), sharey=True)
    colors = dict(zip(levels, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for a, rho in zip(ax, PLOT_RHOS):
        for lv in levels:
            M = process(lv, rho)
            cc = lv.startswith("cc")
            a.plot(ws, [M.vtm(w) for w in ws], color=colors[lv], lw=2.2 if cc else 1.4,
                   ls="--" if cc else "-", label=lv)
            if rho == 0.8:
                a.scatter(sim_w, simulated_idc(M, rng, sim_w), color=colors[lv], s=16, zorder=3)
        a.axvspan(min(HORIZONS[:4]), max(HORIZONS[:4]), color="grey", alpha=0.12, label="horizons 5-20")
        a.axvline(LOOKBACK, color="black", ls=":", lw=1, label="lookback 10")
        a.axvline(MATCH_WINDOW / rho, color="red", ls="--", lw=1, label=f"10 mean gaps = {MATCH_WINDOW / rho:.1f}")
        a.set_xscale("log"); a.set_xlabel("window length t (time units = mean service times)")
        a.set_title(f"rho = {rho} (arrival rate {rho})" + ("; dots = simulated" if rho == 0.8 else ""), fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("IDC(t) = Var N(t) / E N(t)")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Index of dispersion for counts across timescales: the three controlled MAPs agree only "
                 "at the matching window", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "map_timescales_idc.png", dpi=150)
    print("\nSaved: results/map_dependence_measures.csv, results/map_timescales_idc.csv, "
          "results/plots/map_timescales_idc.png")


if __name__ == "__main__":
    main()