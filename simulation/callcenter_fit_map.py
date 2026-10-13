"""
Call-centre case study, step 1: explore the real arrivals and fit a MAP (Azam, 12 Oct)
====================================================================================

Data: "Anonymous Bank" call centre (Technion SEE Lab), November and December 1999,
one line per call. Put the two monthly .txt files in data/callcenter/raw/.

Azam's instructions:
  * weekdays in November and December 1999, regular operating hours;
  * before fitting, examine the arrival rate over the day, overdispersion and
    temporal dependence;
  * fit the MAP to selected periods with relatively stable arrival rates rather
    than treating the whole dataset as stationary;
  * use the fitted MAP in the MAP/PH/1 simulation (option (a)).

Choices made here
-----------------
Arrivals   Calls that asked for an agent: served by an agent (outcome AGENT) or
           abandoned while waiting in the queue (HANG with a queue entry time).
           Calls that ended inside the automated voice system (VRU) never asked
           for an agent, and PHANTOM calls are ignored, as the documentation says.
           Arrival time = the moment the call left the VRU (vru_exit), i.e. the
           moment it asked for an agent.
Weekdays   In Israel the working week is Sunday to Thursday; Friday and Saturday
           are excluded. Regular hours: 07:00 to 24:00.
Stable     Half-hour arrival counts averaged over the weekdays. A stable period is
periods    a run of at least 3 hours (6 half-hours) in which every half-hour mean
           is within +/-10% of the run's own mean; the longest such runs that do
           not overlap are kept.
Outages    A day is left out of a period if it contains a gap of more than 15
           minutes with no calls (one such day exists: a 37-minute silence on
           30 December, almost certainly a system fault).
Counts     Overdispersion = variance-to-mean ratio (VtM) of the number of
           arrivals in windows of k mean gaps. "Within-day" VtM uses each day's
           own mean, so it measures variability INSIDE a stable period; "pooled"
           also includes differences between days. The MAP describes one stable
           period, so it is fitted to the within-day values. Only windows with
           at least 8 per day are used.
The fit    A two-state MMPP (the same family as the controlled experiments),
           scaled to arrival rate 1, chosen to match (least squares on logs) the
           within-day VtM curve, the variability of single gaps (C_a^2), and the
           lag-1 gap correlation. The mean time spent in each state is capped at a
           quarter of the period, so that the switching happens inside a stable
           period rather than standing in for a slow trend across the whole of it.
           The fitted process is then checked against
           simulated arrival sequences, as for the controlled MAPs.

Usage:   python simulation/callcenter_fit_map.py
Writes:  data/callcenter/arrivals_weekdays.csv        (one row per agent-seeking arrival)
         results/callcenter_rate_profile.csv           (half-hour profile)
         results/callcenter_periods.csv                (data vs fitted MAP, per stable period)
         results/callcenter_map_fit.json               (fitted D0, D1 at rate 1: used by map_utils)
         results/plots/callcenter_arrivals.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from map_utils import MAP, mmpp2

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "callcenter" / "raw"
OUT_DATA = ROOT / "data" / "callcenter"
RES = ROOT / "results"
PLOTS = RES / "plots"

OPEN_H, CLOSE_H = 7, 24
WEEKDAYS = {6, 0, 1, 2, 3}           # pandas: Monday=0 ... Sunday=6 -> Sunday to Thursday
MIN_SLOTS, TOL = 6, 0.10             # stable period: >= 3 hours, every half-hour within +/-10%
OUTAGE_S = 15 * 60
WINDOWS = [1, 2, 5, 10, 20, 40, 80]  # in mean gaps
MIN_BINS = 8
GAP_LAGS = [1, 2, 5, 10, 20]
N_SEQ, N_GAPS = 30, 100_000          # simulation check of the fitted MAP


def to_sec(s):
    h, m, x = (int(v) for v in str(s).split(":"))
    return h * 3600 + m * 60 + x


def load_arrivals():
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() == ".txt")
    if not files:
        raise SystemExit(f"No .txt files found in {RAW}")
    df = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda c: c.str.strip())
    print(f"Read {len(df):,} calls from {', '.join(p.name for p in files)}")
    df["day"] = pd.to_datetime(df["date"], format="%y%m%d")
    q_start = df["q_start"].map(to_sec)
    asked_agent = (df["outcome"] == "AGENT") | ((df["outcome"] == "HANG") & (q_start > 0))
    a = df[asked_agent & df["day"].dt.dayofweek.isin(WEEKDAYS)].copy()
    a["t_sec"] = a["vru_exit"].map(to_sec)
    a = a[(a.t_sec >= OPEN_H * 3600) & (a.t_sec < CLOSE_H * 3600)]
    print(f"Agent-seeking calls on weekdays (Sun-Thu) {OPEN_H:02d}:00-{CLOSE_H:02d}:00: {len(a):,} "
          f"on {a.day.nunique()} days (outcome: {dict(a.outcome.value_counts())})")
    a = a[["day", "t_sec", "type", "outcome"]].sort_values(["day", "t_sec"]).reset_index(drop=True)
    return a


def rate_profile(a):
    a = a.assign(slot=a.t_sec // 1800)
    counts = a.groupby(["day", "slot"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(OPEN_H * 2, CLOSE_H * 2), fill_value=0)
    prof = pd.DataFrame({
        "start": [f"{s // 2:02d}:{30 * (s % 2):02d}" for s in counts.columns],
        "mean_calls": counts.mean().values,
        "calls_per_min": counts.mean().values / 30,
        "vtm_across_days": (counts.var(ddof=1) / counts.mean()).values,
    }, index=counts.columns)
    return prof


def stable_periods(prof):
    m = prof["mean_calls"].to_numpy()
    slots = list(prof.index)
    used = np.zeros(len(m), bool)
    found = []
    while True:
        best = None
        for i in range(len(m)):
            for j in range(i + MIN_SLOTS, len(m) + 1):
                if used[i:j].any():
                    break
                seg = m[i:j]
                if np.all(np.abs(seg / seg.mean() - 1) <= TOL):
                    if best is None or (j - i) > (best[1] - best[0]):
                        best = (i, j)
        if best is None:
            break
        used[best[0]:best[1]] = True
        found.append((slots[best[0]] * 1800, slots[best[1] - 1] * 1800 + 1800))
    return sorted(found)


def hhmm(sec):
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}"


def period_stats(a, lo, hi, rng):
    w = a[(a.t_sec >= lo) & (a.t_sec < hi)]
    gaps, keep, dropped = [], [], []
    for d, g in w.groupby("day"):
        t = np.sort(g.t_sec.to_numpy() + rng.random(len(g)))      # spread calls within the same second
        if np.diff(np.r_[lo, t, hi]).max() > OUTAGE_S:
            dropped.append(str(d.date())); continue
        keep.append(d); gaps.append(np.diff(t))
    w = w[w.day.isin(keep)]
    lam = len(w) / (len(keep) * (hi - lo))                      # calls per second
    G = np.concatenate(gaps)
    st = dict(start=hhmm(lo), end=hhmm(hi), days=len(keep), days_dropped=";".join(dropped),
              calls=len(w), calls_per_min=lam * 60, mean_gap_s=G.mean(), gap_scv=G.var() / G.mean() ** 2)
    for j in GAP_LAGS:
        st[f"gap_corr_{j}"] = float(np.mean([np.corrcoef(x[:-j], x[j:])[0, 1] for x in gaps if len(x) > j + 2]))
    usable = []
    for k in WINDOWS:
        tau = k / lam
        nb = int((hi - lo) // tau)
        if nb < MIN_BINS:
            continue
        C = np.array([np.bincount(((g.t_sec.to_numpy() - lo) // tau).astype(int), minlength=nb)[:nb]
                      for _, g in w.groupby("day")])
        st[f"vtm_within_{k}"] = float(np.mean(C.var(axis=1, ddof=1)) / C.mean())
        st[f"vtm_pooled_{k}"] = float(C.var(ddof=1) / C.mean())
        if k == 10:
            st["count_corr_1_at_10"] = float(np.mean([np.corrcoef(c[:-1], c[1:])[0, 1] for c in C]))
        usable.append(k)
    return st, usable


def fit_mmpp(st, usable, rng, max_spell):
    """max_spell: longest allowed mean time in a state (mean gaps), so that the switching
    happens INSIDE a stable period and is not just a slow trend across the whole period."""
    ks = np.array(usable, float)
    target = np.array([st[f"vtm_within_{k}"] for k in usable])
    t_scv, t_c1 = st["gap_scv"], st["gap_corr_1"]

    def build(x):
        ratio = 1.0 + np.exp(x[0])                         # busy rate / quiet rate
        s1, s2 = max_spell / (1.0 + np.exp(-x[1:]))        # mean time in busy / quiet state (< max_spell)
        p = s1 / (s1 + s2)                                 # share of time busy
        l2 = 1.0 / (p * ratio + (1 - p)); l1 = ratio * l2  # overall rate exactly 1
        return mmpp2(l1, l2, 1.0 / s1, 1.0 / s2)

    def loss(x):
        try:
            M = build(x)
            v = np.array([M.vtm(k) for k in ks])
            if not np.all(v > 0):
                return 1e6
            return (np.sum((np.log(v) - np.log(target)) ** 2) + (np.log(M.scv) - np.log(t_scv)) ** 2
                    + 4 * (M.lag_corr(1) - t_c1) ** 2)
        except Exception:
            return 1e6

    best = None
    for _ in range(12):
        x0 = np.r_[np.log(rng.uniform(0.1, 3)), rng.normal(-1, 1.5, 2)]
        r = minimize(loss, x0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-9, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    M = build(best.x)
    return M, best.fun


def simulate_check(M, rng, usable):
    """VtM, gap scv and lag-1 correlation of the fitted MAP from simulated arrival sequences."""
    rows = []
    for _ in range(N_SEQ):
        g, _ = M.sample(N_GAPS, rng)
        t = np.cumsum(g)
        d = {"gap_scv": g.var() / g.mean() ** 2, "gap_corr_1": np.corrcoef(g[:-1], g[1:])[0, 1]}
        for k in usable:
            c = np.bincount((t // k).astype(int))[:-1]
            d[f"vtm_{k}"] = c.var() / c.mean()
        rows.append(d)
    s = pd.DataFrame(rows)
    return s.mean(), s.std(ddof=1) / np.sqrt(N_SEQ)


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    a = load_arrivals()
    a.to_csv(OUT_DATA / "arrivals_weekdays.csv", index=False)

    prof = rate_profile(a)
    prof.to_csv(RES / "callcenter_rate_profile.csv", index=False)
    print("\nArrival rate over the day (mean over weekdays) and day-to-day overdispersion per half hour")
    print(prof.round(2).to_string(index=False))

    periods = stable_periods(prof)
    print("\nStable periods (>= 3 h, every half-hour mean within +/-10% of the period mean): "
          + ", ".join(f"{hhmm(lo)}-{hhmm(hi)}" for lo, hi in periods))

    rows, fits = [], {}
    for lo, hi in periods:
        name = f"cc_{hhmm(lo).replace(':', '')}_{hhmm(hi).replace(':', '')}"
        st, usable = period_stats(a, lo, hi, rng)
        max_spell = st["calls"] / st["days"] / 4          # a quarter of the period, in mean gaps
        M, fun = fit_mmpp(st, usable, rng, max_spell)
        sim_mean, sim_se = simulate_check(M, rng, usable)
        st.update(name=name, fit_loss=fun, map_gap_scv=M.scv, map_gap_corr_1=M.lag_corr(1),
                  map_gap_corr_5=M.lag_corr(5), map_idc_limit=M.idc_limit(),
                  sim_gap_scv=sim_mean["gap_scv"], sim_gap_corr_1=sim_mean["gap_corr_1"])
        for k in usable:
            st[f"map_vtm_{k}"] = M.vtm(k)
            st[f"sim_vtm_{k}"] = sim_mean[f"vtm_{k}"]
        rows.append(st)
        l1, l2 = M.D1[0, 0], M.D1[1, 1]
        r12, r21 = M.D0[0, 1], M.D0[1, 0]
        fits[name] = dict(period=f"{hhmm(lo)}-{hhmm(hi)}", calls_per_min=st["calls_per_min"],
                          D0=M.D0.tolist(), D1=M.D1.tolist(),
                          busy_rate=max(l1, l2), quiet_rate=min(l1, l2),
                          mean_spell_state1=1 / r12, mean_spell_state2=1 / r21,
                          windows_used=usable)

        print(f"\n{'=' * 100}\nStable period {hhmm(lo)}-{hhmm(hi)}  ({name})\n{'=' * 100}")
        print(f"days used {st['days']} (left out: {st['days_dropped'] or 'none'}), calls {st['calls']:,}, "
              f"{st['calls_per_min']:.2f} calls per minute, mean gap {st['mean_gap_s']:.1f} s")
        print(f"{'':28s}{'data':>10s}{'fitted MAP':>12s}{'simulated':>11s}")
        print(f"{'gap variability C_a^2':28s}{st['gap_scv']:10.3f}{M.scv:12.3f}{sim_mean['gap_scv']:11.3f}")
        print(f"{'gap correlation, lag 1':28s}{st['gap_corr_1']:10.3f}{M.lag_corr(1):12.3f}{sim_mean['gap_corr_1']:11.3f}")
        print(f"{'gap correlation, lag 5':28s}{st['gap_corr_5']:10.3f}{M.lag_corr(5):12.3f}")
        for k in usable:
            print(f"{('VtM over ' + str(k) + ' gaps'):28s}{st[f'vtm_within_{k}']:10.3f}{M.vtm(k):12.3f}"
                  f"{sim_mean[f'vtm_{k}']:11.3f}   (pooled over days: {st[f'vtm_pooled_{k}']:.3f})")
        print(f"{'long-run IDC':28s}{'':10s}{M.idc_limit():12.3f}")
        print(f"count correlation between neighbouring 10-gap windows (data): {st.get('count_corr_1_at_10', np.nan):.3f}")
        print(f"fitted MMPP-2 (rate 1): rates {l1:.3f} and {l2:.3f}, "
              f"mean time in each state {1 / r12:.1f} and {1 / r21:.1f} mean gaps")

    pd.DataFrame(rows).to_csv(RES / "callcenter_periods.csv", index=False)
    with open(RES / "callcenter_map_fit.json", "w") as f:
        json.dump(fits, f, indent=2)
    print("\nSaved: data/callcenter/arrivals_weekdays.csv, results/callcenter_rate_profile.csv, "
          "results/callcenter_periods.csv, results/callcenter_map_fit.json")

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(prof)) / 2 + OPEN_H
    ax[0].step(x, prof["calls_per_min"], where="post")
    for lo, hi in periods:
        ax[0].axvspan(lo / 3600, hi / 3600, alpha=0.15, color="green")
    ax[0].set_xlabel("hour of day"); ax[0].set_ylabel("calls per minute (weekday mean)")
    ax[0].set_title("Arrival rate over the day (green = stable periods)")
    ax[1].step(x, prof["vtm_across_days"], where="post", color="C1")
    ax[1].axhline(1, color="grey", lw=0.8)
    ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("VtM of half-hour counts across days")
    ax[1].set_title("Overdispersion between days (1 = Poisson)")
    kk = np.logspace(0, np.log10(100), 50)
    for i, r in enumerate(rows):
        M = MAP(np.array(fits[r["name"]]["D0"]), np.array(fits[r["name"]]["D1"]))
        us = fits[r["name"]]["windows_used"]
        ax[2].plot(kk, [M.vtm(k) for k in kk], color=f"C{i}", label=f"{r['start']}-{r['end']} fitted MAP")
        ax[2].scatter(us, [r[f"vtm_within_{k}"] for k in us], color=f"C{i}", marker="o", label="data (within day)")
        ax[2].scatter(us, [r[f"vtm_pooled_{k}"] for k in us], color=f"C{i}", marker="x", label="data (pooled over days)")
    ax[2].axhline(1, color="grey", lw=0.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("window (mean gaps)"); ax[2].set_ylabel("VtM of counts")
    ax[2].set_title("Count variability inside the stable periods"); ax[2].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "callcenter_arrivals.png", dpi=150)
    print("Saved: results/plots/callcenter_arrivals.png")


if __name__ == "__main__":
    main()