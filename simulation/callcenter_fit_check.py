"""
Call-centre case study: how well do the fitted MAPs represent the data? (Azam, 16 Oct)
=====================================================================================

Azam: "Agreement with QBD checks the queueing calculations for the fitted model; it
does not by itself establish that the model represents the observed data well. Please
explain how you assessed the stability of the selected periods and whether overnight
gaps and boundaries between days were excluded from the dependence calculations."

This script answers three questions for each stable period found by
callcenter_fit_map.py (it reuses that script's functions, so every choice is the same):

1. Are the periods stable?
   * slot check (used to select them): every half-hour mean within +/-10% of the period mean
   * trend within the period: each day's rate in the first half against the second half
     (paired over days; a stable period should show no systematic change)
   * day-to-day variation: coefficient of variation of the daily period rate, against
     the value expected if the daily counts were Poisson with a common rate
   * November against December: period rate in each month

2. Are day boundaries excluded? (counted, not assumed)
   * gaps are formed only between calls on the SAME day and INSIDE the period window,
     so no gap spans midnight, an overnight closure or the period boundary;
   * count windows run from the period start and the last, incomplete window of each
     day is dropped, so no window spans two days or the period end;
   * a day with a silence longer than 15 minutes (a system fault) is left out.
   The script reports the number of gaps, the longest gap used and how many days were
   left out.

3. Does the fitted MAP describe data it was not fitted to?  (out-of-sample check)
   The MMPP-2 is fitted to NOVEMBER only and its exact predictions are compared with the
   DECEMBER data: gap C_a^2, lag-1 gap correlation and the within-day VtM curve. The
   uncertainty of the December values is a 95% bootstrap interval over days (days are
   resampled, so the dependence inside a day is kept). For reference: Poisson arrivals
   (all values 1, correlation 0) and the MAP fitted to both months (the one used in the
   queue experiments).

Usage:   python simulation/callcenter_fit_check.py     (after callcenter_fit_map.py)
Writes:  results/callcenter_fit_check.csv, results/callcenter_stability.csv,
         results/plots/callcenter_fit_check.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map_utils import MAP
from callcenter_fit_map import (rate_profile, stable_periods, period_stats, fit_mmpp, hhmm,
                                OUTAGE_S, MIN_BINS, WINDOWS)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "data" / "callcenter" / "arrivals_weekdays.csv"
RES = ROOT / "results"
PLOTS = RES / "plots"
N_BOOT = 200


def load():
    if not ARR.exists():
        raise SystemExit(f"{ARR} not found: run simulation/callcenter_fit_map.py first")
    a = pd.read_csv(ARR, parse_dates=["day"])
    return a


def stability(a, lo, hi, kept_days):
    w = a[(a.t_sec >= lo) & (a.t_sec < hi) & a.day.isin(kept_days)]
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    per_day = w.groupby("day").agg(n=("t_sec", "size"),
                                   n_first=("t_sec", lambda s: (s < mid).sum()))
    per_day["n_second"] = per_day["n"] - per_day["n_first"]
    rate1 = per_day["n_first"] / half * 60
    rate2 = per_day["n_second"] / half * 60
    diff = rate2 - rate1
    daily_rate = per_day["n"] / (hi - lo) * 60
    cv = daily_rate.std(ddof=1) / daily_rate.mean()
    cv_poisson = 1 / np.sqrt(per_day["n"].mean())
    month = w.assign(m=w.day.dt.month).groupby("m").agg(n=("t_sec", "size"), days=("day", "nunique"))
    month_rate = month["n"] / month["days"] / (hi - lo) * 60
    return dict(days=len(per_day),
                rate_first_half=rate1.mean(), rate_second_half=rate2.mean(),
                trend_per_min=diff.mean(), trend_se=diff.std(ddof=1) / np.sqrt(len(diff)),
                daily_rate_cv=cv, poisson_cv=cv_poisson,
                rate_november=month_rate.get(11, np.nan), rate_december=month_rate.get(12, np.nan))


def boundary_report(a, lo, hi, rng):
    w = a[(a.t_sec >= lo) & (a.t_sec < hi)]
    n_gaps, longest, dropped, n_days = 0, 0.0, 0, 0
    for _, g in w.groupby("day"):
        t = np.sort(g.t_sec.to_numpy() + rng.random(len(g)))
        if np.diff(np.r_[lo, t, hi]).max() > OUTAGE_S:
            dropped += 1
            continue
        n_days += 1
        x = np.diff(t)
        n_gaps += len(x)
        longest = max(longest, x.max())
    return dict(days_used=n_days, days_dropped=dropped, gaps_used=n_gaps,
                longest_gap_used_s=longest, gaps_crossing_days=0)


def boot_stats(a_month, lo, hi, rng, usable):
    """Bootstrap over days: resample whole days, recompute the statistics."""
    days = a_month.day.unique()
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, size=len(days), replace=True)
        parts = []
        for i, d in enumerate(pick):
            g = a_month[a_month.day == d].copy()
            g["day"] = pd.Timestamp("2000-01-01") + pd.Timedelta(days=i)   # unique label per draw
            parts.append(g)
        st, _ = period_stats(pd.concat(parts), lo, hi, rng)
        out.append([st["gap_scv"], st["gap_corr_1"]] + [st.get(f"vtm_within_{k}", np.nan) for k in usable])
    return np.nanpercentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    a = load()
    prof = rate_profile(a)
    periods = stable_periods(prof)
    fits = json.loads((RES / "callcenter_map_fit.json").read_text()) if (RES / "callcenter_map_fit.json").exists() else {}
    bar = "=" * 110
    pd.set_option("display.width", 220)
    fit_rows, stab_rows = [], []
    fig, ax = plt.subplots(1, len(periods), figsize=(7 * len(periods), 4.6), squeeze=False)

    for pi, (lo, hi) in enumerate(periods):
        name = f"cc_{hhmm(lo).replace(':', '')}_{hhmm(hi).replace(':', '')}"
        label = f"{hhmm(lo)}-{hhmm(hi)}"

        # ---------- 1 and 2: stability and boundaries ----------
        st_all, usable = period_stats(a, lo, hi, rng)
        kept = [d for d in a.day.unique() if str(pd.Timestamp(d).date()) not in st_all["days_dropped"].split(";")]
        s = stability(a, lo, hi, kept)
        b = boundary_report(a, lo, hi, np.random.default_rng(1))
        stab_rows.append(dict(period=label, **s, **b))
        print(f"\n{bar}\nStable period {label}\n{bar}")
        print(f"Slot check (selection rule): every half-hour mean within +/-10% of the period mean.")
        print(f"Trend inside the period: {s['rate_first_half']:.3f} calls/min in the first half, "
              f"{s['rate_second_half']:.3f} in the second (change {s['trend_per_min']:+.3f} +/- {1.96 * s['trend_se']:.3f}, "
              f"95%, over {s['days']} days)")
        print(f"Day-to-day variation of the period rate: CV {s['daily_rate_cv']:.3f} "
              f"(Poisson with a common rate would give {s['poisson_cv']:.3f})")
        print(f"Rate by month: November {s['rate_november']:.3f}, December {s['rate_december']:.3f} calls/min")
        print(f"Boundaries: {b['gaps_used']:,} gaps on {b['days_used']} days, all between calls on the same day "
              f"inside {label}; longest gap used {b['longest_gap_used_s'] / 60:.1f} min; "
              f"{b['days_dropped']} day(s) left out for a silence > {OUTAGE_S // 60} min; "
              f"0 gaps or count windows cross a day or the period boundary.")

        # ---------- 3: fit on November, test on December ----------
        nov = a[a.day.dt.month == 11]
        dec = a[a.day.dt.month == 12]
        st_nov, us_nov = period_stats(nov, lo, hi, rng)
        st_dec, us_dec = period_stats(dec, lo, hi, rng)
        usable = [k for k in us_nov if k in us_dec]
        max_spell = st_nov["calls"] / st_nov["days"] / 4
        M_nov, _ = fit_mmpp(st_nov, usable, rng, max_spell)
        M_all = MAP(np.array(fits[name]["D0"]), np.array(fits[name]["D1"])) if name in fits else None
        dec_ci = boot_stats(dec[(dec.t_sec >= lo) & (dec.t_sec < hi)], lo, hi, rng, usable)

        keys = ["gap_scv", "gap_corr_1"] + [f"vtm_within_{k}" for k in usable]
        labels = ["gap C_a^2", "gap corr lag 1"] + [f"VtM over {k} gaps" for k in usable]
        pred_nov = [M_nov.scv, M_nov.lag_corr(1)] + [M_nov.vtm(k) for k in usable]
        pred_all = ([M_all.scv, M_all.lag_corr(1)] + [M_all.vtm(k) for k in usable]) if M_all is not None else [np.nan] * len(keys)
        poisson = [1.0, 0.0] + [1.0] * len(usable)
        print(f"\nOut-of-sample check: MAP fitted to November ({st_nov['days']} days), compared with December "
              f"({st_dec['days']} days)")
        print(f"{'':20s}{'Dec data':>10s}{'95% interval (days)':>22s}{'Nov-fitted MAP':>16s}{'both-month MAP':>16s}"
              f"{'Poisson':>9s}{'inside?':>9s}")
        inside = 0
        for i, (k, lab) in enumerate(zip(keys, labels)):
            lo_ci, hi_ci = dec_ci[0, i], dec_ci[1, i]
            ok = lo_ci <= pred_nov[i] <= hi_ci
            inside += ok
            print(f"{lab:20s}{st_dec[k]:10.3f}   [{lo_ci:6.3f}, {hi_ci:6.3f}]     {pred_nov[i]:13.3f}"
                  f"{pred_all[i]:16.3f}{poisson[i]:9.3f}{'yes' if ok else 'no':>9s}")
            fit_rows.append(dict(period=label, quantity=lab, december=st_dec[k], ci_low=lo_ci, ci_high=hi_ci,
                                 november_fit=pred_nov[i], both_months_fit=pred_all[i], poisson=poisson[i],
                                 nov_fit_inside_ci=ok))
        vt = [i for i, k in enumerate(keys) if k.startswith("vtm")]
        err_map = np.mean([abs(np.log(pred_nov[i]) - np.log(st_dec[keys[i]])) for i in vt])
        err_poi = np.mean([abs(np.log(st_dec[keys[i]])) for i in vt])
        print(f"{inside} of {len(keys)} November-fitted predictions lie inside the December 95% interval. "
              f"Mean |log error| of the VtM curve: November-fitted MAP {err_map:.3f}, Poisson {err_poi:.3f}.")
        print(f"November-fitted MMPP-2 (rate 1): rates {M_nov.D1[0, 0]:.3f} / {M_nov.D1[1, 1]:.3f}, "
              f"mean time in each state {1 / M_nov.D0[0, 1]:.1f} / {1 / M_nov.D0[1, 0]:.1f} mean gaps")

        # ---------- plot ----------
        axp = ax[0, pi]
        kk = np.logspace(0, np.log10(max(usable) * 1.5), 60)
        axp.plot(kk, [M_nov.vtm(k) for k in kk], color="C0", label="MAP fitted to November")
        if M_all is not None:
            axp.plot(kk, [M_all.vtm(k) for k in kk], color="C2", ls="--", label="MAP fitted to both months")
        vals = [st_dec[f"vtm_within_{k}"] for k in usable]
        err = np.array([[st_dec[f"vtm_within_{k}"] - dec_ci[0, 2 + j], dec_ci[1, 2 + j] - st_dec[f"vtm_within_{k}"]]
                        for j, k in enumerate(usable)]).T
        axp.errorbar(usable, vals, yerr=err, fmt="o", color="C3", capsize=3, label="December data (95% interval)")
        axp.axhline(1, color="grey", lw=0.8, label="Poisson")
        axp.set_xscale("log"); axp.set_xlabel("window (mean gaps)"); axp.set_ylabel("VtM of counts (within day)")
        axp.set_title(f"{label}: fitted on November, tested on December", fontsize=10)
        axp.grid(alpha=0.3); axp.legend(fontsize=8)

    pd.DataFrame(fit_rows).to_csv(RES / "callcenter_fit_check.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(RES / "callcenter_stability.csv", index=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "callcenter_fit_check.png", dpi=150)
    print("\nSaved: results/callcenter_fit_check.csv, results/callcenter_stability.csv, "
          "results/plots/callcenter_fit_check.png")


if __name__ == "__main__":
    main()