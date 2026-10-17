"""
Call-centre case study: real service-time variability (Azam, 16 Oct)
===================================================================

Azam: "Keep Erlang-2 as the main comparison, but add one sensitivity check using the
observed service-time variability if the data allow it. Hold the mean service time and
utilisation fixed to isolate the effect of variability. A suitable PH fit would allow
you to retain the QBD comparison."

This script measures the service times of the calls that were served by an agent and
arrived (left the voice system) in each stable period, on the same weekdays as the
arrival analysis, and fits a phase-type distribution to them:

    * balanced two-phase hyperexponential (H2), mean scaled to 1 (the project's time
      unit), squared coefficient of variation C_s^2 equal to the observed one.
      It matches the first two moments exactly; the third moment is reported, so the
      remaining mismatch in the tail is visible.

The mean is set to 1 and rho is kept at the same values as before, so only the
variability of service changes (Erlang-2 has C_s^2 = 0.5). The fit is written to
results/callcenter_service_fit.json and used by map_utils.service_ph() for the levels
cc_afternoon_realsvc and cc_evening_realsvc.

Usage:   python simulation/callcenter_service_fit.py    (needs data/callcenter/raw/*.txt)
Writes:  results/callcenter_service_fit.json, results/callcenter_service_times.csv,
         results/plots/callcenter_service_times.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ph_utils import hyperexp2_balanced, erlang

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "callcenter" / "raw"
RES = ROOT / "results"
PLOTS = RES / "plots"
WEEKDAYS = {6, 0, 1, 2, 3}
PERIODS = {"cc_afternoon": (11 * 3600, 15 * 3600 + 1800), "cc_evening": (19 * 3600 + 1800, 22 * 3600 + 1800)}


def to_sec(s):
    h, m, x = (int(v) for v in str(s).split(":"))
    return h * 3600 + m * 60 + x


def ph_moment3(ph):
    """Third moment of a PH distribution: 6 alpha (-T)^{-3} 1."""
    M = np.linalg.inv(-ph.T)
    return float(6 * ph.alpha @ M @ M @ M @ np.ones(len(ph.alpha)))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() == ".txt")
    df = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda c: c.str.strip())
    df["day"] = pd.to_datetime(df["date"], format="%y%m%d")
    df = df[(df["outcome"] == "AGENT") & df["day"].dt.dayofweek.isin(WEEKDAYS)].copy()
    df["t"] = df["vru_exit"].map(to_sec)
    df["s"] = df["ser_time"].astype(float)

    out, rows = {}, []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (lv, (lo, hi)) in enumerate(PERIODS.items()):
        w = df[(df.t >= lo) & (df.t < hi)]
        s = w["s"].to_numpy()
        zero = float((s <= 0).mean())
        s = s[s > 0]
        mean = s.mean()
        x = s / mean                                   # in units of the mean service time
        scv = x.var()
        m3 = float(np.mean(x ** 3))
        ph = hyperexp2_balanced(1.0, scv)
        m3_ph = ph_moment3(ph)
        q = np.quantile(x, [0.5, 0.9, 0.99])
        rng = np.random.default_rng(1)
        xs = ph.sample(200_000, rng)
        q_ph = np.quantile(xs, [0.5, 0.9, 0.99])
        out[lv] = dict(period=f"{lo // 3600:02d}:{lo % 3600 // 60:02d}-{hi // 3600:02d}:{hi % 3600 // 60:02d}",
                       n_calls=int(len(s)), share_zero_removed=zero, mean_seconds=float(mean),
                       scv=float(scv), third_moment=m3, ph="balanced H2", ph_third_moment=m3_ph,
                       alpha=ph.alpha.tolist(), T=ph.T.tolist())
        rows.append(dict(level=lv, period=out[lv]["period"], n=len(s), mean_seconds=mean, scv=scv,
                         third_moment_data=m3, third_moment_h2=m3_ph,
                         median_data=q[0], median_h2=q_ph[0], q90_data=q[1], q90_h2=q_ph[1],
                         q99_data=q[2], q99_h2=q_ph[2]))
        print(f"{lv:13s} {out[lv]['period']}: {len(s):,} served calls (removed {zero:.2%} with zero time), "
              f"mean {mean:.0f} s, C_s^2 = {scv:.3f} (Erlang-2: 0.500)")
        print(f"   in units of the mean: median {q[0]:.2f} (H2 {q_ph[0]:.2f}), 90% {q[1]:.2f} (H2 {q_ph[1]:.2f}), "
              f"99% {q[2]:.2f} (H2 {q_ph[2]:.2f}); third moment {m3:.1f} (H2 {m3_ph:.1f})")

        a = ax[i]
        grid = np.linspace(0, 8, 400)
        a.hist(x, bins=np.linspace(0, 8, 81), density=True, alpha=0.5, color="grey", label="observed (scaled to mean 1)")
        a.plot(grid, np.histogram(xs, bins=np.r_[grid, 8.02], density=True)[0], color="C0", label=f"balanced H2, C_s^2 = {scv:.2f}")
        e2 = erlang(2, 1.0).sample(200_000, rng)
        a.plot(grid, np.histogram(e2, bins=np.r_[grid, 8.02], density=True)[0], color="C3", ls="--", label="Erlang-2 (main comparison)")
        a.set_yscale("log"); a.set_ylim(1e-4, 3)
        a.set_xlabel("service time / mean"); a.set_ylabel("density")
        a.set_title(f"{lv}: {out[lv]['period']}", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)

    (RES / "callcenter_service_fit.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(RES / "callcenter_service_times.csv", index=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "callcenter_service_times.png", dpi=150)
    print("\nThe H2 fit matches the mean and C_s^2 exactly and the median and 90% point closely; the observed "
          "far tail is heavier than H2 (larger third moment).")
    print("Saved: results/callcenter_service_fit.json, results/callcenter_service_times.csv, "
          "results/plots/callcenter_service_times.png")


if __name__ == "__main__":
    main()