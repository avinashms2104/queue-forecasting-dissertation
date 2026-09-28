"""
Regenerate Figure 1 (M/M/1 simulation vs theory) directly from the simulated
data in data/, so the figure can never drift out of sync with the
simulator again.

For each rho: mean queue length over all 30 runs, compared with the exact
value L_q = rho^2 / (1 - rho). Error bars are +/- 1 standard error of the mean
across the 30 independent runs, so the size of each gap can be judged against
simulation noise (as with the z-scores used for PH/PH/1).

Usage:  python simulation/plot_mm1_validation.py
Needs:  data/mm1_timeseries_rho{rho}.csv   (from mm1_simulator.py)
Writes: results/plots/validation_sim_vs_theory.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RHOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

sim_mean, sim_se, theory = [], [], []
print(f"{'rho':>5} {'simulated':>10} {'+/- SE':>8} {'theory':>8} {'error':>8} {'z':>6}")
for rho in RHOS:
    df = pd.read_csv(DATA_DIR / f"mm1_timeseries_rho{rho}.csv", usecols=["run_id", "queue_length"])
    run_means = df.groupby("run_id")["queue_length"].mean()
    m = df["queue_length"].mean()
    se = run_means.std(ddof=1) / np.sqrt(len(run_means))
    th = rho ** 2 / (1 - rho)
    sim_mean.append(m); sim_se.append(se); theory.append(th)
    print(f"{rho:5.2f} {m:10.3f} {se:8.3f} {th:8.3f} {100 * (m - th) / th:+7.2f}% {(m - th) / se:+6.2f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(RHOS, sim_mean, yerr=sim_se, fmt="o-", capsize=3, label="Simulated (mean of 30 runs, ±1 SE)")
ax.plot(RHOS, theory, "s--", label=r"Theoretical $L_q=\rho^2/(1-\rho)$")
ax.set_xlabel("Traffic intensity (rho)")
ax.set_ylabel("Mean queue length")
ax.set_title("M/M/1 simulation vs theory")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "validation_sim_vs_theory.png", dpi=130)
plt.close(fig)
print("\nSaved: results/plots/validation_sim_vs_theory.png")