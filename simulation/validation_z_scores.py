"""
Standard-error check for the simulator validation of every queue type:
M/M/1, M/G/1 (two shapes), M/PH/1, PH/PH/1, H2/PH/1 and MAP/PH/1.

Is the gap between the simulated and the theoretical mean queue length larger than
simulation noise? For each setting:

    z = (mean queue length over all 30 runs - theoretical L_q) / SE

where SE is the standard deviation of the 30 per-run means divided by sqrt(30).
|z| below about 2 means the gap is within simulation noise.

Theory: M/M/1 -> rho^2/(1-rho); M/G/1 and M/PH/1 -> Pollaczek-Khinchine;
PH/PH/1, H2/PH/1 and MAP/PH/1 -> exact QBD (matrix-analytic) solution, since
P-K needs Poisson arrivals. A stage whose data files are missing is skipped with a note.

Usage:  python simulation/validation_z_scores.py
Needs:  the simulated data in data/ (from the seven simulators)
Writes: results/validation_z_scores.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from mg1_simulator import (weibull_scale_for_mean, weibull_variance,
                           theoretical_Lq_pollaczek_khinchine, MEAN_SERVICE_TIME, RHOS)
from ph_utils import erlang, hyperexp2_balanced
from map_utils import bursty_mmpp2, ph_renewal_map
from qbd_solver import solve_qbd, solve_phph1

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# (label, file for a given rho, theoretical L_q for a given rho)
STAGES = [
    ("M/M/1", lambda r: DATA / f"mm1_timeseries_rho{r}.csv",
     lambda r: r ** 2 / (1 - r)),
]
for shape in [2.0, 0.5]:
    var = weibull_variance(shape, weibull_scale_for_mean(shape))
    STAGES.append((f"M/G/1 Weibull {shape}",
                   (lambda s: (lambda r: DATA / "mg1" / f"mg1_timeseries_shape{s}_rho{r}.csv"))(shape),
                   (lambda v: (lambda r: theoretical_Lq_pollaczek_khinchine(r, r, MEAN_SERVICE_TIME, v)))(var)))
STAGES.append(("M/PH/1 Erlang-2", lambda r: DATA / "mph1" / f"mph1_timeseries_erlang2_rho{r}.csv",
               lambda r: theoretical_Lq_pollaczek_khinchine(r, r, MEAN_SERVICE_TIME, 0.5)))   # Erlang-2, mean 1: var 1/2
STAGES.append(("PH/PH/1 Erlang-2/Erlang-2", lambda r: DATA / "phph1" / f"phph1_timeseries_arrE2_srvE2_rho{r}.csv",
               lambda r: solve_phph1(erlang(2, 1 / r), erlang(2, MEAN_SERVICE_TIME))["Lq"]))
STAGES.append(("H2/PH/1 H2/Erlang-2", lambda r: DATA / "h2ph1" / f"h2ph1_timeseries_rho{r}.csv",
               lambda r: solve_qbd(*ph_renewal_map(hyperexp2_balanced(1 / r, bursty_mmpp2(1.0).scv)).as_qbd_input(),
                                   erlang(2, MEAN_SERVICE_TIME))["Lq"]))
STAGES.append(("MAP/PH/1 MMPP-2/Erlang-2", lambda r: DATA / "mapph1" / f"mapph1_timeseries_rho{r}.csv",
               lambda r: solve_qbd(*bursty_mmpp2(r).as_qbd_input(), erlang(2, MEAN_SERVICE_TIME))["Lq"]))

rows = []
for label, path_fn, theory_fn in STAGES:
    if not path_fn(RHOS[0]).exists():
        print(f"(skipping {label}: no data files found)")
        continue
    for rho in RHOS:
        df = pd.read_csv(path_fn(rho), usecols=["run_id", "queue_length"])
        run_means = df.groupby("run_id")["queue_length"].mean()
        sim = df["queue_length"].mean()
        se = run_means.std(ddof=1) / np.sqrt(len(run_means))
        th = theory_fn(rho)
        rows.append(dict(queue=label, rho=rho, simulated=sim, std_error=se, theory=th,
                         error_pct=100 * (sim - th) / th, z=(sim - th) / se, n_runs=len(run_means)))

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "validation_z_scores.csv", index=False)

pd.set_option("display.width", 200)
for label in res.queue.unique():
    print(f"\n{label}")
    print(res[res.queue == label][["rho", "simulated", "std_error", "theory", "error_pct", "z"]]
          .round({"simulated": 3, "std_error": 3, "theory": 3, "error_pct": 2, "z": 2}).to_string(index=False))
print(f"\nSettings: {len(res)}   |z| > 2 in {int((res.z.abs() > 2).sum())}   largest |z| = {res.z.abs().max():.2f} "
      f"({res.loc[res.z.abs().idxmax(), 'queue']}, rho = {res.loc[res.z.abs().idxmax(), 'rho']})")
print("Saved: results/validation_z_scores.csv")