# Machine Learning for Queue State Forecasting and Congestion Prediction

Dissertation code (DATASCI 792), supervised by Azam Asanjarani. The project simulates
queues of increasing complexity, checks each simulator against exact queueing theory,
and tests how well simple machine-learning models forecast the queue length and
whether the queue will be congested, always against a naive persistence baseline.

The research notebook (`main.tex`) holds the full write-up and reasoning. This README
is a map of the code. Last updated 28 September 2026.

## Stages completed

| Stage | Arrivals | Service | Validated against |
|---|---|---|---|
| M/M/1 | Poisson | Exponential | L_q = rho^2 / (1 - rho) |
| M/G/1 | Poisson | Weibull, shape 2.0 and 0.5 | Pollaczek-Khinchine |
| M/PH/1 | Poisson | Erlang-2 | Pollaczek-Khinchine |
| PH/PH/1 | Erlang-2 | Erlang-2 | Exact QBD (matrix-analytic) solution |

Next stage in the plan: MAP/PH/1 (Markovian arrival process). The QBD solver already
accepts MAP matrices (D0, D1), so it can be reused directly.

## Method in one paragraph

Each queue is simulated exactly with the Lindley recursion and sampled on a regular grid
(dt = 1, in units of mean service time). For each traffic intensity rho in
{0.3, 0.5, 0.7, 0.8, 0.9, 0.95} there are 30 independent runs of 20,000 recorded time
units after a warm-up that scales with 1/(1-rho)^2 and with the variability of the
arrival and service processes. Runs 0-19 are used for training, 20-24 for validation
and 25-29 for testing (the split is by whole run, so no run contributes rows to both
training and testing). Inputs are the last 10 queue lengths ("history-only") or a richer
set with arrivals, departures and rate estimates. Targets are the queue length and a
congestion label (queue length above the 80th percentile of the training runs) at
horizons h = 5, 10, 15 and 20. Models: Linear/Logistic Regression, Decision Tree,
Random Forest, XGBoost. Baselines: naive persistence for both tasks, and a constant
"typical value" predictor. Results are means over the 5 test runs.

## Simulator validation (mean queue length L_q, mean over 30 runs)

M/M/1 (from `simulation/plot_mm1_validation.py`):

| rho | simulated | theory | error | z |
|---|---|---|---|---|
| 0.30 | 0.128 | 0.129 | -0.65% | -0.78 |
| 0.50 | 0.499 | 0.500 | -0.15% | -0.12 |
| 0.70 | 1.665 | 1.633 | +1.95% | +1.81 |
| 0.80 | 3.210 | 3.200 | +0.32% | +0.18 |
| 0.90 | 8.096 | 8.100 | -0.05% | -0.02 |
| 0.95 | 19.131 | 18.050 | +5.99% | +0.79 |

PH/PH/1, Erlang-2 arrivals and Erlang-2 service (from `simulation/phph1_simulator.py`,
exact values from the QBD solver):

| rho | simulated | exact (QBD) | error | z |
|---|---|---|---|---|
| 0.30 | 0.039 | 0.039 | +0.14% | +0.09 |
| 0.50 | 0.195 | 0.195 | -0.03% | -0.04 |
| 0.70 | 0.722 | 0.727 | -0.70% | -0.74 |
| 0.80 | 1.494 | 1.492 | +0.14% | +0.13 |
| 0.90 | 3.821 | 3.923 | -2.61% | -1.29 |
| 0.95 | 8.576 | 8.888 | -3.51% | -0.64 |

Here z is the gap divided by its standard error over the 30 runs, so |z| below 2 means
the gap is within simulation noise. The M/G/1 and M/PH/1 validation tables are in the
notebook (Sections 7.3 and 7.4). Pollaczek-Khinchine no longer applies once arrivals
are not Poisson, and Kingman's approximation is too rough at low rho (it overestimates
L_q by 64% at rho = 0.3 for PH/PH/1), which is why PH/PH/1 is checked against the exact
QBD solution. The solver itself was first checked against M/M/1 and M/PH/1 (agreement
to 1e-10).

## Main findings so far (medium horizon, history-only features, 5 test runs)

- **Classification:** naive persistence has a higher F1 than Random Forest in 29 of the
  30 (queue type x rho) settings; the other is a tie to three decimals (Weibull 0.5,
  rho = 0.95, difference +0.0002). XGBoost behaves like Random Forest. Rebalancing (class weights, over/undersampling) only slides along one
  precision-recall trade-off (M/M/1).
- **Regression depends on the metric.** On RMSE, Random Forest is significantly better
  than persistence in 28 of 30 settings (paired t-test, Holm-corrected). On MAE it is
  better in only 9 and significantly worse in 2. A model trained on absolute error
  beats persistence on MAE in 27 of 30 settings, but at low rho a constant "typical
  value" predictor does as well, so persistence is a weak MAE baseline there. At high
  rho (0.9 and above) nothing beats persistence on MAE by more than about 5%.
- **Variability drives persistence.** Higher service-time variance gives longer-lasting
  queue states. Smoothing the arrivals (Poisson to Erlang-2, service fixed at Erlang-2) shortens
  the half-life of queue-length autocorrelation by about a third (by half at rho = 0.3)
  wherever a half-life exists.
- **Richer features do not help.** For M/M/1, M/PH/1, PH/PH/1 and Weibull 2.0 the Random
  Forest MAE is within about +/-0.1% of the history-only version at every rho and
  horizon; for Weibull 0.5 the gain is up to 1.5% at short horizons.

All of this is simulation-only, uses 5 test runs per setting (low statistical power),
and uses a percentile-based congestion label. See the notebook for caveats.

## Files

`simulation/`
- `mm1_simulator.py`, `mg1_simulator.py`, `mph1_simulator.py`, `phph1_simulator.py`: the simulators
- `ph_utils.py`: phase-type distributions (moments, sampling, Erlang/hyperexponential helpers)
- `qbd_solver.py`: exact PH/PH/1 and MAP/PH/1 solution, plus Pollaczek-Khinchine and Kingman formulas
- `check_ph_utils.py`, `check_qbd_solver.py`: self-checks against known results
- `plot_mm1_validation.py`: regenerates the M/M/1 validation figure from the data

`modeling/` (per stage: build datasets, train, autocorrelation, extra summaries)
- `build_datasets.py`, `build_mg1_datasets.py`, `build_mph1_datasets.py`, `build_phph1_datasets.py`
- `train_baseline_models.py`, `train_mg1_baseline_models.py`, `train_mph1_baseline_models.py`, `train_phph1_baseline_models.py`
- `autocorrelation_analysis.py`, `mg1_...`, `mph1_...`, `phph1_autocorrelation_analysis.py`
- `mg1_extra_summaries.py`, `mph1_extra_summaries.py`, `phph1_extra_summaries.py`
- `class_weight_threshold_tuning.py`, `resampling_experiment.py`, `horizon_sweep_rho08.py` (M/M/1)
- Cross-stage: `compare_service_distributions.py`, `compare_arrival_distributions.py`,
  `summarise_regression_models.py` (MAE and RMSE, all models), `significance_tests.py`
  (paired tests), `train_xgboost_models.py`, `check_threshold_leakage.py`

`results/` holds the CSV outputs and `results/plots/` the figures.

## Reproducing

`data/` is git-ignored (the raw time series are large) and is rebuilt by the simulators,
which use fixed seeds. For a stage, run in order: simulator, `build_*_datasets.py`,
`train_*_baseline_models.py`, then the analysis scripts. For example, PH/PH/1:

```
python simulation/phph1_simulator.py
python modeling/build_phph1_datasets.py
python modeling/train_phph1_baseline_models.py
python modeling/phph1_autocorrelation_analysis.py
python modeling/phph1_extra_summaries.py
```

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, xgboost.

## Note on the congestion threshold

The 80th-percentile threshold is computed from the 20 training runs only in the PH/PH/1
and M/G/1 pipelines. The M/M/1 and M/PH/1 pipelines computed it from all 30 runs;
`modeling/check_threshold_leakage.py` shows the two versions give identical labels at
every rho for those two, so nothing was rerun. For M/G/1 three of twelve settings
differed slightly and were rebuilt and retrained (the mean F1 changed by at most about 0.011 across horizons, and by 0.006 or less at the medium horizon).

## Earlier notes

`modeling/README_step2.md` and `modeling/README_step3.md` are the notes from the first
round of experiments and are superseded (they predate the run-level split and the
persistence baselines).