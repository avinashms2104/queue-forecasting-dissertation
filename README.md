# Machine Learning for Queue State Forecasting and Congestion Prediction

Dissertation code (DATASCI 792), supervised by Azam Asanjarani. The project simulates
queues of increasing complexity, checks each simulator against exact queueing theory,
and tests how well simple machine-learning models forecast the queue length and
whether the queue will be congested, always against a naive persistence baseline.

The research notebook (`main.tex`) holds the full write-up and reasoning. This README
is a map of the code. Last updated 14 October 2026.

## Stages completed

The plan (M/M/1 -> M/G/1 -> M/PH/1 -> PH/PH/1 -> MAP/PH/1) is complete, plus one extra
control stage (H2/PH/1), a controlled experiment on arrival dependence (three MAP levels),
an early-warning / cost-of-under-prediction analysis, and a case study with MAPs fitted
to real call-centre arrivals.

| Stage | Arrivals | Service | Validated against |
|---|---|---|---|
| M/M/1 | Poisson | Exponential | L_q = rho^2 / (1 - rho) |
| M/G/1 | Poisson | Weibull, shape 2.0 and 0.5 | Pollaczek-Khinchine |
| M/PH/1 | Poisson | Erlang-2 | Pollaczek-Khinchine |
| PH/PH/1 | Erlang-2 (C_a^2 = 0.5) | Erlang-2 | Exact QBD (matrix-analytic) solution |
| H2/PH/1 | Hyperexponential (C_a^2 = 1.684), independent gaps | Erlang-2 | Exact QBD solution |
| MAP/PH/1 | 2-state MMPP (C_a^2 = 1.684), correlated gaps | Erlang-2 | Exact QBD solution |
| MAP/PH/1, weak / strong | MMPP-2s with shorter / longer bursts (see below) | Erlang-2 | Exact QBD solution |
| MAP/PH/1, call centre | MMPP-2s fitted to real arrivals, two stable periods | Erlang-2 | Exact QBD solution |

**The MAP.** Azam did not specify one, so MAP/PH/1 uses a two-state Markov-modulated
Poisson process: a busy spell with 4x the arrival rate of a quiet spell, busy 25% of the
time, about 10 arrivals per busy spell. Gap C_a^2 = 1.684, lag-1 gap correlation 0.165,
limiting index of dispersion 4.6. All rates scale with lambda = rho, so the shape is the
same at every rho. Changing the MAP (for example to one fitted to real call-centre data)
only means editing `make_arrivals()` in `simulation/mapph1_simulator.py`.

**Three controlled dependence levels (Azam, 12 Oct).** `map_utils.map_level()` gives three
MMPP-2s with the same arrival rate, the same gap variability (C_a^2 = 1.684) and the same
arrival-count variability over a window of 10 mean interarrival times (variance-to-mean
ratio 3.486), so that only the dependence structure differs: about 5 (weak), 10 (medium,
the original MAP) and 40 (strong) arrivals per busy spell. Long-run index of dispersion
3.72 / 4.62 / 9.63; gap correlation at lag 20: 0.001 / 0.003 / 0.039; correlation of counts
in neighbouring 10-gap windows: 0.03 / 0.15 / 0.45. `simulation/check_map_levels.py`
verifies all of this against 30 simulated sequences of 200,000 arrivals per level.
Every MAP/PH/1 script takes `--level weak|medium|strong|cc_afternoon|cc_evening`.

**Call-centre MAPs.** `simulation/callcenter_fit_map.py` reads the Technion "Anonymous Bank"
data (November and December 1999, weekdays Sunday-Thursday, 07:00-24:00, calls that asked
for an agent), finds the stable periods 11:00-15:30 and 19:30-22:30 (every half-hour mean
within 10% of the period mean), and fits an MMPP-2 to each (gap C_a^2, lag-1 correlation
and the within-day count variability curve). The real arrivals are much less bursty than
the controlled MAPs: count variability over 10 gaps 1.06 (afternoon) and 1.32 (evening),
against 3.49. The raw data are not in the repository; put `november.txt` and
`december.txt` in `data/callcenter/raw/`.

**Why H2/PH/1.** Its hyperexponential arrivals are fitted to the same mean and C_a^2 as
the MMPP, and service is the same Erlang-2, so H2/PH/1 and MAP/PH/1 differ ONLY in the
correlation between successive arrival gaps. Together with PH/PH/1 and M/PH/1 this gives
an arrival-side ladder with service fixed: smooth -> random -> variable -> variable and
correlated.

## Method in one paragraph

Each queue is simulated exactly with the Lindley recursion and sampled on a regular grid
(dt = 1, in units of mean service time). For each traffic intensity rho in
{0.3, 0.5, 0.7, 0.8, 0.9, 0.95} there are 30 independent runs of 20,000 recorded time
units after a warm-up that scales with 1/(1-rho)^2 and with the variability of the
arrival and service processes (for MAP arrivals the index of dispersion replaces C_a^2).
Runs 0-19 are used for training, 20-24 for validation and 25-29 for testing (the split is
by whole run, so no run contributes rows to both training and testing). Inputs are the
last 10 queue lengths ("history-only") or a richer set with arrivals, departures and rate
estimates. Targets are the queue length and a congestion label (queue length above the
80th percentile of the training runs) at four horizons. Models: Linear/Logistic
Regression, Decision Tree, Random Forest, XGBoost. Baselines: naive persistence for both
tasks, a constant "typical value" predictor (regression) and "always congested"
(classification). Results are means over the 5 test runs.

## Simulator validation

`simulation/validation_z_scores.py` checks the 42 settings of the 7 main queue types
(7 x 6 rho); the four newer MAP/PH/1 settings print their own z-scores
(`results/mapph1_{weak,strong,cc_afternoon,cc_evening}_validation.csv`).
z = (simulated L_q - theoretical L_q) / standard error over the 30 runs.

| Queue | largest abs(z) over the 6 rho |
|---|---|
| M/M/1 | 1.81 |
| M/G/1 Weibull 2.0 | 1.56 |
| M/G/1 Weibull 0.5 | 1.23 |
| M/PH/1 Erlang-2 | 1.93 |
| PH/PH/1 E2/E2 | 1.29 |
| H2/PH/1 | 2.40 (rho = 0.5) |
| MAP/PH/1 | 1.98 |
| MAP/PH/1 weak | 1.75 |
| MAP/PH/1 strong | 1.43 |
| MAP/PH/1 call centre, afternoon | 1.34 |
| MAP/PH/1 call centre, evening | 2.20 (rho = 0.9) |

Two of 66 settings have abs(z) > 2, against about 3 expected by chance at the 5% level.
One (H2/PH/1, rho = 0.5) was rechecked with 150 runs instead of 30 and gave z = +0.88
(and z = -0.01 with a longer warm-up), so there is no bias.

The QBD solver with MAP input is checked in `simulation/check_map_utils.py`: a Poisson
MAP reproduces Pollaczek-Khinchine (to 4e-11), an Erlang-2 renewal MAP reproduces the
PH/PH/1 solver (identical), P(empty) = 1 - rho holds exactly, and an independent
brute-force solution of the full truncated Markov chain agrees to 1e-8 or better.
Pollaczek-Khinchine does not apply once arrivals are not Poisson, and Kingman's
approximation is too rough at low rho (64% off at rho = 0.3 for PH/PH/1).

## Main findings (medium horizon, history-only features, 5 test runs)

- **Congestion (classification).** Persistence has a higher F1 than Random Forest in 38
  of 42 settings; in the other 4 the difference is 0.001 or less. XGBoost behaves the
  same. Repeated for all seven queues, class weighting with a validation-tuned cut-off
  beats persistence by more than 0.01 in 11 of 42 settings, but 8 of those 11 equal
  exactly what the trivial "always congested" rule scores (all at rho <= 0.5, where the
  tuned cut-off flags almost everything); the other 3 gains are 0.011 to 0.014.
  Over- and undersampling are worse than persistence by more than 0.01 in 26 of 42
  settings.
- **Queue length (regression) depends on the metric.** On RMSE, Random Forest is
  significantly better than persistence in 39 of 42 settings (paired t-test,
  Holm-corrected); on MAE in only 11, and significantly worse in 2. XGBoost trained on
  absolute error beats persistence on MAE in 39 of 42 settings, but beats a constant
  "typical value" predictor in only 29, and that constant itself beats persistence in 17
  (all at rho <= 0.7). Real skill on MAE appears mainly at rho >= 0.7.
- **Longer horizons leave more room for a model.** At rho = 0.8 the Random Forest MAE
  gain over persistence grows steadily from h = 5 to h = 40 for every queue, reaching
  13.6-16.8% at h = 40 for five of the queues but only 7.7% for MAP/PH/1 and 4.4% for
  Weibull 0.5, the two queues with the longest memory.
- **Memory explains persistence, and correlated arrivals create the most memory.** With
  service fixed at Erlang-2, the half-life of queue-length autocorrelation at rho = 0.7 is
  6 (Erlang-2 arrivals), 9 (Poisson), 12 (H2, C_a^2 = 1.684) and 35 (MAP, same C_a^2 but
  correlated). Correlation alone, at the same C_a^2, makes the queue about 2.5x longer
  (exact L_q 3.44 vs 8.54 at rho = 0.8) and roughly triples its memory. Accordingly,
  Random Forest beats persistence on MAE at every rho for H2/PH/1 but not at low rho for
  MAP/PH/1.
- **Richer features do not help (until dependence is strong, see below).** Averaged
  over rho and horizons the Random Forest MAE gain from rich features is 0.34% for
  MAP/PH/1 and 0.04% for H2/PH/1, in line with the earlier queues (largest: up to 1.5%
  for Weibull 0.5 at short horizons).
- **Arrival dependence on its own (three controlled MAP levels).** With rate, gap
  variability and 10-gap count variability fixed, longer bursts make the queue longer
  (exact L_q at rho = 0.95: 39 / 47 / 90 for weak / medium / strong) and its memory longer
  (half-life at rho = 0.8: 61 / 63 / 116). The additional arrival features help only when
  the dependence is strong: +1.6 to +4.4 points of MAE gain for strong, 0.2-0.7 for medium,
  about 0 for weak and H2.
- **Early warning and the cost of under-prediction** (`early_warning_analysis.py`).
  Persistence can never warn before congestion starts. With a missed congested step
  counted as 5x a false alarm, the Random Forest classifier lowers the per-step cost
  against persistence by 10-24% (no dependence) and 34-44% (strong dependence) at
  rho = 0.8-0.9, and warns at least 5 steps ahead for 63-70% of congestion onsets with
  strong dependence, but at most 11% with weak dependence. A simple lowered-threshold
  rule gets much of this; ML clearly beats it only with strong dependence. For
  queue-length forecasts under asymmetric cost, quantile gradient boosting beats
  persistence plus a safety margin by 6-14% with strong dependence and 0-2% otherwise.
- **The call centre.** In every measure the two call-centre settings sit next to the
  no-dependence case (H2), not the bursty MAPs: Random Forest's MAE gain over
  persistence is between -0.5% and +4%, the additional features add nothing, and early
  warning cuts the cost by about 20% at rho = 0.9, as much as the simple rule.
  So the conclusions from the controlled experiments carry over (`callcenter_summary.py`).

All of this uses 5 test runs per setting (low statistical power), a percentile-based
congestion label, one family of MAPs (MMPP-2), fixed model settings, and Erlang-2 service
also in the call-centre case. See the notebook for caveats.

## Files

`simulation/`
- Simulators: `mm1_simulator.py`, `mg1_simulator.py`, `mph1_simulator.py`,
  `phph1_simulator.py`, `h2ph1_simulator.py`, `mapph1_simulator.py`
- `ph_utils.py`: phase-type distributions (moments, sampling, Erlang/hyperexponential helpers)
- `map_utils.py`: Markovian arrival processes (exact gap moments and correlations, index of
  dispersion, exact count variance / variance-to-mean ratio / count correlation, sampler,
  MMPP-2 constructor, the three controlled levels, the call-centre MAPs, `LEVEL_INFO`)
- `check_map_levels.py`: rate, gap and count variability and autocorrelations of the three
  levels, exact against simulated (writes `plots/map_levels_properties.png`)
- `callcenter_fit_map.py`: call-centre arrivals, rate profile, stable periods, MAP fit
- `qbd_solver.py`: exact PH/PH/1 and MAP/PH/1 solution, plus Pollaczek-Khinchine and Kingman formulas
- `check_ph_utils.py`, `check_qbd_solver.py`, `check_map_utils.py`: self-checks against known results
- `validation_z_scores.py`: the 42-setting validation table
- `plot_mm1_validation.py`: regenerates the M/M/1 validation figure from the data

`modeling/` (per stage: build datasets, train, autocorrelation, extra summaries)
- `build_datasets.py`, `build_{mg1,mph1,phph1,h2ph1,mapph1}_datasets.py`
- `train_baseline_models.py`, `train_{mg1,mph1,phph1,h2ph1,mapph1}_baseline_models.py`
- `autocorrelation_analysis.py`, `{mg1,mph1,phph1,h2ph1,mapph1}_autocorrelation_analysis.py`
- `{mg1,mph1,phph1,h2ph1,mapph1}_extra_summaries.py`
- All seven queues: `class_weight_all_queues.py` (class weighting, cut-off tuned on the
  validation runs), `resampling_all_queues.py` (over/undersampling, "always congested"
  reference), `horizon_sweep_all_queues.py` (h = 5 to 40 at rho = 0.8)
- Cross-stage: `compare_service_distributions.py`, `compare_arrival_distributions.py`,
  `summarise_regression_models.py` (MAE and RMSE, all models), `significance_tests.py`
  (paired tests), `train_xgboost_models.py`, `check_threshold_leakage.py`
- MAP dependence and the call centre: `map_dependence_summary.py` (H2 vs the three
  levels), `horizon_sweep_rich_features.py --map-levels`, `mae_rmse_investigation.py
  --map-levels`, `class_imbalance_summary.py --map-levels`, `early_warning_analysis.py`
  (and `--callcenter`), `callcenter_summary.py` (real vs controlled)
- Original M/M/1-only versions, kept so earlier notebook numbers stay reproducible:
  `class_weight_threshold_tuning.py` (its cut-off was tuned on the test runs; the
  all-queue version fixes this), `resampling_experiment.py`, `horizon_sweep_rho08.py`

`results/` holds the CSV outputs and `results/plots/` the figures.

## Reproducing

`data/` is git-ignored (the raw time series are large) and is rebuilt by the simulators,
which use fixed seeds. For a stage, run in order: simulator, `build_*_datasets.py`,
`train_*_baseline_models.py`, then the analysis scripts. For example, MAP/PH/1:

```
python simulation/check_map_utils.py
python simulation/mapph1_simulator.py
python modeling/build_mapph1_datasets.py
python modeling/train_mapph1_baseline_models.py
python modeling/mapph1_autocorrelation_analysis.py
python modeling/mapph1_extra_summaries.py
```

For the dependence levels and the call centre:

```
python simulation/check_map_levels.py
python simulation/callcenter_fit_map.py              # needs data/callcenter/raw/*.txt
python simulation/mapph1_simulator.py --level strong # likewise weak, cc_afternoon, cc_evening
python modeling/build_mapph1_datasets.py --level strong
python modeling/train_mapph1_baseline_models.py --level strong
python modeling/mapph1_autocorrelation_analysis.py --level strong
python modeling/map_dependence_summary.py
python modeling/early_warning_analysis.py
python modeling/early_warning_analysis.py --callcenter
python modeling/callcenter_summary.py
```

The cross-stage and all-queue scripts need every stage's data and results to exist.

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, xgboost.

Reproducibility note: seeds are fixed, but the H2/PH/1 runs at rho = 0.5 came out
slightly differently on Windows and Linux (probably floating-point differences in a few
runs; every other setting matched digit for digit). The committed results are from the
Windows machine.

## Note on the congestion threshold

The 80th-percentile threshold is computed from the 20 training runs only in the M/G/1,
PH/PH/1, H2/PH/1 and MAP/PH/1 pipelines. The M/M/1 and M/PH/1 pipelines computed it from
all 30 runs; `modeling/check_threshold_leakage.py` shows the two versions give identical
labels at every rho for those two, so nothing was rerun. For M/G/1 three of twelve
settings differed slightly and were rebuilt and retrained (the mean F1 changed by at most
about 0.011 across horizons, and by 0.006 or less at the medium horizon). Over all 42
settings the two thresholds differ in 6, affecting 0.24% of labels.