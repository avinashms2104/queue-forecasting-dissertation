# Step 3: Baseline Models and Evaluation

## What this is
`train_baseline_models.py` trains and evaluates the models Azam asked for
(Linear/Logistic Regression, Decision Tree, Random Forest — no neural nets),
on both Dataset A (history-only) and Dataset B (rich features), across all
6 rho levels and all 3 forecast horizons, and answers her four questions
directly.

## Headline findings (worth writing up prominently)

### 1. The richer feature set barely helps — sometimes it doesn't help at all
Comparing Random Forest on Dataset A vs. Dataset B (medium horizon, MAE):

| rho  | naive MAE | history-only MAE | rich-features MAE | improvement |
|------|-----------|-------------------|--------------------|-------------|
| 0.30 | 0.25 | 0.25 | 0.25 | +0.3% |
| 0.50 | 0.70 | 0.68 | 0.68 | +0.4% |
| 0.70 | 1.65 | 1.60 | 1.58 | +0.9% |
| 0.80 | 2.14 | 2.10 | 2.09 | +0.4% |
| 0.90 | 2.71 | 2.72 | 2.72 | +0.0% |
| 0.95 | 3.10 | 3.12 | 3.13 | **-0.2%** |

At every utilisation level, adding arrivals/departures/utilisation features
on top of queue-length history improves MAE by **less than 1%** — and at
rho=0.95 it's very slightly *worse*. See `results/plots/dataset_comparison_summary.png`.

**This is a genuinely useful (if humbling) finding, not a failure of the
experiment.** It directly answers Azam's question ("I am particularly
interested in whether the additional queueing features provide a
meaningful improvement... this could become an important part of the
research contribution") — and the answer, at least for this simple M/M/1
setting with these simple models, is *no, not meaningfully*. Worth
discussing WHY: for an M/M/1 queue, the queue-length history already
implicitly encodes the recent arrival/service dynamics (a queue that's been
growing tells you arrivals have recently outpaced departures), so the
"extra" features may be largely redundant information already present in
the lag structure. This might change for more complex systems (M/G/1,
multi-server, priority classes) where queue length alone says less about
*why* the system is behaving as it is — a natural thing to flag as future
work.

### 2. None of the simple models beat the naive baseline by much
Even Random Forest is barely better than "assume the queue length stays
the same" (naive persistence) at every rho and horizon (see
`results/plots/regression_mae_by_rho.png`). This matches Taton et al.'s
finding in your lit review (ensemble methods beat LSTM, but by a modest
margin) and reinforces Velarde et al.'s central lesson: always benchmark
against naive, because "the model works" is not the same as "the model is
worth the complexity."

### 3. Congestion classification recall is near-ZERO at low utilisation
This is the most important number in the whole step, and it's exactly what
Azam asked us to check for ("pay particular attention to whether the model
can correctly identify congested periods rather than only reporting
overall accuracy"):

| rho  | history-only recall (congested class) | rich-features recall |
|------|-----------------------------------------|------------------------|
| 0.30 | 0.00 | 0.00 |
| 0.50 | 0.00 | 0.01 |
| 0.70 | 0.26 | 0.25 |
| 0.80 | 0.43 | 0.43 |
| 0.90 | 0.74 | 0.75 |
| 0.95 | 0.85 | 0.85 |

At rho=0.3 and 0.5, the classifier essentially **never** catches a future
congestion event, despite congestion events existing (~9-12% of the time,
per Step 2's congestion rates) — it's defaulting to the majority class
almost entirely. Overall accuracy would look deceptively good here (since
"not congested" is ~90% of cases), which is exactly why Mitzenmacher &
Shahout's point about asymmetric costs matters: a model that never predicts
congestion is *useless* for the actual purpose (early warning), no matter
how good its accuracy number looks. See
`results/plots/classification_recall_by_rho.png`.

This is worth investigating further in a later step — likely causes: (a)
congestion is genuinely rare and hard to trigger at low rho, since the
system spends most of its time near-empty, and (b) class imbalance is
punishing simple models that optimise for accuracy rather than recall.
Worth trying class-weighting or a different congestion threshold at low
rho as a follow-up.

### 4. Error grows with both utilisation and horizon (as expected)
Both `results/plots/regression_mae_by_rho.png` and
`results/plots/regression_mae_by_horizon.png` show the expected pattern:
harder to forecast further ahead, and harder to forecast at higher
utilisation (more volatile queue dynamics near rho=1, as seen in Step 1's
`example_timeseries.png`).

## Files
- `train_baseline_models.py` — the training/evaluation script
- `results/model_comparison_regression.csv` — full MAE/RMSE table, every
  (dataset, rho, horizon, model) combination
- `results/model_comparison_classification.csv` — full
  accuracy/precision/recall/F1 table, same combinations
- `results/plots/*.png` — the 4 summary plots referenced above

## A caveat worth noting
Logistic Regression showed a convergence warning at rho=0.95 (didn't fully
converge within 1000 iterations) — doesn't invalidate the result but
suggests feature scaling would help if logistic regression is used more
seriously later (Random Forest, which doesn't need scaling, is unaffected
and is the stronger model throughout anyway).

## Next steps (not yet done)
- Write up the congestion DEFINITION discussion in Overleaf (comparing all
  4 candidate definitions Azam listed, not just the one used here)
- Write the formal problem statement (inputs/outputs/horizons) in Overleaf
- Add the ML methods background section
- Investigate the low-rho congestion recall problem further (class
  weighting, alternate thresholds)
- Write up all of this in Overleaf with the plots/tables above
