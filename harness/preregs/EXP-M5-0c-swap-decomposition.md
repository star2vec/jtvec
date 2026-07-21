# Preregistration — EXP-M5-0c: swap-intervention decomposition (410M vs 1.4B)

- experiment-id: EXP-M5-0c
- status: **RATIFIED** by Ecaterina 2026-07-21 (via session instruction,
  "ratify 0c"). This is the prereg act; the RTX session may build and run it.
  Thresholds ratified as drafted.
- claim: none (instrument diagnostic; decides whether 1.4B's Q2/Q6 failure is a
  genuine potency-scaling effect or a flip-rate/base-margin confound — per
  D-029, no Q2/Q6 bar change without this verdict on record first)
- models: EleutherAI/pythia-410m@9879c9b (M1-certified reference) and
  EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79
- configs: configs/m5_0c_swap_pythia410m.yaml, configs/m5_0c_swap_pythia1p4b.yaml
  (drafted with this file)
- author + date: Claude (proposal), 2026-07-21, per D-029.

## Hypothesis

The EXP-M5-0 lens gate failed Q2 on 1.4B: the swap moved answer probability
strongly (dp 0.483, clearing the 0.30 dp bar) but flipped top-1 only 0.5625
(below the 0.75 bar), and Q6's dp IQR (0.0707) just exceeded 0.05. Two
explanations, this experiment decides between them:

- H-potency: the sham-controlled causal effect of the swap (the logit-gap shift
  toward the swapped answer) is genuinely SMALLER at 1.4B than at 410M — a
  potency-scaling effect (HYPOTHESIS tier; registered as an observation, NOT a
  gate failure, per D-029).
- H-confound: the gap shift is comparable across scales, but 1.4B holds the
  original answer at a HIGHER base margin, so the same shift flips top-1 less
  often — the top-1 flip rate is a base-margin-confounded metric, and a
  margin-normalized flip statistic is the correct recalibrated Q2 measure.

## Decision rule (bars [proposed] until ratified)

Per substrate, per matched task, over >= 3 lens draws (median/IQR), all three
quantities computed under the IDENTICAL statistic across substrates:

1. sham-controlled logit-gap-shift = (gap-shift under the real swap) minus
   (gap-shift under the norm-matched sham), where gap = logit(swapped answer) −
   logit(original answer) at the final position. Distribution reported (median,
   IQR, per-item).
2. top-1 flip rate (the current Q2 metric).
3. margin-normalized flip statistic (CANDIDATE recalibrated metric) =
   fraction of items whose sham-controlled gap-shift exceeds the item's base
   margin gap(original) − gap(runner-up) at the pre-swap final position — i.e.
   "did the swap move enough to flip THIS item, given how dominant its base
   answer was", decoupled from whether the argmax happened to cross.

Readout of the decision:
- H-potency SUPPORTED iff 1.4B's median sham-controlled gap-shift is
  materially below 410M's (>= [proposed 0.15 logit units] lower, non-overlapping
  IQRs) AND the margin-normalized flip statistic is ALSO low at 1.4B (the swap
  genuinely under-moves relative to the margin). Registered as a potency-scaling
  observation (HYPOTHESIS tier); NOT a gate failure.
- H-confound SUPPORTED iff the sham-controlled gap-shifts are comparable across
  scales (overlapping IQRs) but the margin-normalized flip statistic is HIGH at
  1.4B while the raw top-1 flip is low — i.e. the swap moves enough given the
  margins, the argmax just did not cross. Then the margin-normalized flip is
  proposed as the recalibrated Q2 metric (a later EXP-M5-0 amendment).
- Mixed → reported per-substrate, per-task; no outcome unpublishable.

This experiment does NOT itself change any Q2/Q6 bar (D-029): it puts the
verdict on record; any recalibration is a separate ratified amendment.

## What counts as failure

- Instrument-control failure (below) voids the verdict (instruments LAW).
- A substrate/task where the swap does not move the answer above sham even at
  410M (the certified reference) indicates a broken swap pipeline, not a
  scaling result — investigated, not interpreted.
- Post-hoc analyses of stored tensors are labelled post-hoc forever.

## Estimator plan

The estimator is the M1 lens-based causal swap (jvec.evals.swap, truncated
pinv rcond 0.05, source-token-position edit, norm preservation) — the same
instrument Q2 uses. 3 lens draws per substrate = the certified/cached draws
(410M cache/draw{0,1,2}; 1.4B cache/m5/p14b_draw{0,1,2}). Only the lens-draw
RNG varies; median/IQR over draws (DrawSet). Deterministic readout given a
lens. The gap-shift, flip, and margin-normalized statistics are computed by a
new post-processing pass over the swap's per-item logit records (no new model
RNG).

## Instruments

- swap-effect readout: positive control = on 410M the sham-controlled gap-shift
  median >= [proposed 0.30 logit units] (the M1-certified swap works, dp +0.60
  reference); negative control = sham gap-shift median within [−0.03, 0.03]
  logit units on both substrates. require_controlled() gates the verdict.
- identical statistic across substrates: same gap definition, same
  margin-normalization, same layers/positions, same sham construction, same
  draw handling — enforced by a shared code path parameterized only by
  substrate config.

## Sample plan

- Matched tasks: swap-capitals (the Q2 task) plus >= 1 further certified swap
  pair the model does behaviourally on both substrates (capital-recall-derived
  swap), N recorded per task; probe only items both substrates get right
  pre-swap (matched item set across substrates).
- Draws: 3 per substrate.
- Raw per-item logit records retained per (substrate, task, draw, arm ∈
  {swap, sham}): base gap, post-swap gap, runner-up gap, flip bit.

## Resource estimate (TIER: RTX laptop, D-008 stack, fp16/fp32 per the win32 config)

Swap eval is cheap (no fit; forward + pinv edit per item). 410M ~minutes/draw;
1.4B ~4x. 2 substrates x 2 tasks x 3 draws + the sham arms → projected
~1.5-2.5 h on the RTX; well under the 12 h ceiling. Detached + Monitor per the
background-kill lesson. (This is the tier Ecaterina runs; nothing here depends
on the Mac.)

## Deviations

(none yet — draft)

## Ratification

ratified: EXP-M5-0c 2026-07-21 — Ecaterina (via session instruction "ratify 0c").
