# Preregistration — EXP-M1: lens port + 9-check gate on Pythia-410M

- experiment-id: EXP-M1
- claim: none (instrument validation; this run is the control record for the
  J-lens instrument, not a scientific claim)
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m1_pythia410m_draw{0,1,2}.yaml (committed)
- author + date: Claude (proposed; tolerances await Ecaterina's sign-off with
  the M1 report), 2026-07-18

## Hypothesis

(VERIFIED-tier basis, narrow scope) The v1 lens gate result on Pythia-410M
skip4_n10 [VERIFIED: CONSTRAINTS "J-lens pipeline reproduces on Pythia-410M"]
reproduces in the v2 environment from vendored byte-identical code: 9/9 gate
checks pass and headline numbers land within the tolerances below. The
PROVISIONAL defaults (skip_first=4, n=10, band L4-L16) are thereby re-derived
on this machine rather than assumed. No HYPOTHESIS-tier statement is tested
by this run.

## Decision rule

All evaluated on the outputs of this run, before any interpretation:

- R1: draw-0 gate verdict is PASS on all 9 checks — criteria (A) and (B) for
  capital-operand, capital-recall, opposites, word-pairs, plus swap
  criterion (C) — exactly as computed by the vendored `jvec.report`.
- R2: draw-0 swap-capitals mean dp(swap_answer) in [0.55, 0.66]
  (v1: +0.6046 ± 0.05); mean random-direction control |dp| <= 0.03
  (v1: +0.0086); top-1 flip rate >= 75% (v1: 87.5%).
- R3: draw-0 capital-recall min-over-band(L4-L16) J-lens HMR <= 5.0
  (v1: 2.5 at L16, tolerance x2), and at that same layer the logit-lens HMR
  is >= 5x the J-lens HMR (v1 ratio: 61.5/2.5 = 24.6x).
- R4: draw-0 calibration prompt sha256 list is exactly v1's manifest list
  (deterministic seeded selection must reproduce bit-for-bit).
- R5: task-baseline include/drop set matches v1 (included: capital-operand,
  capital-recall, opposites, swap-capitals, word-pairs; dropped:
  context-binding, multihop-scaled, typo-robustness) and per-task accuracy
  is within ±3 percentage points of the v1 table.
- R6 (draw stability): draws 1 and 2 (independently re-sampled calibration
  prompts) each produce a PASS gate verdict; headline numbers (swap dp, sham
  dp, per-task best-band J-lens HMR) are reported only as median/IQR over
  the 3 draws.

M1 done = R1-R6 all hold.

## What counts as failure

- Any of R1-R5 fails: v2 does not reproduce v1. Next action is a raw-output
  replay (per-item records in probe.json/swap.json), not interpretation;
  then a flagged report to Ecaterina. M2 does not start.
- R6 fails while R1-R5 hold: the seed-0 result reproduces but the skip4/n10
  lens is draw-fragile. This is itself reportable; the defaults stay
  PROVISIONAL, the failing draw's raw outputs are replayed, and Ecaterina
  rules on whether M2 may start.
- Non-convergence of the fit or a >12h projection from the timing probe:
  abort locally, flag for the A100.

## Estimator plan

- Lens fitting is treated as a stochastic estimator; its draw = the seeded
  calibration-prompt selection. 3 independent draws, seeds 0/1/2, one lens
  per draw, separate cache dirs (cache/draw{k}). Median/IQR over draws for
  every headline number (jtvec.core.draws.DrawSet).
- Task baselines are greedy argmax scoring with no RNG; computed once
  (draw-0 stage) and shared by all draws.

## Instruments

- J-lens readout (instrument under validation). This run is its control
  record: positive control = 4 known-signal probing tasks (criterion A/B),
  negative control = 10-seed Frobenius-matched random-matrix arm per layer
  and the random-direction swap control.
- Logit lens: comparator arm, same forward passes.

## Interventions and shams

- Causal swap (vendored v1 protocol): truncated pinv rcond=0.05,
  source-token positions, norm-preserving, alpha=1.0, band L4-L16.
- Sham: same edit energy onto a random unit direction, 10 seeds per item
  (vendored control), reported in the same table as the swap effect.

## Sample plan

- Tasks: the 8 vendored task JSONs; inclusion by the vendored baseline gate
  (>= 80% in-context top-1).
- N per the task files (v1: 36/36/16/16/24 items for the included set).
- No report-phrasing probes in M1 (those begin at E-experiments).

## Resource estimate

- Fits: 3 x ~902 s (v1 measured this machine, manifest wall_clock_s) ~ 45 min.
- Baselines once ~ 5-10 min; evals + report ~ 15-40 min per draw.
- Total projected ~ 2-3.5 h, peak RSS ~ 2.6-4 GB (v1 fit measured 2.61 GB).
  Well under the 12 h local ceiling; script 01's own timing probe re-checks
  before committing and aborts toward the A100 if projections blow up.

## Deviations

(none at commit time)
