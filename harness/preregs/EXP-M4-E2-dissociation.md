# Preregistration — EXP-M4-E2-dissociation: execution vs verbalization on singular-plural

- experiment-id: EXP-M4-E2-dissociation
- claim: CLM-002
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m4_e2_dissociation_pythia410m.yaml
- author + date: Claude (proposal), 2026-07-19; scope ruled by Ecaterina
  2026-07-19 (D-016, Path A + "build as designed, high-N"); thresholds
  ratified by Ecaterina 2026-07-19 (D-017, session Q&A). Committing this file
  is the prereg act.

## Hypothesis

HYPOTHESIS (CONSTRAINTS.md): task execution and task verbalization are causally separable; v1 Exp-3 reported a double dissociation on singular-plural (single draw, single model, N=60, human-verification never completed). E2 re-tests the singular-plural arm on v2's gated apparatus (D-016 Path A; landmark-country deferred, its FV + instruments uncertified). Cross-draw ablation transfer is built in per CONSTRAINTS. Nothing below is asserted; the run decides.

## Decision rule

Two measures on singular-plural: execution = greedy top-1 accuracy on
held-out queries (exact-match, case-sensitive; D-012 answer_first_tokens);
report = report_score under P3 = log p(" plural" | coherent+P3) − neutral
baseline (the report-score-prior-corrected@singular-plural instrument gated
2026-07-19, P3 only). Two ablations at the final position of band layers
4–16, each with its M3 matched sham: fv-direction (project out the certified
Todd FV) and jspace (project out the top-10 J-lens-readout atoms).

Cross-draw: the fv ablation is re-derived from each of the 3 M2-certified FV
draws; the jspace ablation from each of the 3 M1 lens draws (jspace reads the
lens — the E1 lens-draw nuisance). Context sets (exec queries; report
contexts) are sampled ONCE (rng 5252) and reused across every condition, so
each effect is a paired clean-vs-ablated contrast. Every ablation effect is a
DrawSet over its 3 draws: effect_k = clean − ablated_k (positive = hurt);
median/IQR reported; the sham effect is the matched DrawSet.

An ablation HURTS a measure iff effect_median − sham_median ≥ δ, with
δ_exec = 0.15 and δ_report = 0.10 (log-prob). Directions:
- D1 (fv dissociates execution FROM report): fv hurts execution AND NOT report.
- D2 (jspace dissociates report FROM execution): jspace hurts report AND NOT execution.

Verdict: DOUBLE-DISSOCIATION iff D1 AND D2; ONE-WAY iff exactly one;
NO-DISSOCIATION otherwise. Cross-draw transfer flag per key arm = every one
of its 3 draws clears sham_median + δ (a median that fires while draws
disagree is reported as non-transferring, not a clean hurt). CLM-002 moves
hypothesis → preliminary iff the verdict is DOUBLE-DISSOCIATION with both
transfer flags true; any other outcome is recorded with the full effect
table and CLM-002 stays hypothesis.

[Thresholds await Ecaterina's ratification; the constants in
jtvec/e2_dissociation.py and scripts/m4_e2_dissociation.py match this file.]

## What counts as failure

- NO-DISSOCIATION, or fv/jspace effects not separable from their shams:
  evidence against the singular-plural arm at this model/config; CLM-002
  stays hypothesis, effect table recorded.
- Report arm inconclusive: given the weak P3 signal (mapping-specific margin
  ~+0.22 log-prob, D-016 report-gate), the jspace→report effect may fall
  below δ_report or straddle it across draws. That is a reportable
  outcome (the measure lacks the power to confirm D2 here), NOT engineered
  around — the transfer flag and per-draw values make it visible.
- Cross-draw transfer failing (median fires, draws disagree): the ablation
  effect does not transfer across independent draws; reported as such, D-
  direction not counted as clean.
- Any consumed instrument not gated at run time (fv-direction-ablation,
  jspace-ablation, report-score-prior-corrected@singular-plural): the run
  aborts at the require_controlled assertion before measuring.
- Post-hoc analyses of this run's records are labeled post-hoc forever.

## Estimator plan

- FVs: certified fv_todd@singular-plural, 3 draws (M2, full-trial
  n_trials_aie=200 ≥ converged_at=25), StabilityGatedFV, evidence run
  results/m2/20260718-114950-fv-stability-gate (commit 6a3a00b), backing
  cache/m2/draw{1,2,3}. Lens: 3 M1 draws — draw 0 from cache/m3 (M3-verified),
  draws 1–2 from cache/m4e1/lensdraw{1,2} (E1), each identity-checked against
  the committed M1 manifest per draw at run start.
- Every ablation effect is a 3-draw DrawSet (median/IQR the only summary):
  the fv arm's draws are the FV draws, the jspace arm's the lens draws. No
  single-draw number is reported. Within-draw N (exec 50, report 80) tightens
  each draw's mean; the paired design cancels context variance.

## Instruments

All consumed instruments are gated and asserted (require_controlled) before
any measurement, citing committed ControlRecords:
- fv-direction-ablation — M3 (results/m3/20260718-174954-instrument-gate),
  gated on singular-plural.
- jspace-ablation — M3 (same run), gated (anchor capital-recall).
- report-score-prior-corrected@singular-plural — the D-016 report-gate
  (results/m4/20260719-053911-e2-reportgate), gated under P3.
- execution scoring is deterministic (exact-match), not an instrument.

## Interventions and shams

- fv-direction ablation: project out the certified unit FV direction at the
  final position of each band layer; sham = one matched random unit direction,
  same layers/position (vendored make_hooks; sham random-direction seed varies
  per draw so the sham is itself a 3-draw DrawSet).
- jspace ablation: project out the span of the top-10 J-lens-readout atoms per
  band layer at the final position; sham = 10 random unit directions, same
  layers/position, sham seed per draw.
- Every ablated reading is reported next to its sham in the same effect row
  (report.md), per the sham LAW.

## Sample plan

- Task: singular-plural. Execution queries from the test split; report
  contexts and the neutral pool (mixed {capitalize, english-french}) from
  train. N_exec = 50, N_report = 80 per condition; neutral pool N = 80.
- Conditions: none (shared clean), fv/sham_fv × 3 FV draws, jspace/sham_jspace
  × 3 lens draws.
- Raw cells (≥ 20 records each): exec_none, report_none, exec_{fv,sham_fv}_draw{1,2,3},
  report_{fv,sham_fv}_draw{1,2,3}, exec_{jspace,sham_jspace}_lens{0,1,2},
  report_{jspace,sham_jspace}_lens{0,1,2}, neutral_P3.

## Resource estimate

~1,730 forward passes: clean (130) + neutral (80) + fv arm (3 draws × 2
conditions × 130) + jspace arm (3 × 2 × 130). fv/sham forwards ~0.15 s;
jspace forwards do a per-band-layer QR (~0.2–0.3 s). Lens loads are cache
hits; model load ~1 min. Projected ~15–25 min wall-clock, bounded 40 min.
Peak RSS ≤ ~4 GB, VRAM ~2/8.2 GB. Under the 12 h LAW with wide margin; > 10
min, so the run launches only after this file's commit and Ecaterina's
threshold ratification.

## Deviations

(none at commit time)
