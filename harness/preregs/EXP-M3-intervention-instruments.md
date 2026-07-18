# Preregistration — EXP-M3: intervention-instrument gate

- experiment-id: EXP-M3
- claim: none (instrument gate; no CLAIMS.md entry moves — it produces the
  ControlRecord pairs the M4 experiment preregs must cite)
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m3_pythia410m.yaml
- author + date: Claude (proposal), 2026-07-18; scope ruled by Ecaterina
  2026-07-18 (D-011); thresholds and launch ratified by Ecaterina
  2026-07-18 (session Q&A). Committing this file is the prereg act.

## Hypothesis

No scientific hypothesis is tested. This run applies the instruments LAW to
the intervention instruments M4 consumes: each must pass a positive AND a
negative control before its readings count. Instruments under the gate:
fv-direction-ablation, jspace-ablation, report-probe-forced-choice, fv-swap
(FV injection was gated at M2). E1–E4 stay in M4.

## Decision rule

Prerequisite, lens re-materialization (the M1 lens cache is machine-local):
vendored 01_fit_lens.py refits skip4_n10 on this machine; identity must
equal M1 draw 0's committed manifest on the vendored IDENTITY_KEYS
(calibration sha256 exact, jlens commit, fit hyperparameters, seed);
functional spot-check on capital-recall must give band-min jlens HMR
≤ 1.5× the committed M1 draw-0 value AND logit-lens HMR at that layer
≥ 5× the refit jlens HMR. Identity or spot-check failure aborts the gate.

Controls (context rng 9090; sham seeds cfg.seed=0; every deviation bound is
quantization-aware per D-010: max(base, 1/N)):

- fv-direction-ablation (certified FVs, band layers 4–16, final position),
  on capitalize AND singular-plural, N=30 exec items each:
  positive = exec(none) − exec(fv) ≥ 0.15 on both tasks;
  negative = |exec(none) − exec(sham_fv)| ≤ max(0.05, 1/30) on both.
- jspace-ablation on swap-capitals (M1-anchored: the band's lens-readability
  for capital content is M1-VERIFIED; no E-experiment claim is presupposed),
  all task items (N=16):
  positive = exec(none) − exec(jspace) ≥ 0.15;
  negative = |exec(none) − exec(sham_jspace)| ≤ max(0.05, 1/16).
- report-probe-forced-choice (8-way candidate set, prior 1/8), per certified
  task, all three phrasings P1–P3 reported:
  positive = explicit-rule detection ≥ 0.8 under the best phrasing per task
  (context: rule sentence naming the label + label-shuffled pairs — a
  detection-ceiling control, deliberately not an ICL-content claim);
  negative = shuffled-context accuracy − prior ≤ max(0.15, 1/36), per task
  (N=36: 12 contexts × 3 phrasings).
- fv-swap on capitalize→singular-plural (both certified; D-011), N=30 shared
  queries with distinct targets, conditions none/lens_swap/direct_swap/
  random_target (rcond 0.05, norm-preserving, band layers, final position):
  positive = max(direct, lens) b-rate − none b-rate ≥ 0.2;
  negative = |random_target b-rate − none b-rate| ≤ max(0.05, 1/30).

An instrument is GATED iff its positive and negative controls both pass.
M3 PASS = all four instruments gated. ControlRecord pairs (run, passed,
date) are the deliverable, in controls.json.

[Thresholds ratified by Ecaterina, 2026-07-18; the constants in
scripts/m3_gate.py match.]

## What counts as failure

- Lens identity mismatch or spot-check failure: no instrument is gated;
  investigate the refit (raw replay first) before any re-run.
- Any instrument failing either control: that instrument is not gated and
  every M4 design depending on it is blocked until it is redesigned and
  re-controlled under a new prereg (instruments LAW; precedent: the
  withdrawn v1 J-space-fraction reader stays banned).
- Post-hoc analyses of this run's records are labeled post-hoc forever.

## Estimator plan

No stochastic estimator is drawn here. The two estimators involved carry
their own gates: the lens (M1, 3-draw stability) and the FVs (M2
certificates; the concrete artifacts are draw 1's full-trial FVs,
n_trials_aie=200 ≥ converged_at=25, loaded through the manifest-checked
vendored loader as StabilityGatedFV). Control accuracies are deterministic
given the preregistered seeds; per-item raw records are retained for every
number. No DrawSets are reported because nothing here is re-drawn; any
future use of these instruments on re-drawn estimators inherits the M2
draw discipline.

## Instruments

This experiment IS the instrument-control run. Each instrument's positive
and negative arms are specified in the Decision rule; ControlRecord pairs
point at this run's results directory.

## Interventions and shams

- fv-direction ablation: project out the certified unit FV direction at the
  final position of each band layer; sham = matched count (1) of random
  unit directions, same layers/position (vendored make_hooks, seed 0).
- jspace ablation: project out the span of the top-10 J-lens-readout atoms
  per band layer at the final position; sham = 10 random unit directions,
  same layers/position (vendored make_hooks, seed 0).
- fv-swap: move the FV_A component onto FV_B at the final position of each
  band layer, lens coordinates (truncated pinv, rcond 0.05) and direct
  residual coordinates; control arm = norm-matched random target
  (vendored make_swap_hooks, seed 0). Norm-preserving throughout.
- Every ablated/swapped reading is reported next to its sham/control arm in
  the same line (report.md), per the sham LAW.

## Sample plan

- Tasks: certified set (capitalize, singular-plural, english-french) for
  the report probe; capitalize + singular-plural for fv-ablation;
  swap-capitals (16 items) for jspace-ablation; capitalize→singular-plural
  shared queries (N=30) for the swap.
- Raw per-item cells: lens-spotcheck_capital-recall,
  fvablate_{task}_{cond}, jspace_swap-capitals_{cond},
  report_{task}_{P*}_explicit-rule, report_{task}_shuffled,
  swap_capitalize-to-singular-plural_{cond}.

## Resource estimate

From measured M2-probe rates on this machine (D-009 GPU) and M1's committed
fit time:

- Lens fit: 919 s on the M1 MacBook (committed manifest); expected 2–8 min
  on this GPU, bounded by ~15 min. One-time; cached under cache/m3.
- Controls: ~700 hooked/clean forwards total (ablation 180, jspace 48,
  report 216, swap 120, spot-check 16) at 0.1–0.5 s/forward (per-layer CPU
  hook round-trips dominate) ≈ 2–6 min. Model load ~1 min.
- Total: ~10–25 min wall-clock; peak RSS ≤ ~4 GB, VRAM ~2/8.2 GB. Under
  every resource LAW threshold; run launches only after the prereg commit
  and Ecaterina's go (>10 min rule).

## Deviations

- D-012 (Ecaterina, 2026-07-18; after run 2,
  results/m3/20260718-141523-instrument-gate). Run 2 gated only
  fv-direction-ablation; the other three failed for distinct reasons found
  by raw-output replay (LABNOTES). Four ruled changes, all applied in one
  commit with tests:
  1. Execution scoring: exact-match, case-sensitive first-token scoring
     (jtvec.m3_instruments.answer_first_tokens) replaces the vendored
     surface_token_ids case+space relaxation for the three execution
     controls (fv-ablation, jspace, swap). The relaxation had (a) let a
     jspace-broken "Te" still count as a hit for " Tehran" and (b) made an
     uppercase capitalize output (" K") collide with a plural target's
     capitalized variant (" Kettles") in the cross-task swap. The vendored
     surface scoring is retained for the M1 lens spot-check only (it mirrors
     the M1 probing protocol there).
  2. jspace-ablation anchor task: capital-recall (36 items, the M1-VERIFIED
     probing task) replaces swap-capitals (16 items), for finer-than-1/16
     granularity and a cleaner anchor. Positive/negative criteria unchanged.
  3. report-probe negative control: the null is now a random-word-output
     context (jtvec.m3_instruments.random_word_null_context; outputs drawn
     from the union of the OTHER tasks' outputs) instead of the v1
     shuffled-context baseline, which did not null morphological-output
     tasks (singular-plural null was 1.0). Threshold unchanged
     (null_acc − prior ≤ max(0.15, 1/N)).
  4. fv-swap negative control: one-sided — random_target must not ELEVATE
     the task-B rate over clean (b_rate[random] − b_rate[none] ≤
     max(0.05, 1/N)); a random swap that destroys computation (B → 0,
     observed) is expected per CONSTRAINTS and no longer counts as a
     failure. The two-sided |random − none| criterion is withdrawn.
  scripts/m3_gate.py constants and tests/test_m3_instruments.py updated in
  the same commit; re-run is evals-only on the cached lens. No change to
  the M1/M2 gate outputs this run consumes.
