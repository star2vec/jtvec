# Preregistration — EXP-M2: FV extraction-stability gate (convergence ladder)

- experiment-id: EXP-M2
- claim: none (measurement gate; no CLAIMS.md entry moves on this run — it
  produces the gate certificates every later FV-dependent claim rests on)
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m2_pythia410m.yaml
- author + date: Claude (proposal), 2026-07-18; thresholds, compute
  placement, and the singular-plural handling ruled by Ecaterina 2026-07-18
  (D-009, session Q&A). Committing this file is the prereg act.

## Hypothesis

No scientific hypothesis is tested. This run applies the stability-gate LAW
to the Todd-FV estimator and decides the CONSTRAINTS known-unknown: the AIE
trial count at which FV extraction converges on Pythia-410M, and whether it
converges at all. v1 evidence (VERIFIED tier): at 25 AIE trials,
re-extraction on identical weights gave cosine 0.43–0.61, top-head overlap
3–6/10, capitalize induction +38.8% → +1.8% across draws.

## Decision rule

Per task, per rung T ∈ {25, 50, 100, 200}, over 3 draws:

- min pairwise cosine of fv_todd@T (3 pairs), and
- IQR over draws of the induction gain (0-shot top-1 with the FV at layer 8
  minus 0-shot top-1), on fixed eval contexts.

A rung passes iff min pairwise cosine ≥ 0.95 AND gain IQR ≤ 0.05.
converged_at(task) = the smallest T that passes with every larger rung also
passing; a pass at T=200 alone is NOT convergence (no stability witness).
M2 PASS = all 3 tasks have a converged_at ≤ 200. Certificates
(estimator "fv_todd@<task>", converged_at, n_draws=3, evidence run) are
issued per converged task only.

Top-head overlap and Hendel-vector cosines are recorded as descriptive
diagnostics with no decision weight.

[Thresholds 0.95 / 0.05 ratified by Ecaterina, 2026-07-18 (D-009); the rule
constants in scripts/m2_gate.py match.]

## What counts as failure

- Any task with no converged_at (including a max-rung-only pass): documented
  non-convergence for that task, no certificate, and nothing FV-dependent
  runs for that task until a follow-up ladder (≥ 400 trials, A100-scale) is
  ruled on.
- Instrument-control failure (below) voids all verdicts: the readout is
  withdrawn per the instruments LAW and M2 reports an instrument failure,
  not a convergence result.
- Post-hoc analyses of this run's stored tensors are labeled post-hoc
  forever.

## Estimator plan

- Estimator under the gate: fv_todd (Todd top-10-head FV; n_trials_mean=100
  fixed, AIE trial count laddered). 3 independent draws, seeds
  set_seed(cfg.seed·1000 + k) = {1, 2, 3}; only the extraction RNG stream
  varies — model weights, dataset splits (load_dataset seed 0), and eval
  contexts are held fixed. The per-draw filter set is computed inside the
  draw's stream (design-ref behavior): it is part of extraction
  stochasticity, not of the evaluation.
- Rungs derive from the stored per-trial AIE tensor by prefix slicing; the
  RNG-prefix property and the rung recomputation are pinned by unit tests
  (tests/test_m2_stability.py).
- fv_hendel: descriptive 3-draw cosines only (fixed n_trials_mean=100); M2
  issues no Hendel certificate. A later experiment needing a certified
  Hendel estimator requires a follow-up gate.
- Median/IQR are the only cross-draw summaries (DrawSet).

## Instruments

- fv-induction-readout (n_shot_eval top-1 on the test split):
  positive control = 10-shot ICL top-1 exceeds 0-shot top-1 by ≥ 0.10 on
  every task (known signal, evaluated in-run); negative control =
  |median sham gain| ≤ 0.02 at every rung on every task (known noise,
  in-run). require_controlled() gates the verdict computation.

## Interventions and shams

- Intervention: inject fv_todd@T at layer 8 (config edit_layer), last-token
  position, 1 direction, natural FV norm.
- Sham twin per (task, draw, rung): norm-matched random direction
  (torch.Generator seed 9000 + 10·draw_k + rung_index) injected at the same
  layer/position. Every reported gain appears with its sham in the same
  line/row (scoped_intervention, InterventionResult).

## Sample plan

- Tasks: capitalize, singular-plural, english-french (the D-007 ladder set;
  inclusion fixed by D-007, not re-gated here).
- Known small-task caveat, ruled kept-and-flagged (D-009): singular-plural
  has N_test=43 (induction-gain granularity 1/43 ≈ 2.3 pp against the 5 pp
  IQR criterion) and 17 correct-valid items feeding trial sampling (the
  min_correct_valid=50 floor of scripts/05 was never applied to the D-007
  set). The report carries this caveat next to the task's numbers.
- Rungs: 25/50/100/200 AIE trials; extraction once per (task, draw) at 200.
- Evals: full test split per (task, draw, rung, arm); N recorded per task in
  the report. Raw per-item rank records per cell: {task}_zeroshot,
  {task}_icl10shot, {task}_rung{T}_induction, {task}_rung{T}_sham.

## Resource estimate

Probe (scripts/m2_probe.py, 2026-07-18, 81 s GPU + 222 s CPU): AIE
8.77/8.97/9.09 s/trial (capitalize / singular-plural / english-french) on
the win32 laptop GPU; 69.8/72.2 s/trial CPU reference.

- Machine (D-009 ruling): this win32 laptop's RTX 2000 Ada 8 GB (D-008
  stack, torch 2.13.0+cu130, fp32). Alternatives priced and declined:
  CPU ~42 h (over the LAW), MacBook 26–28 h (over the LAW), A100 ~3–4 h
  (kept as the escalation path if the ladder must extend to ≥400 trials).
- Wall-clock: ≈5.5 h (extraction 3×5,410 s + eval grid ≈1,122 s, ×1.15
  slack; eval grid corrected for per-task test sizes 170/43/987). Planning
  window 6–7 h for laptop thermal derating (61 °C @93% util at probe).
- Peak memory: 3.35 GB RSS at probe scale, ~4–5 GB expected at
  n_trials_mean=100; VRAM 1.8/8.2 GB. Under the 12 h laptop LAW; no waiver
  needed.
- Launch mode: detached with a log monitor (M1 incident precedent:
  harness-tracked background tasks are killed at ~1 h). Extraction caches
  per (task, draw) make the run resumable at stage granularity.

## Deviations

(none at commit time)
