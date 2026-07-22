# Preregistration — EXP-M5-1d: S1 ablation-potency probe (410M)

- experiment-id: EXP-M5-1d
- status: **RATIFIED** by Ecaterina 2026-07-22 with two adjustments folded in
  (bar #1 split into non-adjacent poles: H-POTENT ≥0.15, H-INERT ≤0.05, 0.05–0.15
  weak/ambiguous toward neither; bar #3 positive control locked to the identical
  statistic + ≥3-draw marginalization as the test). Committing this file is the
  prereg act; run on 410M. No certificate under either branch.
- AMENDMENT BUDGET: this **counts as ONE amendment cycle** (Ecaterina's ruling —
  not to be reclassified as "not a cycle"). **One cycle remains after this.**
- WHY approved (recorded verbatim in intent): a decodable-but-not-potent claim
  is UNPUBLISHABLE on a single potency probe (injection only). The ablation arm
  is REQUIRED to make it a real add/remove dissociation — NOT because it might
  rescue a certificate. Two orthogonal probes (add = injection, remove =
  ablation), both null, is a sufficient and publishable finding; a THIRD potency
  knob is explicitly banned as harvesting.
- claim: none (the second, orthogonal potency probe for S1; with EXP-M5-1b it
  decides the add/remove dissociation).
- models: EleutherAI/pythia-410m@9879c9b (the M5.1/1b substrate; the concept
  directions already exist there).
- config: configs/m5_1d_ablation_pythia410m.yaml (drafted with the orchestrator
  on approval).
- author + date: Claude (proposal), 2026-07-22, per Ecaterina's ratification-
  with-conditions.

## Hypothesis

The two potency probes ask orthogonal causal questions of the SAME EXP-M5-1b
concept direction d(c):
- ADD (EXP-M5-1b, done): does injecting d(c) steer the output toward c? → NO
  (0/8 potent; unmeasurable at 410M via residual addition).
- REMOVE (this probe): does projecting d(c) out of the residual DEGRADE the
  model's ability to produce c on the task?

Two branches, decided by this run (stopping rules fixed BELOW, before running):
- H-INERT: ablation is also null → S1 concept directions are decodable/draw-
  stable but BOTH injection- and ablation-inert. That IS the finding, published
  as-is and paired with S2. No third knob.
- H-POTENT: ablation degrades the task beyond sham → injection was the wrong
  potency probe; S1 is potent-and-decodable → the S1 certificate path opens
  (convergence from 1b + potency from here), pending Ecaterina's sign-off.

## Extractor / directions under the probe

The UNCHANGED EXP-M5-1b concept directions d(c): per-layer residual mean-
difference over certified capital-recall answer states, band [4,16], the same
3 draws (seeds 1/2/3; only the context-resampling stream varies). The 8-capital
roster (Paris, London, Rome, Berlin, Madrid, Vienna, Athens, Cairo).

## Ablation instrument (E2 machinery, identical statistic)

Reuse jvec.evals.exp3.ProjectOutHook (`h[:,-1] -= P h[:,-1]`, P onto span(dir))
at the final position of each band layer, and jtvec.e2_dissociation.effect_drawset
/ DissociationRule — the SAME statistic as EXP-M4-E2 (D-017):
- per concept c, per draw k: ablate d(c) at band-layer final positions; measure
  greedy top-1 accuracy on held-out capital-recall prompts whose answer is c
  (exact-match, D-012). effect(k) = clean_acc − ablated_acc (positive = ablation
  HURT). effect DrawSet over the 3 draws.
- sham twin per (concept, draw): a norm-matched random-direction project-out
  (exp3 sham_fv), identical layers/positions; sham DrawSet.
- "hurts" iff (effect.median − sham.median) ≥ delta AND cross-draw transfer
  (EVERY draw clears sham.median + delta) — the E2 transfer check.

## Decision rule — A2 bars STATED NOW (not after); split NON-ADJACENT branches

Per-concept, let g = sham-controlled accuracy drop = effect.median − sham.median
over the 3 draws. The poles are SPLIT and non-adjacent (Ecaterina 2026-07-22, to
protect the H-INERT headline):

- **ABLATION-POTENT** iff **g ≥ 0.15** (E2 execution delta / M3 fv-ablation bar,
  `min_exec_drop`) WITH cross-draw transfer (EVERY draw clears sham.median + 0.15).
- **ABLATION-INERT** iff **g ≤ 0.05** (at the E2 sham-noise floor ~0.02) — NOT
  merely "below 0.15". "Decodable but inert" is the strong headline claim and is
  defensible only if inert means AT-FLOOR.
- **WEAK/AMBIGUOUS** iff **0.05 < g < 0.15** — reported per concept as such,
  counted toward NEITHER pole.

Roster verdict (re-derived on the split thresholds):
- **H-POTENT** iff ≥ 6/8 concepts are ablation-potent (g ≥ 0.15, transfer).
- **H-INERT** iff ≥ 6/8 concepts are ablation-inert (g ≤ 0.05).
- else **MIXED** → reported per concept, no forced call.

### Branch stopping rules (fixed before running)

- (a) **H-INERT (ablation also null):** the finding is **"on 410M, S1 concept
  directions are draw-stable and decodable but both injection-inert (1b) and
  ablation-inert (1d) — add/remove-inert"**, published as-is and paired with S2.
  **NO third potency knob** is proposed to "confirm" — two orthogonal nulls
  suffice; a third is harvesting (Ecaterina's condition). This closes the S1
  potency question at 410M.
- (b) **H-POTENT (ablation degrades):** injection was the wrong probe; **S1 is
  potent-and-decodable**. The S1 certificate then rests on 1b convergence +
  this ablation potency; issuing it is a separate step gated on Ecaterina's
  sign-off (no certificate is written by this run).

## Instruments and controls

- MECHANISM positive control (injection), ON RECORD (Ecaterina's condition): the
  EXP-M5-1c null-check injected the **unembedding direction of "Paris"** through
  the residual-addition mechanism and moved **Δp(Paris) = +0.80** (band 0.005) —
  cited from results/m5/20260722-012727-m5-1c-nullcheck. This proves the ADD
  mechanism works, so 1b's injection-null is a real property, not a dead knob.
- ABLATION positive control (this run): project out the **unembedding direction
  of c** at the band-layer final positions must degrade c-accuracy by ≥ 0.30
  above sham — a direction known to carry the answer — computed with the IDENTICAL
  accuracy-drop statistic and the SAME ≥3-draw median/IQR marginalization as the
  A2 test bars (so the control and the test cannot diverge on method; Ecaterina
  2026-07-22). If it does NOT clear 0.30, band-layer ablation is too weak to
  matter and the result is INCONCLUSIVE (neither branch), NOT H-INERT.
  require_controlled() gates the verdict.
- Negative control / specificity: ablating d(c) on prompts whose answer is NOT c
  must not degrade those beyond sham (the direction is c-specific).
- Sham twin in every reported row (sham LAW).

## What counts as failure

- Ablation positive control fails (unembed-direction project-out does not degrade
  c) → band-layer ablation too weak → INCONCLUSIVE, investigated not interpreted.
- Control failure → instrument withdrawn (instruments LAW).
- Post-hoc analyses labeled post-hoc forever.

## Estimator plan

Deterministic per draw given the concept direction; forward-only (accuracy under
the project-out hook). 3 draws (seeds 1/2/3); the concept-direction draw is the
only nuisance axis; median/IQR over draws (DrawSet, E2 effect_drawset). No new
RNG beyond the sham seeds.

## Sample plan

- Held-out capital-recall prompts answering c: N ≥ [proposed 30] per concept,
  distinct from the extraction contexts; N recorded.
- Conditions per (concept, draw): clean / ablate-d(c) / sham / ablate-unembed(c)
  (positive) / ablate-d(c)-on-other-answers (specificity). Raw per-item top-1
  retained; ≥ 20 records per headline cell.

## Resource estimate (Mac tier, 410M, MPS fp32)

Forward-only accuracy evals under a project-out hook; no fit, no Jacobian. 8
concepts × 3 draws × ~5 conditions × N≈30 ≈ 3.6k short generations → projected
~20–40 min wall, peak RSS ~2 GB — well under the 12 h LAW, Mac-eligible. Detached
+ Monitor.

## Deviations

(none yet)

## Ratification

ratified: EXP-M5-1d 2026-07-22 — Ecaterina, with two adjustments (bar #1 split
non-adjacent poles H-POTENT ≥0.15 / H-INERT ≤0.05, 0.05–0.15 weak/ambiguous
neither; bar #3 positive control uses the identical accuracy-drop statistic +
≥3-draw median/IQR marginalization as the test). Roster 6/8 unchanged; both
branches + stopping rules + no-third-knob-under-H-INERT stand. Mechanism positive
control = the null-check +0.80 "Paris" injection, on record. Counts as ONE
amendment cycle; ONE remains. This run writes NO certificate under either branch
(H-POTENT opens the certificate path gated on Ecaterina's separate sign-off).
