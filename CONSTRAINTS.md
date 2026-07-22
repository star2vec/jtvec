# CONSTRAINTS.md — v2 axioms and epistemic ledger

Every statement in this file carries one of four tags. Nothing untagged may be
added. Nothing may be promoted a tier without the evidence line filled in.

- **LAW** — process rule. Not empirical. Violating it invalidates the run.
- **VERIFIED** — supported by v1 evidence that survived its own controls.
  Each entry cites what the evidence actually was. Scope is stated narrowly;
  do not generalize beyond it.
- **PROVISIONAL** — adopted as default because it worked in v1, but never
  independently stress-tested. Must be re-verified in v2 before any claim
  depends on it.
- **HYPOTHESIS** — v1 produced suggestive results that are contaminated,
  single-draw, or unconfirmed. These are the things v2 exists to test. They
  must NEVER be stated as findings in code comments, logs, drafts, or reports
  until v2 confirms them.

---

## Laws (process)

- LAW: No number derived from a stochastic estimator is reported from a
  single draw. Minimum 3 independent draws; report median and IQR. This
  applies to FV extraction, lens fitting, and anything else with an RNG.
- LAW: Every estimator passes a stability gate before its outputs are used:
  demonstrate convergence of the estimate under re-sampling at the chosen
  sample size, on the target model, before the first scientific number.
- LAW: Every instrument passes a positive control AND a negative control
  before its readings count. An instrument that cannot separate known-signal
  from known-noise is withdrawn (precedent: v1 Exp-2).
- LAW: Every intervention runs with an auto-generated sham twin (matched
  norm, count, layers, positions). No intervention effect is quoted without
  its sham in the same table.
- LAW: Preregistration file per experiment, committed BEFORE the first run,
  containing: hypothesis, decision rule, and what result would count as
  failure. Post-hoc analyses are labeled post-hoc forever.
- LAW: Human verification gate between "results exist" and "results are
  claimed": the AI lays out the raw completions (≥20 per headline cell) and
  its re-derivation of the headline numbers; Ecaterina reviews and confirms
  (she need not re-derive by hand), and the AI may transcribe the verify line
  on her explicit confirmation. Logged in LABNOTES with date. No claim
  advances without this entry. (Amended 2026-07-21 by Ecaterina via session
  instruction, from the original "Ecaterina re-derives by hand": the human
  confirmation remains the gate; only the by-hand-derivation burden is lifted.
  Sign-off lines stay Ecaterina's own to type.)
- LAW: One commit per experiment. Raw model outputs are retained on disk for
  every reported number. Configs are copied into every results directory.
- LAW: Language discipline in all reports and notes: state observations with
  their scope ("on Pythia-410M, config X, N=60"), never as general facts.
  The words "shows", "proves", "demonstrates" require a VERIFIED-tier basis.
- LAW: A claims ledger (CLAIMS.md) tracks every scientific claim with status
  ∈ {hypothesis, preliminary, verified, withdrawn} and the evidence commit.
  The paper may only contain claims at "verified".
- LAW: Instrument-amendment discipline (added 2026-07-21 by Ecaterina, session
  ruling, after three consecutive gate-fail → diagnose-miscalibration → amend
  cycles: M1 lens gate/D-027, 1.4B Q2/0c H-CONFOUND, M5.1 concept/D-033 — each
  individually defensible, the pattern a live risk that amendment is harvesting
  researcher degrees of freedom rather than fixing instruments). No FOURTH
  amendment to any instrument is ratifiable without a NULL-CHECK on record
  first: the recalibrated instrument, run under its own statistic against a
  known-null condition (existing sham twins / scrambled-label controls, on a
  substrate+task where nothing is to be found, ≥ 3 draws, median/IQR), must
  report NULL on that null, per a pass condition fixed before the run. An
  instrument that reports signal on the null is withdrawn and the amendment it
  supports is not ratifiable pending re-spec.
- LAW: Amendment budget (standing, 2026-07-21, Ecaterina). Two instrument-
  amendment cycles remain across ALL instruments. When exhausted, M6 runs on
  whatever instruments are certified and every unmeasurable species is REPORTED
  as unmeasurable — a taxonomy with holes is preferred to recalibrating into
  one. Escalating the substrate to make a gate pass counts as an amendment
  cycle (the S4/2.8B precedent): the prior question is always specification,
  not scale.

## Verified (v1 evidence that survived controls — narrow scope)

- VERIFIED: The J-lens pipeline reproduces on Pythia-410M and passes a
  9-check sanity gate (probing HMR 2.5 vs logit 61.5 on capital-recall in
  the L4–L16 band; 10-seed random controls; swap Δp +0.60 vs +0.009 sham,
  matching the token-patching ceiling). Evidence: v1
  `results/phase1_report/20260714-051555/`, reproduced independently by the
  endpoint mini-gate during the sweep. Scope: Pythia-410M, skip4_n10.
- VERIFIED: On GPT-2-small the same pipeline fails probing criteria at every
  layer while the causal swap still works (Δp +0.38 vs +0.003). Scope: the
  observation. The tied-embedding *explanation* is PROVISIONAL (one tied vs
  one untied model = n=1 vs n=1; mechanism inferred, not isolated).
- VERIFIED: Causal swaps require truncated pseudoinverse (rcond≈0.05),
  source-token-position edits, and norm preservation; full pinv destroys
  computation on both v1 models. Evidence: v1 Phase-1, both substrates.
- VERIFIED: Todd-style FV extraction at 25 AIE trials is not draw-stable on
  Pythia-410M: re-extraction on identical weights gave cosine 0.43–0.61,
  top-head overlap 3–6/10, capitalize induction +38.8% → +1.8% across draws.
  Evidence: v1 endpoint consistency check. Consequence: every v1 number
  downstream of a single FV draw is contaminated (see HYPOTHESIS section).
- VERIFIED (negative): k=25 gradient-pursuit J-space-fraction cannot
  separate demonstrably lens-readable residuals from arbitrary directions on
  Pythia-410M (positive control failed). This instrument is banned in v2
  unless rebuilt and re-controlled.
- VERIFIED: `add_function_vector` from the Todd repo silently no-ops under
  transformers 5.x (tuple-assuming hook). Any v2 use requires the patched
  hook plus a unit test proving the vector actually lands.

## Provisional (defaults; re-verify per model before claims depend on them)

- PROVISIONAL: untied-embedding substrates only (Pythia family primary).
  Design choice motivated by the GPT-2/Pythia contrast; the mechanism is
  untested. Re-verify the contrast on a second tied/untied pair if the paper
  leans on the explanation.
- PROVISIONAL: lens config skip_first=4, n=10 calibration prompts; band
  layers ≈ L4–L16 on 410M. Cheap to re-derive per model; do so.
- PROVISIONAL: task inclusion by zero-shot FV induction strength rather than
  ICL accuracy. Reasonable ruling, never stress-tested.
- PROVISIONAL: forced-choice report probe (P3 phrasing, scored over task-label
  tokens, shuffled-context prior baseline). Known risk: single phrasing.
  v2 must pre-specify ≥3 phrasings and report all.

## Hypotheses (v2 exists to test these; never state as findings)

- HYPOTHESIS: FVs carry a J-lens-readable task-label component invisible to
  the logit lens. (v1 Exp-1: single FV draw per task → contaminated.)
  (superseded-by-taxonomy 2026-07-19: E1 counter-evidence — NOT-DECODABLE on
  3/3 M2-certified tasks, `results/m4/20260719-021823-e1-decodability`;
  recast as the S2×A1 matrix cell in TAXONOMY_DESIGN.md. Retained, never
  deleted.)
- HYPOTHESIS: task execution and task verbalization causally dissociate
  (double dissociation on singular-plural; one-way on landmark-country).
  (v1 Exp-3: single draw, single model, N=60, human verification pass never
  completed.) Highest-priority confirmation target. Include cross-draw
  ablation transfer as the design.
  (taxonomy recast 2026-07-19: generalized by axes A2/A4 and H1/H3 in
  TAXONOMY_DESIGN.md; v2 evidence so far = E2 ONE-WAY on singular-plural
  (CLM-002 stays hypothesis; CLM-003 preliminary). Still the umbrella
  HYPOTHESIS; not retired.)
- HYPOTHESIS: label-decodability and output-vocabulary decodability are
  separable FV properties (english-spanish vs english-french contrast).
  (superseded-by-taxonomy 2026-07-19: english-spanish remains uncertified in
  v2; the contrast folds into per-species A1 measurement. Retained.)
- HYPOTHESIS: Nadaf's steerable-but-not-logit-decodable pattern replicates.
  (7/8 in v1, but on contaminated single-draw FVs.)
  (superseded-by-taxonomy 2026-07-19: the pattern is now the predicted S2
  row — A1 dark (E1) with A2/A3 potent (E2/E3); measured per-species going
  forward. Retained.)
- HYPOTHESIS: ICL execution matures early in training while portable FVs and
  their decodability emerge late. (v1 sweep: confounded by draw instability
  AND fixed head-selection; both must be resolved. Requires per-checkpoint
  AIE recomputation and stability-gated FVs at every checkpoint.)
- HYPOTHESIS: J-space ablation can *raise* report accuracy on some tasks
  (v1 anomaly, quarantined; unexplained).
  (absorbed-into-H3 2026-07-19: the rise is now a predicted dark-species
  signature per TAXONOMY_DESIGN.md H3; still HYPOTHESIS-tier.)
- HYPOTHESIS: models can be made to report a task they cannot perform
  (confabulation via label injection + FV ablation). Never run in v1.

### Taxonomy phase (H1–H5, added 2026-07-19; construct + matrix in TAXONOMY_DESIGN.md)

- HYPOTHESIS (H1, dichotomy, confirmatory): S1 and S2 occupy opposite poles
  of the matrix at preregistered bars. Anchor result; both poles already
  have supporting certified evidence (M1 +0.60 lens-coord swaps; E1/E3
  negatives).
  (2026-07-22 REFRAMED, Ecaterina ruling (a) after EXP-M5-6: the original
  (decodability × potency) 2×2 is RETIRED as the anchor — it compared S1 and S2
  on DIFFERENT senses of "decodable" (S1 = EXP-M5-1b convergence; S2 =
  decode_vector), and under the SAME instrument (decode_vector) BOTH S1 and S2
  are lens-DARK (jlens label-rank 192 / 436). The instrument-CONSISTENT anchor
  is DRAW-STABILITY × POTENCY: S1 = (draw-stable [EXP-M5-1b, all 8 cross 0.95],
  injection-inert [1b 0/8; EXP-M5-1c null-check +0.80 confirms the knob; 1d
  corroborates]) vs S2 = (draw-UNSTABLE [VERIFIED M2, cosine 0.43-0.61], potent
  [E2/E3, CLM-003/004 VERIFIED]). Additional axes recorded under the same
  decode_vector (EXP-M5-6): lens-readout — S1 dark, S2 dark, S5 logit-trivial;
  logit-privilege (A1b) — empty (no representation is J-lens-privileged);
  output-alignment — S5. CENTRAL FINDING: "A1 decodability" conflated THREE
  measurements (draw-stability, lens-readout, logit-privilege) — CLM-006, the
  paper's spine. Still HYPOTHESIS-tier (S1 legs diagnostic; S2 potency verified).
  NO instrument hunted to restore the old 2×2 (harvesting; budget 0). CLM-005/006.)
- HYPOTHESIS (H2, axis coupling): A1 and A3 agree per species. A
  dissociation (decodable-but-basis-agnostic or dark-but-basis-mediated) in
  any species is a headline refinement: verbalizable != workspace-functional.
- HYPOTHESIS (H3, report specificity): A4 tracks residency; dark-species
  ablation spares or RAISES report readouts (the E2 fv-report rise becomes
  a predicted signature, not an anomaly).
- HYPOTHESIS (H4, relational split): the LRE operator is dark while its
  outputs are resident; workspace ablation impairs relation NAMING but not
  relation APPLICATION.
- HYPOTHESIS (H5, dark boundary): binding vectors are potent and dark on
  all other axes.

## Known-unknowns (decide empirically in v2, do not assume)

- The AIE trial count at which FV extraction converges on 410M (and whether
  it converges at all). The v2 stability gate produces this number first.
- Whether Todd-FVs and Hendel-FVs identify the same object at this scale.
- Whether any v1 "unreportable" task becomes reportable with better probes
  or larger models (capability vs representation-absence confound).
