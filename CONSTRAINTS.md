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
  claimed": Ecaterina reads raw completions (≥20 per headline cell) and
  re-derives headline numbers by hand. Logged in LABNOTES with date. No
  claim advances without this entry.
- LAW: One commit per experiment. Raw model outputs are retained on disk for
  every reported number. Configs are copied into every results directory.
- LAW: Language discipline in all reports and notes: state observations with
  their scope ("on Pythia-410M, config X, N=60"), never as general facts.
  The words "shows", "proves", "demonstrates" require a VERIFIED-tier basis.
- LAW: A claims ledger (CLAIMS.md) tracks every scientific claim with status
  ∈ {hypothesis, preliminary, verified, withdrawn} and the evidence commit.
  The paper may only contain claims at "verified".

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
- HYPOTHESIS: task execution and task verbalization causally dissociate
  (double dissociation on singular-plural; one-way on landmark-country).
  (v1 Exp-3: single draw, single model, N=60, human verification pass never
  completed.) Highest-priority confirmation target. Include cross-draw
  ablation transfer as the design.
- HYPOTHESIS: label-decodability and output-vocabulary decodability are
  separable FV properties (english-spanish vs english-french contrast).
- HYPOTHESIS: Nadaf's steerable-but-not-logit-decodable pattern replicates.
  (7/8 in v1, but on contaminated single-draw FVs.)
- HYPOTHESIS: ICL execution matures early in training while portable FVs and
  their decodability emerge late. (v1 sweep: confounded by draw instability
  AND fixed head-selection; both must be resolved. Requires per-checkpoint
  AIE recomputation and stability-gated FVs at every checkpoint.)
- HYPOTHESIS: J-space ablation can *raise* report accuracy on some tasks
  (v1 anomaly, quarantined; unexplained).
- HYPOTHESIS: models can be made to report a task they cannot perform
  (confabulation via label injection + FV ablation). Never run in v1.

## Known-unknowns (decide empirically in v2, do not assume)

- The AIE trial count at which FV extraction converges on 410M (and whether
  it converges at all). The v2 stability gate produces this number first.
- Whether Todd-FVs and Hendel-FVs identify the same object at this scale.
- Whether any v1 "unreportable" task becomes reportable with better probes
  or larger models (capability vs representation-absence confound).
