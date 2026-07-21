# Preregistration — EXP-M5-2: S3 relational-operator stability gate (1.4B)

- experiment-id: EXP-M5-2
- status: **RATIFIED** by Ecaterina 2026-07-22 (session instruction "ratify
  m5.2"). Committing this file is the prereg act; the RTX may build + run against
  it. Thresholds ratified as drafted. Independent of the EXP-M5-1c null-check
  (functional gate; see the independence section) and not counted against the
  amendment budget.
- claim: none (S3 species certificate gate; produces the S3 stability
  certificate + ControlRecords the S3 rows of the M6 axis battery rest on).
- models: EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79 — the
  ONLY S3-admitted substrate (EXP-M5-0 qualification: LRE 8/12 at 1.4B; 410M
  4/12, NOT admitted). GPU-tier (RTX cuda).
- config: configs/m5_2_operator_pythia1p4b_cuda.yaml (drafted by the RTX with the
  orchestrator; standard schema, revision pinned).
- author + date: Claude (proposal), 2026-07-22.

## Independence from the instruments under null-check (stated up front)

The S3 gate stands ENTIRELY on functional / behavioural quantities — operator
faithfulness (top-1 output accuracy on held-out subjects), cross-draw output-
token agreement, and output-state cosine. It does NOT use the amended-Q5
max-contrast statistic or the D-033 concept-ladder readout (the instruments
under EXP-M5-1c). So EXP-M5-2 is INDEPENDENT of the null-check and does not wait
on it, and — being a new species extractor rather than a recalibration of an
existing one — it does NOT consume the CONSTRAINTS amendment budget. (The
J-lens/A1 decodability of the operator outputs is an M6 axis measurement, out of
scope here; only functional convergence gates the certificate.)

## Hypothesis

No hypothesis about the models is tested. This applies the stability-gate LAW to
the S3 relational-operator extractor (M5_SPEC §M5.2), producing the entry
certificate TAXONOMY_DESIGN demands per species before any A1–A5 axis (M6). The
S3 rows of H4 (the relational split) are unmeasurable without this certificate.

## Extractor under the gate

s3_operator@<relation>: the Hernandez/Hendel linear relational operator
(W, b) — a per-relation affine map on the subject representation — estimated by
the vendored JacobianIclMeanEstimator (third_party/relations/src/operators.py,
pinned submodule @1b9ec3c, D-022), at a k-shot ICL context, wrapped for the
pinned 1.4B with a LANDING TEST (the operator applied to held-out subject states
reproduces the estimator's reported faithfulness; the wrapper is not trusted
without it). Vendored code is called as a library, never modified (VENDORING).

Relations under the gate = the 8 that cleared the M5.0 S3 bar at 1.4B (faithful
≥ 0.60): factual/country_capital_city, factual/food_from_country,
factual/product_by_company, linguistic/adj_antonym, linguistic/verb_past_tense,
linguistic/word_first_letter, commonsense/object_superclass,
commonsense/work_location.

Draws: ≥ 3, seeds {1,2,3}; only the ICL-context + subject-sampling stream varies
(model weights, relation data, probe set held fixed — the M2 pattern).

## Decision rule (bars [proposed] until ratified)

Per relation, over the 3 draws, on a FIXED held-out probe set of subjects:

- Functional agreement (the certified quantity, M5_SPEC §M5.2):
  - top-1 output-token agreement across draws ≥ [proposed 0.90] (the 3 operators
    map the same probe subject to the same top-1 object), AND
  - output-state cosine across draws ≥ [proposed 0.95] (the operator-output
    residual directions agree; these feed the S3-output row of M6).
- Raw parameter cosine of W across draws is reported DESCRIPTIVELY only (W is
  high-dim; parameter-space agreement is not the certified quantity — M5_SPEC).

converged / certified per relation iff BOTH functional criteria hold across all
3 draws AND both controls pass (below). The S3 species certificate on 1.4B is
issued for the set of relations meeting the bar; the certificate records which
relations certified (≥ [proposed 6 of 8] to call S3 certified as a species,
mirroring the LRE ≥8-of-12 admission logic scaled to this 8-relation set).

## Instruments and controls (per relation)

- Positive control: operator faithfulness on HELD-OUT subjects ≥ the M5.0 bar
  (0.60) — the operator actually predicts the relation on unseen subjects, not
  just its estimation set. require_controlled() gates the verdict.
- Negative control: a SHUFFLED-RELATION operator — estimated on mismatched
  subject→object pairs (labels permuted within the relation) — MUST FAIL
  faithfulness (≤ [proposed 0.10] on held-out subjects). If the shuffled
  operator is faithful, the estimator is fitting something other than the
  relation and the instrument is withdrawn.
- Sham twin (for the LATER S3 interventions, recorded now): a norm-matched
  random affine perturbation (δW, δb) matched to (W, b) norms; every future
  intervention effect appears with this sham.

## What counts as failure

- A relation whose operators do not meet the functional bar: documented
  non-convergence for that relation; if the whole-set rule then fails, no S3
  certificate on 1.4B and the S3 pole carries the hole explicitly (escalation is
  a flagged ruling, not an automatic re-run).
- Control failure (either) → instrument withdrawn, an instrument-failure report
  instead of a certificate (instruments LAW).
- Landing-test failure → the wrapper is wrong; fixed before any gate number.
- Post-hoc analyses of stored tensors labeled post-hoc forever.

## Estimator plan

The estimator is the vendored JacobianIclMeanEstimator producing a
LinearRelationOperator per relation per draw; the gate certifies its cross-draw
FUNCTIONAL convergence (output agreement + output-state cosine) — median/IQR over
the 3 draws are the only cross-draw summaries (DrawSet). Deterministic given a
draw's ICL/subject sample. The Jacobian is taken at the pinned 1.4B on cuda
(fp32; fp16 acceptable if VRAM-bound — cuda fp16 is fine — but fp32 keeps the
operator numerics matched to the M5.0 faithfulness numbers).

## Sample plan

- Held-out probe set: subjects NOT used in any draw's estimation, per relation;
  N_probe recorded per relation (target ≥ 30 where the relation's subject pool
  allows; smaller pools reported with achieved N).
- Estimation ICL context: k-shot per the estimator default (recorded); the
  subject/context stream is the only thing the seed varies.
- Raw per (relation, draw): operator top-1 predictions + output-state vectors on
  the probe set; faithfulness (real + shuffled); ≥ 20 records per headline cell.

## Resource estimate (RTX, 1.4B cuda)

Jacobian estimation is the cost (one backward per ICL context per relation per
draw); operator application is cheap. 8 relations × 3 draws × (estimate +
held-out eval + shuffled-control estimate) on 1.4B cuda → projected [RTX to
refine with a one-relation probe before the full run]; expected well under the
12 h ceiling. Detached + Monitor per the background-kill lesson.

## Certificate

certificates.json entries s3_operator@<relation>@pythia-1.4b: converged (bool),
functional agreement + faithfulness numbers, n_draws=3, ControlRecord pair
(positive faithfulness + negative shuffled), evidence run dir with raw
completions. Ecaterina's sign-off line in LABNOTES completes the M5.2
deliverable; M6's S3 rows do not start before it.

## Deviations

(none yet)

## Ratification

ratified: EXP-M5-2 2026-07-22 — Ecaterina (session instruction "ratify m5.2").
Thresholds ratified as drafted. The RTX builds its own orchestrator against this
committed prereg (the 0c precedent) and runs it on 1.4B; the S3 certificate +
sign-off remain Ecaterina's.
