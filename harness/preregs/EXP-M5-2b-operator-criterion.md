# Preregistration — EXP-M5-2b: S3 operator-criterion diagnostic (1.4B)

- experiment-id: EXP-M5-2b
- status: **RATIFIED** by Ecaterina 2026-07-22 (session "ratify"). Committing this
  file is the prereg act; the RTX builds + runs against it. Conditions folded in:
  both branches fixed; NO SIXTH CRITERION (close-but-under = FAIL, full stop);
  D-034 fixed negative control; report per-relation + roster. Bars ratified as
  drafted. THE LAST budgeted amendment cycle — 0 remain after.
- AMENDMENT BUDGET: this **counts against budget** (Ecaterina). After EXP-M5-1d
  (1 cycle) one cycle remained; **this is the LAST cycle — 0 remain after it.**
  When it resolves, M6 runs on whatever is certified and unmeasurable species are
  reported as such (CONSTRAINTS amendment-budget LAW).
- claim: none (S3 convergence-criterion diagnostic).
- models: EleutherAI/pythia-1.4b@fedc38a (the S3-admitted substrate); RTX cuda.
- config: configs/m5_2b_operator_criterion_pythia1p4b_cuda.yaml (drafted by the
  RTX with the orchestrator on ratification; reuses the EXP-M5-2 estimator).
- author + date: Claude (proposal), 2026-07-22.

## Hypothesis

EXP-M5-2 returned 0/8 converged because the top-1 OUTPUT-TOKEN agreement was low
(0.00–0.80, none ≥ 0.90) DESPITE decent output-STATE cosine (0.90–0.95; 2/8 clear
0.95). That is argmax draw-instability atop direction-space stability — the SAME
phenomenon as v1 FV draw-instability and as EXP-M5-1d's argmax-insensitivity: one
instrument problem (argmax is a brittle read of a stable-in-direction quantity)
across three species. The same argmax→direction-space fix Ecaterina approved for
1d is applied here.

Two branches (stopping rules fixed below):
- H-S3-DIRECTION-STABLE: S3 certifies under a draw-marginalized / cosine-based
  operator criterion → S3 ADMITTED, plus the cross-species headline "operators
  are stable in direction, unstable in argmax."
- H-S3-HARD: S3 fails even the marginalized criterion → recorded as genuinely
  hard-to-measure at 1.4B; NO further probes (budget exhausted; the taxonomy
  carries the S3 hole explicitly).

## What changes vs EXP-M5-2 (criterion only; estimator UNCHANGED)

The estimator (vendored JacobianIclMeanEstimator → LinearRelationOperator, 3
draws, EXP-M5-2 §Extractor) is unchanged. Only the convergence criterion changes,
from top-1 output agreement to a direction-space + draw-marginalized pair:

1. **Direction-space convergence:** cross-draw output-STATE cosine ≥ [proposed
   0.95] on the fixed probe set (the stable quantity M5.2 already showed at
   0.90–0.95). This REPLACES top-1 output-token agreement.
2. **Draw-marginalized faithfulness:** the draw-ENSEMBLE operator — per-probe
   majority-vote top-1 across the 3 draws' operators (and, descriptively, the
   mean-(W,b) operator) — achieves held-out faithfulness ≥ [proposed 0.60] (the
   M5.0 bar). This checks the marginal operator is functional despite per-draw
   argmax churn.

## Decision rule (bars [proposed] until ratified)

Per relation: CERTIFIES iff output-state cosine ≥ 0.95 AND draw-marginalized
faithfulness ≥ 0.60 AND the D-034 negative control fails (below). Raw top-1
agreement and raw per-draw faithfulness are reported descriptively (to show the
argmax churn the marginalization absorbs).

Roster: **H-S3-DIRECTION-STABLE iff ≥ [proposed 6/8] relations certify;
H-S3-HARD iff < 6/8**. The 8 relations = the EXP-M5-2 set.

### Branch stopping rules (fixed before running — the LAST budgeted cycle)

- H-S3-DIRECTION-STABLE → S3 ADMITTED on 1.4B under the marginalized criterion;
  the certificate + the "stable-in-direction, unstable-in-argmax" cross-species
  story (v1 FV / 1d / S3) go to Ecaterina's sign-off. This run writes NO
  certificate.
- H-S3-HARD → S3 recorded as not stably measurable at 1.4B; **no further probes**
  (budget exhausted). The taxonomy reports S3 as a hole, honestly.
- **NO SIXTH CRITERION (Ecaterina 2026-07-22):** if 2b lands close-but-under any
  bar, that is a **FAIL (H-S3-HARD)**, NOT a prompt for another criterion or a
  bar adjustment. Whatever 2b returns is the S3 disposition, FULL STOP. This is
  the final budgeted amendment cycle; after it, M6 runs on what is certified.

## Instruments and controls

- Positive control: draw-marginalized faithfulness on held-out subjects ≥ 0.60
  (the operator predicts the relation), per relation. require_controlled().
- Negative control: **the D-034 control** — a cross-relation (unrelated-relation)
  operator applied to this relation's probe (and/or a random-token-prompt
  operator) MUST FAIL faithfulness (≤ 0.10). Label-shuffling is banned (it does
  not null the model-derived Jacobian; EXP-M5-2 evidence).
- The L12 estimation layer is NOT re-selected (the M5.2 discipline: no
  layer-shopping); only the convergence CRITERION changes.

## What counts as failure

- Control failure (positive or the D-034 negative) → instrument withdrawn.
- If even the direction-space + marginalized criterion fails on ≥ 3 relations,
  that is the H-S3-HARD outcome, not a call for a fourth criterion.
- Post-hoc analyses of the stored M5.2 tensors are labeled post-hoc forever;
  this is a fresh run (or a pre-registered re-analysis of M5.2's retained raw,
  labeled as such).

## Estimator plan

The unchanged EXP-M5-2 estimator; 3 draws; the criterion is computed over the
retained per-draw operator outputs (majority-vote + cosine). Median/IQR of the
descriptive per-draw quantities reported; the certified quantities are the
marginalized cosine + faithfulness (DrawSet where applicable).

## Sample plan

- Fixed held-out probe set per relation (the EXP-M5-2 probe set), N recorded.
- Raw per (relation, draw): operator top-1 + output states on the probe set;
  the majority-vote marginal; the cross-relation negative arm. ≥ 20 records/cell.

## Resource estimate (RTX, 1.4B cuda)

Reuses the EXP-M5-2 estimation (or its retained tensors); the criterion is a
cheap post-pass. If re-estimating: ~the EXP-M5-2 cost (well under 12 h). Detached
+ Monitor. Nothing on the Mac.

## Deviations

(none yet)

## Ratification

ratified: EXP-M5-2b 2026-07-22 — Ecaterina ("ratify"). The LAST budgeted
amendment cycle (0 remain after). The RTX builds its own orchestrator against
this committed prereg and runs on 1.4B; whatever it returns is the S3 disposition,
full stop (no sixth criterion). The certificate + sign-off remain Ecaterina's.
