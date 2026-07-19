# Preregistration — EXP-M4-E1: FV label decodability (J-lens vs logit lens)

- experiment-id: EXP-M4-E1
- claim: CLM-001
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m4_e1_pythia410m.yaml (lands with the build commit,
  before this file is committed; calibration/fit identity mirrors M1)
- author + date: Claude (proposal), 2026-07-18; thresholds, Hendel
  exclusion, and launch ratified by Ecaterina 2026-07-18, the variant
  proportional-inclusion rule 2026-07-19 (D-014, session Q&A). Committing
  this file is the prereg act.

## Hypothesis

HYPOTHESIS (CONSTRAINTS.md): FVs carry a J-lens-readable task-label component invisible to the logit lens.
v1 Exp-1 produced this on single-draw FVs (contaminated per the M2-cited
VERIFIED instability entry). E1 re-tests it on M2-certified FVs with the
full v2 draw discipline. Scope here: the three certified tasks only
(capitalize, singular-plural, english-french). Note: singular-plural is IN
scope — D-013 withdrew the forced-choice report probe on that task; the
lens readout used here is a different instrument, gated at M1.

## Decision rule

Statistic (v1 Exp-1 operationalization, vendored jvec/evals/fvprobe.py):
for a residual-space vector v, lens draw j, layer set L(j):

- R_jlens(v, j) = min over l in L(j), over the task's Set-1 label words'
  surface tokens t, of the full-vocab rank of t in unembed(J_l^(j) v)
  (rank 1 = top).
- R_logit(v) = the same min over Set-1 tokens in unembed(v) (no transport).

Label sets 1-3 and lens variants are v1's registered ones, fixed before
scoring: Set-1/2/3 exactly as vendored in fvprobe.TASK_LABEL_WORDS and
14_harden_exp1.LABEL_SETS (restricted to the three tasks); variants
skip4_n10 (primary), skip16_n10, skip4_n5.

Grid: FV draws i in {1,2,3} (M2 seeds), lens draws j in {0,1,2} (M1 seeds;
primary variant); variants x label-sets at lens draw 0 only. Headline
number per task: median and IQR of R_jlens over the 9 (i x j) primary
cells; R_logit has 3 values (no lens dependence), median/IQR over draws.

Per-task criteria, all fixed now [constants await Ecaterina's ruling]:

- C1 (lens-arm decodability): median R_jlens over the 9 primary cells <= 20.
- C2 (ordering, every cell): R_jlens < R_logit in all 9 primary cells AND
  in all 27 robustness cells (3 FV draws x 3 variants x 3 label sets, lens
  draw 0). Robustness cells voided by a variant's failed control are
  recorded not-evaluable and excluded (D-014); primary cells can never be
  excluded.
- C3 (logit-arm floor): median R_logit over FV draws >= 200.
- C4 (random anchor, per FV draw): at lens draw 0, primary variant, Set-1,
  R_jlens(v_i, 0) beats >= 95 of 100 norm-matched random vectors (seeds
  1000-1099, norms matched per FV draw).

Task verdict: DECODABLE-AND-SEPARATED iff C1-C4 all hold; NOT-DECODABLE if
C1 or C4 fails; NOT-SEPARATED if C1 and C4 hold but C2 or C3 fails;
INSTRUMENT-VOID if either in-run instrument control (below) fails for that
task — void readings are neither for nor against the hypothesis.

Descriptive arm (recorded, outside the decision rule): output-vocabulary
mean rank per arm (v1's second statistic), retained for E2/E3 design input.

CLM-001 moves hypothesis -> preliminary iff >= 1 task is
DECODABLE-AND-SEPARATED and no task is INSTRUMENT-VOID; the claim statement
records the full per-task verdict table either way.

## What counts as failure

- Zero tasks DECODABLE-AND-SEPARATED (C1-C4 evaluable everywhere): evidence
  against the hypothesis at this model/config; CLM-001 stays `hypothesis`
  and the counter-evidence is recorded in LABNOTES + the claim entry.
- NOT-SEPARATED outcomes count against the "invisible to the logit lens" clause of the HYPOTHESIS specifically.
- Any INSTRUMENT-VOID task: no verdict for it; the readout is redesigned
  and re-controlled under a new prereg before that task is read again
  (instruments LAW).
- Lens identity mismatch on any draw (prerequisite below): abort, raw
  replay before any re-run.
- Post-hoc analyses of this run's records are labeled post-hoc forever.

## Estimator plan

- FVs: fv_todd per task per draw at the full stored trial count
  (n_trials_aie=200 >= converged_at=25), loaded as StabilityGatedFV with
  the M2 certificate payloads
  (results/m2/20260718-114950-fv-stability-gate/certificates.json;
  evidence commit 6a3a00b). Backing tensors: cache/m2/draw{1,2,3}/fvs.
  Hendel vectors are OUT of scope: no certificate (M2 measured them
  descriptively only).
- Lens: three M1 draws. Draw 0 is the M3-re-materialized instrument
  (identity + spot-check already reproduced on this machine); draws 1-2
  are refit here with the same identity protocol: calibration sha256 and
  fit hyperparameters must equal the committed M1 manifests per draw.
  The two robustness variants (skip16_n10, skip4_n5; lens draw 0
  calibration stream) have no M1 manifest; they carry their own in-run
  controls (below) and feed only C2.
- Median/IQR is the only summary; no number is reported from fewer than 3
  draws of the estimator that produced it.

## Instruments

The J-lens label-rank readout on static vectors is a new instrument
surface (M1 gated the lens on residual streams predicting answers). In-run
ControlRecord pair, per task and per lens draw/variant, fixed before any
FV is read:

- Positive control (round-trip detection ceiling): for each layer l of
  the instance's band-overlap layer set, v+_l = pinv_trunc(J_l, rcond
  0.05) @ W_U[t*], with t* the first surface token of the task's Set-2
  word (single-word label). Layer l PASSES iff the rank of t* in
  unembed(J_l v+_l) is <= 10. The control PASSES iff >= ceil(0.75 x
  n_layers) layers pass (D-014 proportional rule; for the primary
  13-layer instances this is exactly the ratified 10/13; skip16_n10's
  single band layer L16 must pass). L (the layer set every statistic
  minimizes over, for that instance) = the passing layers. A robustness
  VARIANT failing its control is void for that variant only: its C2
  cells are recorded not-evaluable and C2 is decided on the remaining
  cells. A PRIMARY instance failing, or the primary L(j) across lens
  draws differing by symmetric difference > 3 layers, is INSTRUMENT-VOID
  for the affected task(s).
- Negative control (no label from noise): per task and per lens
  instance, median over the 100 draw-1-norm-matched random vectors of
  R_jlens >= 100. (The lens-draw-0 readings also supply C4.)
- The lens itself carries the M1 ControlRecord pair
  (results/m1/20260718-010559-lens-gate) and the M3 re-materialization
  (results/m3/20260718-174954-instrument-gate). The forced-choice report
  probe is NOT used in E1 (its D-013 per-task withdrawal is untouched).

## Interventions and shams

None. E1 is read-only decodability: no model weights, activations, or
computations are modified — vectors are decoded outside the forward pass.
The sham-twin LAW is therefore not triggered; the norm-matched
random-vector arm is the reading-level null and is reported in the same
table as every FV reading.

## Sample plan

- Tasks: capitalize, singular-plural, english-french (certified set).
- Cells (raw records retained per the retention LAW, >= 20 records per
  headline cell by construction):
  - decode_<task>.jsonl — one record per (fv_draw x lens_draw x variant x
    label_set x arm) with per-layer ranks and top-10 decoded tokens
    (>= 36 records/task).
  - random_<task>.jsonl — 700 records: 100 seeds x 5 lens instances at
    the draw-1 norm (negative control) + 100 seeds x 2 companion FV-draw
    norms at lens draw 0 (the C4 cell; rank statistics are invariant to
    the norm rescaling, recorded nonetheless).
  - poscontrol_<task>.jsonl — one record per (lens draw/variant x band
    layer): 65 records/task.
  - outputvocab_<task>.jsonl — descriptive arm, one record per (fv_draw x
    arm) with per-token ranks.
- Report-probe phrasings: N/A (no report probe in E1).

## Resource estimate

Anchors: M1 committed fit 919 s (MacBook); M3 bounded a refit on this
machine at 15 min and completed within it; unembed/transport are single
matmuls (d=1024, V=50k).

- Lens fits (one-time, cached): 4 fresh fits (draws 1-2 primary variant;
  skip16_n10 + skip4_n5 at draw 0), 3-8 min each expected, bounded 15 min
  each => 12-60 min.
- Decodes: ~400 FV/pos-control readings + ~3,900 random-vector layer
  readouts, all single matmuls => < 5 min GPU. pinv on CPU (vendored
  convention), 65 matrices of 1024x1024 => < 5 min.
- Model load ~1 min. Total: ~25-50 min, bounded 75 min. Peak RSS <= 4 GB,
  VRAM ~2/8.2 GB (M3-like). Under the 12 h LAW with wide margin; > 10 min,
  so the run launches only after this file's commit and Ecaterina's go.

## Deviations

- D-015 (Ecaterina, 2026-07-19; text-only, after the run): the
  Instruments-section parenthetical "skip16_n10's single band layer L16
  must pass" rested on a false premise — skip_first in the vendored jlens
  excludes leading PROMPT POSITIONS from the Jacobian average
  (jlens/fitting.py valid_position_mask); it does not restrict source
  layers. Every lens instance has all 13 band layers (run evidence:
  positive controls 13/13 on every instance,
  results/m4/20260719-021823-e1-decodability/controls.json). The
  operative rule — >= ceil(0.75 x n_layers) — is unchanged and equals
  the ratified 10/13 at n=13 for every instance; no criterion, code, or
  verdict is affected. Full record: LABNOTES D-015.
