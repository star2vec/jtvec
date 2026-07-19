# Preregistration — EXP-M5-1: S1 concept-direction stability gate

- experiment-id: EXP-M5-1
- claim: none (certificate gate; produces the S1 stability certificates +
  ControlRecords that the S1 rows of the M6 axis battery rest on)
- models: the substrates admitted by EXP-M5-0 (410M expected; 1.4B pending
  its qualification + lens gate)
- config: configs/m5_1_concept_pythia410m.yaml (+ 1.4B sibling once
  admitted)
- author + date: Claude (proposal), 2026-07-19. Thresholds RULED "ratified
  as drafted" by Ecaterina 2026-07-20 (session instruction); committing
  this file is the prereg act. NOTE: the M5.1 RUN still waits on EXP-M5-0
  to produce the admitted-substrate set and the re-derived band; the prereg
  is committed early to lock the predictions. The 1.4B sibling config is
  drafted once 1.4B is admitted. Bars below marked [proposed] are ratified
  as drafted.

## Hypothesis

No scientific hypothesis is tested. This run applies the stability-gate
LAW to the S1 concept/entity extractor (M5_SPEC §M5.1), producing the
entry-requirement certificate that TAXONOMY_DESIGN demands per species
BEFORE any A1-A5 axis is measured. The S1 pole of H1 is unmeasurable
without this certificate.

## Extractor under the gate

s1_concept@<concept>: mean-difference direction over certified-task answer
states at the final position of band layers —
d(concept c) = mean(resid_final over contexts whose answer is c) −
mean(resid_final over matched contexts with other answers), unit-scaled to
the natural mean answer-state norm. Two concept families [roster proposed,
ratified with the thresholds]:

- capital-recall answer states: >= 8 capitals (Paris, London, Rome, Berlin,
  Madrid, Vienna, Athens, Cairo), contexts drawn from the certified
  capital-recall task templates;
- entity-identity contrasts: >= 8 country/city entities in neutral carrier
  templates (the M1/M3 apparatus).

Draws: 3, seeds {1, 2, 3}; only the context-resampling stream varies —
model weights, eval prompts, and templates held fixed (M2 pattern).

## Decision rule

Convergence ladder over n_contexts per class T ∈ {8, 16, 32, 64}; per
concept, per rung, over 3 draws:

- min pairwise cosine of d@T (3 pairs), and
- IQR over draws of the downstream effect: Δp(concept answer token) under
  injection of d@T at the band layers, final position, natural norm, on N
  fixed eval prompts (M1 swap machinery: truncated pinv rcond 0.05, norm
  preservation).

A rung passes iff min pairwise cosine >= 0.95 AND downstream-effect IQR
<= 0.05 [M5_SPEC values; M2 semantics]. converged_at(concept) = smallest
passing T with every larger rung also passing; a pass at T=64 alone is NOT
convergence (no witness). The S1 species certificate on a substrate is
issued iff every roster concept has a converged_at <= 64 [proposed: whole
roster, not majority].

## Estimator plan

[Conformance section added 2026-07-20 (text-only, D-015 precedent): the
prereg as first committed (113d04f) omitted this required heading, so
start_run rejected it; no threshold or decision rule changed. Flagged for
Ecaterina's acknowledgement.]

The estimator is the s1_concept@<concept> mean-difference direction defined
under "## Extractor under the gate": 3 independent draws (seeds {1,2,3},
only the context-resampling stream varies), laddered over n_contexts ∈
{8,16,32,64} by prefix slicing of each draw's context stream (RNG-prefix
property, unit-tested before the run — M2 precedent). The per-draw direction
is the estimator; the gate (Decision rule) certifies its cross-draw
convergence. Median/IQR over the 3 draws are the only cross-draw summaries.

## Instruments and controls

- Positive control (per concept): the extracted direction moves its own
  readout — Δp(concept answer) median over draws >= +0.10 above the sham
  median [proposed bar], via the M1 swap/injection machinery.
- Negative control: norm-matched random direction (10 seeds), |Δp| <=
  max(0.02, 1/N) [quantization-aware bound from the start — D-010 lesson].
- Sham twin per (concept, draw, rung): norm-matched random direction, seed
  9100 + 10·draw_k + rung_index, injected at identical layers/positions;
  every reported effect appears with its sham in the same row
  (scoped_intervention).
- require_controlled() gates the verdict computation; control failure
  withdraws the instrument per the instruments LAW (no certificate, an
  instrument-failure report instead).

## What counts as failure

- Any roster concept without converged_at: documented non-convergence for
  the concept; if the roster rule then fails, no S1 certificate on that
  substrate — the S1 pole of H1 is unmeasurable there and the taxonomy
  matrix carries the hole explicitly (escalation to a deeper ladder or
  another substrate is a flagged ruling, not an automatic re-run).
- Control failure: instrument withdrawn (above).
- Post-hoc analyses of stored tensors are labeled post-hoc forever.

## Sample plan

- Extraction contexts: resampled per draw from the certified task template
  pools; rungs by prefix slicing of each draw's context stream (RNG-prefix
  property, unit-tested before the run — M2 precedent).
- Eval prompts: N >= 40 fixed prompts per concept readout; N recorded.
  Effect quantum 1/N <= 0.025 sits below the 0.05 IQR criterion.
- Raw per-item records per cell: {concept}_rung{T}_{arm}.jsonl (arm ∈
  injection, sham, random); every headline cell >= 20 records.
- Median/IQR are the only cross-draw summaries (DrawSet).

## Certificate

certificates.json entries s1_concept@<concept>@<substrate>: converged_at
with witness rung, n_draws=3, ControlRecord pair (positive + negative),
evidence run dir with raw completions. Ecaterina's sign-off line in
LABNOTES completes the M5.1 deliverable; M6's S1 rows do not start before
it.

## Resource estimate (Mac tier unless ruled otherwise)

Extraction is forward-only (no Jacobian, no gradients): 64 contexts × 16
concepts × 3 draws ≈ 3k forwards + ladder evals (4 rungs × 3 arms × N=40
× 16 concepts ≈ 8k short generations). Projected ~1-2 h at 410M, ~3-5 h at
1.4B (MPS fp32) — under the 12 h LAW; peak RSS dominated by weights
(410M: ~3 GB; 1.4B: ~7 GB). [Refined with the EXP-M5-0 probe's measured
s/forward before ratification.]

## Deviations

(none yet)
