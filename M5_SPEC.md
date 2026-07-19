# M5_SPEC.md — species extractors + stability certificates (M2-gate style)

M5 delivers, per species: an extractor, a stability gate run, a certificate,
and instrument ControlRecords — BEFORE any A1–A5 axis measurement (M6).
Pattern per species mirrors M2: >= 3 extraction draws (seeds vary only the
extraction stream), convergence ladder with witness rung, sham arm,
positive + negative controls, certificate in certificates.json, prereg
committed before the gate run, thresholds ratified by Ecaterina.

## M5.0 Substrate qualification (blocks everything)

- Baseline-gate Pythia-1.4B (and 410M where applicable) on: the S1 concept
  tasks, >= 8 LRE relations (Hernandez et al. relation set, filtered to
  what the model can do zero/few-shot — faithfulness >= 0.6 to count),
  and a binding task battery (>= 70% baseline to admit S4).
- Deliverable: qualification report; the admitted species x substrate set.
  If 1.4B fails LRE/binding gates -> escalate to 2.8B (flag compute).
- New lens requirement: M1-style lens gate on each admitted substrate
  (3 draws, 9-check gate, PROVISIONAL skip4/n10 re-derived per model).

## M5.1 S1 concept directions

- Extractor: mean-difference over certified-task answer states (and
  entity-identity contrast pairs), final-position band layers; >= 3 draws
  via resampled context sets.
- Convergence ladder: n_contexts in {8, 16, 32, 64}; rule: min pairwise
  cosine >= 0.95 AND downstream-effect IQR <= 0.05 at T and every larger
  rung (M2 pattern).
- Controls: positive = the direction moves its own task readout
  (swap/injection, M1 machinery); negative = norm-matched random,
  quantization-aware bounds max(base, 1/N) from the start (D-010 lesson).

## M5.2 S3 relational operators (LRE)

- Extractor: per relation, affine map (W, b) via Jacobian + bias at a
  few-shot context (Hernandez et al. recipe), vendored/reimplemented with
  a landing test (operator applied to subject state reproduces the
  reported faithfulness on held-out subjects).
- Draws: >= 3 via resampled few-shot contexts and subject sets.
- Convergence: operator agreement measured functionally — top-1 output
  agreement >= 0.9 and output-state cosine >= 0.95 across draws on a fixed
  probe set (raw parameter cosine is reported descriptively; W is
  high-dim and parameter-space agreement is not the certified quantity).
- Controls: positive = faithfulness on held-out subjects >= the M5.0 bar;
  negative = shuffled-relation operator (train on mismatched s->o pairs)
  must fail faithfulness. Sham twin for later interventions = norm-matched
  random affine perturbation.
- Certificate covers BOTH measurement targets: the operator itself and the
  operator-output directions (the latter feed the S3-output row).

## M5.3 S4 binding vectors

- Admitted only if M5.0's binding battery passes.
- Extractor: Feng & Steinhardt-style binding-ID differences (entity/
  attribute position contrasts), >= 3 draws via resampled entity fillers.
- Convergence: cross-draw cosine ladder as M2; functional agreement =
  swap-binding intervention flips attribute assignment consistently.
- Controls: positive = binding swap flips the bound attribute above
  chance; negative = the same vector applied to unbound contexts moves
  nothing beyond sham.

## M5.4 S5 steering direction (optional)

- Sentiment-class mean-difference on a labeled prompt set; standard ladder;
  positive = shifts continuation sentiment classifier score; negative =
  sham. Include only if M5.1–M5.3 complete on schedule.

## Deviations / ban list carried forward

- BANNED: report-probe-forced-choice@singular-plural (D-013);
  k=25 gradient-pursuit J-fraction (CONSTRAINTS). Neither returns in M5/M6.
- Every readout-based axis in M6 marginalizes over >= 3 lens draws
  (E1 lesson) — M5's lens gates must therefore produce 3 certified draws
  per substrate.
- Report-coupling (A4) preregs inherit the E2 power caveat: paired
  contexts, continuous scores, pre-committed INCONCLUSIVE outcome.

## Definition of done (M5)

certificates.json contains entries for every admitted species x substrate;
each has: evidence run dir with raw completions, converged_at with witness
rung, >= 3 draws, ControlRecord pair, and Ecaterina's sign-off line in
LABNOTES. M6 (the axis battery) does not start before that line.
