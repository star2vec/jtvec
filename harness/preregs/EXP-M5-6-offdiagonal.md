# Preregistration — EXP-M5-6: off-diagonal test of the decodability × potency 2×2 (S5 steering, 410M)

- experiment-id: EXP-M5-6
- status: **RATIFIED** by Ecaterina 2026-07-22 ("ratify, run it" + confirmed the
  A1 decode_vector correction). A FRESH experiment, NOT an amendment: the
  amendment budget does NOT apply. Full apparatus (own prereg, sham twins, ≥ 3
  draws median/IQR, positive+negative controls, same statistics as S2). Committing
  this file is the prereg act; runs on the Mac. The A2 potency axis is
  feasibility-confirmed (sham-controlled ΔS +11.9).
- claim: none until run (tests H1's double-dissociation; result is a new CLAIMS
  entry at Ecaterina's verify).
- models: EleutherAI/pythia-410m@9879c9b — the SAME footing as S1/S2 (their
  substrate; the A1 J-lens is M1-certified + null-check-validated here; the A2
  injection/ablation instruments are the 1b/1d ones).
- config: configs/m5_6_offdiag_pythia410m.yaml (drafted with the orchestrator).
- author + date: Claude (proposal), 2026-07-22, per Ecaterina.

## Hypothesis

The anchor dichotomy has S1 = (decodable, not-potent) and S2 = (not-decodable,
potent) — the DIAGONAL of the (A1 decodability × A2 potency) 2×2. Two diagonal
corners are explainable by ONE hidden axis (a deflation mode where decodability
and potency are anti-correlated by a single latent), so the dichotomy is
currently SUGGESTIVE, not a proven double dissociation. Filling ONE OFF-DIAGONAL
cell — a direction that is BOTH decodable AND potent, or NEITHER — refutes the
single-axis explanation and proves the two axes are independent.

Primary candidate: S5 STEERING vectors — untouched by the taxonomy so far, and
built to be potent (activation-addition steering). Measured on A1 (J-lens
decodability) × A2 (injection + ablation potency), on the same footing as S1/S2.

Two branches, decided by this run (fixed below, before running):
- H-AXES-INDEPENDENT: S5 lands OFF-diagonal — (decodable, potent) or (neither) —
  → the deflation-mode single-axis explanation is refuted; the two axes are
  independent; the S1/S2 dichotomy is a genuine DOUBLE DISSOCIATION. Strong.
- H-AXES-COUPLED: S5 lands ON-diagonal (the S1 corner or the S2 corner) → the
  single-axis explanation is NOT excluded; axes not shown independent; reported
  honestly as coupling not ruled out.
Both publishable.

## Extractor under test

s5_steer: a sentiment STEERING vector — mean-difference over the band-layer
final-position states of positive vs negative sentiment prompts (M5_SPEC §M5.4),
per band layer, unit-scaled to the natural answer-state norm (the same extractor
FORM as s1_concept, different class contrast). ≥ 3 draws, seeds {1,2,3}; only the
sentiment-prompt-resampling stream varies. Sentiment prompt pool: a fixed
labelled set (positive/negative short templates); pool + labels recorded.

## A1 — decodability instrument (E1 decode_vector — the SAME instrument that read S2)

Corrected 2026-07-22 (Ecaterina confirmed): the first draft named the max-contrast
ratio; same-footing-with-S2 requires the E1 instrument (which gave S2 its
NOT-DECODABLE, CLM-001), NOT the lens-gate statistic. So A1 = E1 decode_vector
(jvec.evals.fvprobe logic): per band layer, the J-lens readout unembed(J_l · v)
vs the logit-lens readout unembed(v), scored by LABEL-RANK — the min full-vocab
rank of the sentiment-polarity label words (the S5 analogue of E1's task-label
words). 9 primary cells (3 steering draws × 3 cached 410M lens draws), median
over band layers within a cell then median over cells (the E1 structure + the
lens-draw marginalization lesson).

DECODABLE iff (E1 C1 + C3, ratified D-014): median jlens label-rank ≤ 20 (the
J-lens verbalizes the sentiment polarity) AND median logit label-rank ≥ 200 (the
logit lens does NOT — the contrast). NOT-DECODABLE otherwise. Same statistic and
bars that read S2.

## A2 — potency instruments (injection + ablation, same as S1)

Readout (flagged design choice): sentiment-logit-difference S = logit(positive-
sentiment token set) − logit(negative-sentiment token set) at the final position,
mean over N fixed NEUTRAL carrier prompts — token-level, same footing as S1's
Δlogit. (A continuation sentiment-classifier score is reported descriptively as a
secondary, not the gate.)

- Injection (primary, the 1b machinery): inject the steering vector at band-layer
  final positions, natural norm; effect = ΔS = S_injected − S_clean, sham-
  controlled (norm-matched random injection). POTENT iff sham-controlled ΔS
  median ≥ [proposed 1.0 logit] with cross-draw transfer.
- Ablation (corroborating, the 1d machinery): project the steering vector out at
  band-layer final positions; sham-controlled ΔS-drop, same statistic.
Potency verdict = POTENT iff the INJECTION arm clears its bar (S5 is built to
steer via addition); ablation corroborates. (Mirrors S1: injection is the
load-bearing axis, ablation corroborating.)

## Decision rule (bars [proposed] until ratified)

Classify S5 into the 2×2 by (A1 decodable? A2 potent?), each a ≥3-draw
median/IQR verdict against its bar with its controls passing:
- OFF-diagonal = (decodable, potent) OR (¬decodable, ¬potent) → **H-AXES-
  INDEPENDENT** (double dissociation; single-axis deflation refuted).
- ON-diagonal = (decodable, ¬potent) [S1 corner] OR (¬decodable, potent) [S2
  corner] → **H-AXES-COUPLED** (single-axis explanation not excluded).
The verdict is the cell S5 occupies; no forced call if a control fails
(INCONCLUSIVE, below).

## Instruments and controls (full apparatus)

- A1 positive control: a direction known J-lens-decodable (a certified-task
  concept/latent direction) reads DECODABLE (low jlens label-rank) under
  decode_vector. A1 negative (E1 ReadoutNegativeRule): norm-matched random
  directions read NOT-DECODABLE (jlens label-rank high in both arms).
- A2 positive control: a known-potent direction (the sentiment-token unembed
  difference) injected moves ΔS ≥ the potency bar (the knob is live — the null-
  check +0.80 precedent). A2 negative: norm-matched random directions, |ΔS| ≤
  max(quantum, sham floor) — quantization-aware.
- Sham twin in every A1 (random arm) and A2 (random injection/ablation) row.
- ≥ 3 draws, median/IQR (DrawSet) the only cross-draw summaries.
- require_controlled() gates each axis independently; a failed control on an axis
  makes that axis INCONCLUSIVE (and the cell-verdict inconclusive), NOT a pole.

## What counts as failure

- If the A2 positive control does not fire (410M steering too weak to move ΔS
  above sham), the potency axis is INCONCLUSIVE — flagged, and the (decodable,
  potent) corner is untestable on 410M; the (neither) corner may still be
  reachable if S5 is also ¬decodable. Investigated, not forced.
- Control failure → the axis's instrument withdrawn.
- Post-hoc analyses labeled post-hoc forever.

## Estimator plan

Forward-only. The s5_steer extractor (3 draws); A1 = the cached-lens max-contrast
probe; A2 = injection + ablation ΔS on N carriers. Median/IQR over draws.
Deterministic given a draw's sentiment sample + sham seeds.

## Sample plan

- Sentiment extraction prompts: ≥ [proposed 32] pos + 32 neg per draw, resampled.
- Sentiment-token sets: fixed pos/neg token lists (recorded).
- A2 carriers: ≥ [proposed 40] fixed neutral prompts.
- Raw per cell (A1 ratios per draw; A2 ΔS per carrier per draw, inject/sham/
  ablate/random) ≥ 20 records.

## Resource estimate (Mac tier, 410M, MPS fp32)

A1 lens probe on cached draws ~10–15 min; extraction + A2 injection/ablation ΔS
(3 draws × ~40 carriers × arms) ~20–35 min. **Projected ~35–50 min wall, peak
~2.5 GB — Mac.** Detached + Monitor. (Machine = the Mac, same footing as S1/S2;
the RTX is free but 410M + the 410M lens cache live on the Mac.)

## Deviations

- A1 instrument corrected pre-run (2026-07-22, Ecaterina confirmed): from the
  max-contrast ratio first drafted to E1 decode_vector + the C1/C3 label-rank
  rule — the instrument that produced S2's NOT-DECODABLE, required for a valid
  same-footing 2×2. Bars: jlens label-rank median ≤ 20 AND logit ≥ 200. No other
  change.
- A2 feasibility probe (2026-07-22, pre-ratification, scratchpad): the sentiment
  steering vector moves the sentiment-logit-difference ΔS by sham-controlled
  +11.9 on 410M (+steer +9.1, −steer −2.7, sham −2.8) — the potency axis is
  testable; the (decodable, potent) corner is reachable.

## Ratification

ratified: EXP-M5-6 2026-07-22 — Ecaterina ("ratify, run it"; A1 decode_vector
correction confirmed). Bars: A1 = E1 C1/C3 (jlens label-rank median ≤ 20 AND
logit ≥ 200); A2 = injection sham-controlled ΔS ≥ 1.0 logit + cross-draw transfer,
A2 positive control ≥ 1.0. Fresh experiment; amendment budget does not apply.
Runs on the Mac; result is a new CLAIMS entry at Ecaterina's verify.
