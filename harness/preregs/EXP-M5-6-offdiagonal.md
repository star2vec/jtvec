# Preregistration — EXP-M5-6: off-diagonal test of the decodability × potency 2×2 (S5 steering, 410M)

- experiment-id: EXP-M5-6
- status: **CONSTRUCT REFRAME — pending Ecaterina's re-ratification; RUN HELD.**
  Ratified 2026-07-22 in the original off-diagonal framing; the pre-run smoke then
  exposed a construct problem (S5 is logit-trivially decodable, so A1 conflates two
  sub-axes) and Ecaterina ruled the reframe below (A1 decomposes; report both
  sub-axes; do NOT chase a bar). This file is redrafted to that framing for her
  re-ratification; NO run until she ratifies the reframe. A FRESH experiment
  (budget does not apply); full apparatus. A2 potency feasibility-confirmed
  (ΔS +11.9). Runs on the Mac.
- claim: none until run (tests H1's double-dissociation; result is a new CLAIMS
  entry at Ecaterina's verify).
- models: EleutherAI/pythia-410m@9879c9b — the SAME footing as S1/S2 (their
  substrate; the A1 J-lens is M1-certified + null-check-validated here; the A2
  injection/ablation instruments are the 1b/1d ones).
- config: configs/m5_6_offdiag_pythia410m.yaml (drafted with the orchestrator).
- author + date: Claude (proposal), 2026-07-22, per Ecaterina.

## Hypothesis

The apparent (A1 decodability × A2 potency) dichotomy — S1 = (dec, ¬pot),
S2 = (¬dec, pot) — could be ONE hidden DEFLATION axis rather than two independent
ones. The named deflation axis is OUTPUT-ALIGNMENT: a direction aligned with the
unembedding is logit-readable (looks decodable) AND moves the output when injected
(looks potent), so a single latent could produce the diagonal. An off-diagonal
cell refutes deflation ONLY IF the cell is not itself explained by output-
alignment; otherwise filling it is circular.

Construct correction (2026-07-22, from the pre-run smoke; Ecaterina's ruling):
A1 "decodability" CONFLATES TWO SUB-AXES that output-alignment pulls apart —
- A1a: the J-lens reads the content AT ALL (E1 C1: jlens label-rank ≤ 20).
- A1b: the J-lens reads what the LOGIT lens CANNOT (E1 C1 ∧ C3: jlens ≤ 20 AND
  logit ≥ 200) — "J-lens-privileged" decodability.
For output-MISaligned directions (S2 FV: jlens 278, logit dark) A1a and A1b
COINCIDE. For output-ALIGNED directions (S5 steering: jlens ~4 ≈ logit ~6) they
SEPARATE — S5's decodability is LOGIT-TRIVIAL (the logit lens gets it for free),
not A1b-privileged, and S5's decodability and its potency are the SAME fact
(output-alignment). So S5 does NOT refute deflation; it DECOMPOSES A1.

Primary candidate: S5 STEERING vectors (built potent via activation-addition,
output-aligned by construction). This experiment does NOT chase whichever bar
yields an off-diagonal; it reports S5 on BOTH sub-axes + the raw jlens-vs-logit
label-ranks, as the case that decomposes A1.

Branches (fixed before running):
- H-A1-DECOMPOSES (expected from the smoke): S5 is A1a-decodable (C1) but NOT
  A1b-decodable (fails C3 — logit-trivial). The finding is "A1 decomposes into
  two sub-axes (reads-at-all vs J-lens-privileged), separated by output-
  alignment" — NOT a double dissociation (HYPOTHESIS). S5 is NOT labelled
  off-diagonal; the deflation axis is NOT refuted.
- H-AXES-INDEPENDENT (would require A1b-privileged AND potent, which output-
  alignment cannot supply): S5 is A1b-decodable (jlens ≤ 20 AND logit ≥ 200) AND
  potent → an off-diagonal NOT explained by output-alignment → the double
  dissociation (HYPOTHESIS) is supported. Pre-registered though not expected.
- Other cells reported as they fall.
Both branches publishable.

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

## Decision rule (report BOTH sub-axes + raw ranks; never collapse to one)

Report S5 explicitly on both A1 sub-axes and the raw ranks:
- A1a-decodable iff median jlens label-rank ≤ 20 (C1).
- A1b-decodable iff median jlens ≤ 20 AND median logit ≥ 200 (C1 ∧ C3).
- the raw median jlens and median logit label-ranks reported SIDE BY SIDE, so the
  reader sees whether S5 is logit-trivially decodable (jlens ≈ logit, both low) or
  J-lens-privileged (jlens low, logit high).
- A2-potent iff sham-controlled injection ΔS median ≥ 1.0 with cross-draw transfer.

Verdict:
- S5 A1a-decodable ∧ ¬A1b-decodable (jlens ≈ logit, logit-trivial) → **A1-DECOMPOSES**:
  A1 splits into reads-at-all vs J-lens-privileged, separated by output-alignment;
  S5 is NOT off-diagonal; the double dissociation (HYPOTHESIS) is NOT supported
  and deflation is NOT refuted. (The expected outcome from the smoke.)
- S5 A1b-decodable ∧ potent → **AXES-INDEPENDENT**: an off-diagonal not reducible
  to output-alignment; the double dissociation (HYPOTHESIS) is supported.
- Control failure on an axis → that axis INCONCLUSIVE, the verdict inconclusive.
S5 is NEVER labelled "off-diagonal" on the strength of A1a alone.

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
  testable; steering is potent.
- CONSTRUCT REFRAME (2026-07-22, from the wiring smoke; Ecaterina's ruling): the
  smoke showed S5's A1 = jlens label-rank ~4 ≈ logit ~6 — logit-trivially
  decodable, output-aligned. Since output-alignment is the named deflation axis,
  scoring S5 "decodable" by C1-only and calling it off-diagonal would be circular
  (its decodability and potency are the same fact). The Hypothesis + Decision rule
  were reframed from "fill an off-diagonal → double dissociation" to "A1
  decomposes into A1a (reads-at-all) vs A1b (J-lens-privileged), which output-
  alignment separates." No bar was chosen to yield an outcome; both criteria +
  raw ranks are reported. Pending Ecaterina's re-ratification before the run.

## Ratification

ratified: EXP-M5-6 2026-07-22 — Ecaterina ("ratify, run it"; A1 decode_vector
correction confirmed). Bars: A1 = E1 C1/C3 (jlens label-rank median ≤ 20 AND
logit ≥ 200); A2 = injection sham-controlled ΔS ≥ 1.0 logit + cross-draw transfer,
A2 positive control ≥ 1.0. Fresh experiment; amendment budget does not apply.
Runs on the Mac; result is a new CLAIMS entry at Ecaterina's verify.
