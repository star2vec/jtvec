# Preregistration — EXP-M5-1b: S1 concept-gate diagnostic (410M)

- experiment-id: EXP-M5-1b
- status: **RATIFIED (conditions folded in); HOLD CLEARED** — RATIFIED WITH
  CONDITIONS by Ecaterina 2026-07-21 (D-033, governance item 3); the hold's
  release-condition ("EXP-M5-1c null-check clears") was met 2026-07-22 (null-check
  PASS, results/m5/20260722-012727-m5-1c-nullcheck) and Ecaterina released it
  ("fire 1b"). Committing this file is the prereg act; conditions (a)/(b)/(c) are
  folded in below. Bars marked [proposed], ratified as drafted.
- claim: none (instrument diagnostic; decides whether the EXP-M5-1 gate FAIL on
  410M was a resolvable instrument problem or a genuine S1 extractor result).
  Produces no certificate.
- models: EleutherAI/pythia-410m@9879c9b (the M5.1 substrate)
- config: configs/m5_1b_concept_diagnostic_pythia410m.yaml (standard Config
  schema, band [4,16]; drafted with this file).
- author + date: Claude (proposal), 2026-07-21, per D-033.

## Background (the FAIL this diagnoses)

EXP-M5-1 on 410M (results/m5/20260721-132124-m5-1-concept): S1 certificate NOT
issued, positive control 0/8, converged_at=None 8/8. Raw replay isolated two
instrument problems: the ladder stopped at T=64 while the mean-difference cosine
was still climbing (0.6→0.94), and the potency readout sat at the p=0.00056
concept-token floor where an absolute Δp cannot resolve.

## Hypothesis (verdict NOT pre-named — condition (c))

The run decides between, without a favoured outcome:

- H-RESOLVABLE-POSITIVE: under a longer (but ceiling-bounded) ladder and a
  potency readout made resolvable in its own terms, the s1_concept direction
  crosses the convergence bar with a witness AND moves the concept readout above
  sham. The M5.1 FAIL was a resolvable instrument problem.
- H-NEGATIVE-CONVERGENCE: the cosine PLATEAUS below the bar within the ceiling —
  the direction converges to something that is not a certifiable concept
  direction (a real negative on S1, not a call to extend further).
- H-UNMEASURABLE-POTENCY: the potency quantity cannot be resolved even after
  increasing trials — S1 potency is declared unmeasurable at 410M, not
  reinterpreted into significance.

The cosine trajectory is consistent with BOTH a merely-short ladder and
convergence to a non-concept direction; nothing here presumes which.

## What is varied vs EXP-M5-1 (only the instrument, never the extractor def)

The extractor is UNCHANGED (residual mean-difference over certified
capital-recall answer states, band [4,16], 3 draws seeds 1/2/3). Two instrument
changes, each with a pre-fixed stopping rule:

### (a) Ladder extension WITH a fixed ceiling and plateau test — condition (a)

RUNGS {8,16,32,64,128,256}. **Ceiling = 256, firm: the ladder is NOT extended
again beyond 256 under any outcome.** Per concept:
- converged_at is the smallest rung crossing the 0.95 cosine bar with the
  witness-rung rule (256 witnesses 128).
- PLATEAU test (fixed now): let Δcos(T) = cos@T − cos@(T/2). The direction has
  PLATEAUED BELOW THE BAR iff cos@256 < 0.95 AND Δcos(128) < [proposed 0.01] AND
  Δcos(256) < [proposed 0.01] — diminishing returns short of the bar. A plateau
  below the bar is a NEGATIVE result on S1 convergence (H-NEGATIVE-CONVERGENCE),
  NOT a request to extend.
- If cos@256 < 0.95 but the increments have not collapsed (still climbing at the
  ceiling), the certificate still does not issue — recorded as ceiling-limited,
  and, per condition (a), the ladder is still not extended again.

### (b) Potency floor is a RESOLUTION failure — condition (b)

The p=0.00056 floor is fixed by making the quantity resolvable IN ITS OWN TERMS
(Δp on the concept token), NOT by switching to a log/odds metric that rescues
the floor. Two levers, pre-specified:
- Raise N_eval to [proposed N=200] readout carriers (quantum 1/N = 0.005, an
  order below the +0.10 potency scale), N recorded.
- Raise resolution by INJECTION STRENGTH, not by re-labelling the metric: the
  dose sweep alpha ∈ {1,2,4,8}×natural-norm on the existing off-target carriers.
  At higher alpha a genuinely potent direction produces a resolvable Δp(c) (it
  overcomes the competing prediction); an inert direction stays at the
  quantization floor across all alphas. (Off-target carriers keep p_base low so
  any rise is attributable to the injection; the alpha axis, not a bespoke
  neutral-carrier corpus, supplies the resolution — cleaner to specify.)
Measured quantity stays sham-controlled Δp(concept). Pre-fixed unmeasurable
rule: if, at every alpha and under N=200 carriers, the sham-controlled
Δp(concept) does not exceed its own quantization bound max(0.005, 1/N), S1
potency is DECLARED UNMEASURABLE at 410M (H-UNMEASURABLE-POTENCY) — recorded as
unmeasurable, not reinterpreted.

## Decision rule (bars [proposed] until ratified)

Per concept, 3 draws, median/IQR:
- Convergence: converged_at ≤ 128 (256 witness) → PASS; plateau below bar →
  NEGATIVE; ceiling-limited-still-climbing → no-issue (recorded).
- Potency: sham-controlled Δp(concept) median ≥ [proposed +0.10] above sham with
  non-overlapping draw IQRs, at some alpha, with a monotone non-decreasing
  dose-response → PASS; below the quantization bound at all alphas → UNMEASURABLE.

Per-concept outcome ∈ {converged+potent, negative-convergence, unmeasurable-
potency, mixed}. S1 species outcome reported over the 8-capital roster; NO
certificate issues from this diagnostic (a diagnostic, not the gate). A
certificate, if warranted, is a separate EXP-M5-1 amendment + clean re-run,
itself downstream of a passed null-check.

## Estimator plan

Unchanged s1_concept mean-difference direction; a fresh forward-only run
(re-extraction ~cheap at 410M), NOT a post-hoc pass. 3 draws, seeds {1,2,3};
only the context-resampling stream varies. Deterministic given the draws.

## Instruments and controls

- Positive control (resolvable): under the N=200 neutral-carrier Δp readout at
  the ratified alpha, the direction moves its own readout ≥ the potency bar above
  sham — the control the floored off-target readout could not express.
- Negative control: norm-matched random directions (10 seeds), sham-controlled
  Δp within max(0.005, 1/N) (quantization-aware, D-010).
- Sham twin per (concept, draw, rung, alpha): norm-matched random at identical
  layers/positions/scale; every effect reported with its sham.
- require_controlled() gates the verdict; control failure → instrument-failure
  report, no verdict.

## What counts as failure

- Instrument-control failure voids the verdict.
- If the resolvable positive control does not fire even on the M1-certified
  capital-recall concept at any alpha, the injection pipeline (not the extractor)
  is suspect — investigated, e.g. the M1 lens-coordinate injection as the
  recorded fallback, not interpreted.
- Post-hoc analyses labeled post-hoc forever.

## Sample plan

- Concepts: the 8-capital roster. Extraction contexts identical to EXP-M5-1
  (seeds 1/2/3). Readout: N=200 fixed neutral carriers (seed EVAL_SEED),
  N recorded.
- Ladder: {8,16,32,64,128,256} by prefix slice (ceiling 256).
- alpha sweep {1,2,4,8}. Raw per cell {concept}_rung{T}_alpha{a}_{arm}.jsonl,
  ≥ 20 records; p_base, p_inj, Δp per item.
- Median/IQR over the 3 draws are the only cross-draw summaries.

## Resource estimate (Mac tier, 410M, MPS fp32)

Ladder-256 extraction ≈ 4× M5.1; N=200 carriers × 4 alphas × arms ≈ 6–8× the
M5.1 readout. Projected ~1.5–2.5 h wall (M5.1 was 785 s), peak RSS ~2 GB — under
the 12 h LAW, Mac-eligible, no swap. Detached + Monitor. [Refined against the
null-check's measured s/forward before this runs.]

## Deviations

(none yet)

## Ratification

ratified: EXP-M5-1b 2026-07-22 — Ecaterina. RATIFIED WITH CONDITIONS 2026-07-21
(D-033); HOLD CLEARED 2026-07-22 on the EXP-M5-1c null-check PASS + her "fire 1b".
Conditions (a)/(b)/(c) are folded in above. Issues no certificate; a certificate
is a later ratified EXP-M5-1 amendment + clean re-run.
