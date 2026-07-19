# Preregistration — EXP-M5-0: substrate qualification (taxonomy phase M5.0)

- experiment-id: EXP-M5-0
- claim: none (qualification gate; no CLAIMS.md entry moves on this run — it
  produces the admitted species × substrate set and the per-substrate lens
  certificates that every M5/M6 measurement rests on)
- models: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
  (small anchor) and
  EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79
  (D-023 RULED 2026-07-20)
- configs: configs/m5_0_qualification_pythia410m.yaml,
  configs/m5_0_qualification_pythia1p4b.yaml,
  configs/m5_lens_pythia1p4b_draw{0,1,2}.yaml
- author + date: Claude (proposal), 2026-07-19. Thresholds, D-022 (LRE data
  vendoring), D-023 (1.4B pin), and compute placement (1.4B lens gate Mac
  overnight, skip4-only; binding on Mac; skip-sweep not purchased) RULED by
  Ecaterina 2026-07-20 (session instruction). Committing this file is the
  prereg act. Bars below marked [proposed] in the draft are ratified as
  drafted unless annotated otherwise.

## Hypothesis

No scientific hypothesis is tested. This run applies M5_SPEC.md §M5.0:
qualify the substrates and admit species per substrate before any species
extraction (M5.1+) or axis measurement (M6). It also discharges, for
Pythia-1.4B, the PROVISIONAL re-derivation duties (skip_first/n
calibration, band layers) and the lens-gate requirement that M6's
readout-based axes marginalize over >= 3 certified lens draws (E1 lesson).

## Decision rules (bars ratified as drafted 2026-07-20; [proposed] tags are historical)

Scoring everywhere: greedy argmax exact-match per D-012; deterministic
single pass (D-006 precedent: no RNG in the baseline stage, so the 3-draw
LAW attaches to the lens draws and later extraction draws, not to these
baselines). 10-shot is the admission regime; zero-shot is recorded
descriptively next to every number.

1. S1 concept-task admission (per substrate): 10-shot exact-match >= 0.80
   [proposed; M1 `baseline_threshold` precedent] on the task's full
   `tasks/*.json` set. Battery: the 8 v1 tasks (capital-operand,
   capital-recall, context-binding, multihop-scaled, opposites,
   swap-capitals, typo-robustness, word-pairs). 410M anchor values exist in
   the M1 baseline table; 1.4B is measured fresh here.
2. FV-task baselines (per substrate): capitalize, singular-plural,
   english-french (the D-007 set; membership fixed by D-007, not re-gated).
   Recorded for the 1.4B substrate with the same 0.80 bar; on 410M the M2
   certificates stand untouched.
3. LRE relation admission (per substrate; data = D-022 submodule): 10-shot
   relation accuracy >= 0.60. Reading RATIFIED by Ecaterina 2026-07-20: the
   M5_SPEC M5.0 bar is read as few-shot CAPABILITY >= 0.60 at qualification
   (operator faithfulness is only computable once M5.2 fits the operators);
   faithfulness >= 0.60 then serves as the M5.2 operator positive-control
   bar. 12-relation battery from evandez/relations @1b9ec3c, by dataset
   path: factual/{country_capital_city, landmark_in_country,
   food_from_country, product_by_company}; linguistic/{adj_antonym,
   verb_past_tense, word_first_letter, word_last_letter};
   commonsense/{object_superclass, fruit_inside_color, task_done_by_tool,
   work_location}. Per-relation N = min(50, remainder) held-out examples
   (country_capital_city has only 24 samples); N recorded per relation.
   Prompt uses the dataset's own prompt_templates[0] (10-shot ICL from the
   remaining samples). S3 is admitted on a substrate iff >= 8 relations
   pass there (M5_SPEC).
4. Binding battery (S4 admission, per substrate): Feng & Steinhardt-style
   n-entity binding — contexts binding n entities to n attributes
   ("The <object> is in the <container>. ... Q: Where is the <object>?"),
   subtasks n=2 and n=3, N = 60 queries each, plus the existing
   tasks/context-binding.json as the v1-anchored subtask (410M anchor
   0.533). S4 admitted iff the n=2 subtask 10-shot score >= 0.70 (M5_SPEC
   bar) [proposed: admission on n=2; n=3 and context-binding recorded
   descriptively]. Runs may be deferred to the GPU tier (placement ruling).
5. 1.4B lens gate (3 draws, M1-style; new-model rules — the v1-reproduction
   anchors R3/R4/R5 of EXP-M1 do not transfer). Draws: seeds 0/1/2,
   re-sampled calibration prompts, separate caches, skip4/n10 carried as
   the PROVISIONAL default and re-derived the M1 way — by gate evidence at
   skip4 on this model. RULED 2026-07-20: skip4-only, on the Mac overnight;
   the {2,4,8} skip sweep is NOT purchased now (a sweep-on-all-draws ≈ 16 h
   breaches the 12 h Mac LAW; if a later result needs the skip re-derived it
   goes to the GPU tier as a flagged run). Band re-derived from the probing
   profile. Rules, evaluated over the 3 draws:
   - Q1: every draw passes the vendored 9-check sanity gate.
   - Q2 (positive control): dp(swap_answer) median >= +0.30 with flip rate
     >= 75% [proposed absolute bars; anchors: 410M +0.60, GPT-2 +0.38].
   - Q3 (sham): |dp| median <= max(0.03, 1/N).
   - Q4 (negative control): 10-seed random-matrix arm and random-direction
     swap arm each |dp| <= max(0.03, 1/N).
   - Q5 (probing contrast): on >= 2 anchor tasks, band-min J-lens HMR <=
     5.0 with >= 5x separation from the logit-lens HMR [proposed; 410M
     anchor 2.5 vs 61.5].
   - Q6 (draw stability): dp IQR <= 0.05 and band-min HMR IQR <= 0.5
     across the 3 draws.
   PASS issues the 1.4B lens ControlRecord + 3 certified lens draws.

## What counts as failure

- A species × substrate battery below its bar: that cell is excluded from
  the taxonomy matrix, recorded in the qualification report. If 1.4B fails
  the LRE or binding gates, escalation to Pythia-2.8B is flagged for a
  compute ruling (TAXONOMY_DESIGN scope); nothing silently proceeds.
- 1.4B lens-gate failure (any Q-rule): 1.4B is inadmissible for
  lens-readout axes (A1/A3 readout arms); escalation ruling required.
  Non-lens axes (A2 potency vs sham) are unaffected by lens failure.
- Instrument-control failure (Q2-Q4) voids the lens verdict per the
  instruments LAW: the readout is withdrawn, and the report states an
  instrument failure, not a qualification result.
- Post-hoc analyses of stored tensors are labeled post-hoc forever.

## Estimator plan

[Conformance section added 2026-07-20 (text-only, D-015 precedent): the
prereg as first committed (113d04f) omitted this required heading, so
start_run rejected it; no threshold or decision rule changed. Flagged for
Ecaterina's acknowledgement.]

Two estimator classes, both deterministic at qualification (no RNG →
the 3-draw LAW attaches downstream, not here):

- Capability readouts (greedy exact-match, D-012): S1 task, FV task, LRE
  relation, and binding scores. Single deterministic pass per (battery,
  regime); ICL context fixed by seed 0. No estimator variance to gate;
  numbers feed admission only, not any claim.
- J-lens estimator (1.4B lens gate): the vendored fit → swap → probe
  pipeline (scripts 01-04) at skip4, run as 3 independent draws
  (seeds 0/1/2 re-sample the calibration prompts; separate caches). The
  per-draw lens is the estimator; Q6 gates its cross-draw stability
  (dp IQR, band-min HMR IQR). Median/IQR over the 3 draws are the only
  cross-draw summaries. The lens ControlRecord is the Q2 positive arm plus
  the Q3/Q4 matched-noise arms (swap random-direction control + the 10-seed
  random-matrix probe arm).

## Instruments

- Task/relation/binding readout: greedy exact-match (D-012), deterministic;
  its controls are the anchored 410M values (positive: tasks known >= 0.8
  on 410M reproduce; negative: multihop-scaled known-fail 0.500 stays below
  bar) [proposed as the in-run sanity pair].
- J-lens readout on 1.4B: ControlRecord = Q2 positive arm + Q3/Q4 negative
  arms, mirroring the M1 pattern that M4 relied on.

## Sample plan

- S1/context tasks: full task files (36-60 items each), 10-shot and 0-shot.
- FV tasks: full test splits (170/43/987).
- LRE: fixed 10-shot context per relation (seed 0), N >= 50 held-out
  examples per relation (or the relation's full remainder if smaller; N
  recorded per relation).
- Binding: N = 60 per subtask per regime.
- Raw per-item completions retained per cell:
  {battery}_{task}_{regime}.jsonl. Every headline cell >= 20 records.

## Resource estimate (Mac tier, M1 MacBook 16GB, MPS fp32)

Probe (scratchpad m5probe, 2026-07-19, non-scientific; pythia-1.4b@fedc38a
fp32 on MPS): model load 11.1 s; greedy 8-token generation 1.02 s/gen;
Jacobian fit 661.6 s/prompt (probe pass) and 621.6 s (fit pass) over 23
source layers at d=2048, dim_batch=8; peak RSS 7.39 GB at the fit stage
(ru_maxrss understates MPS-side allocations; treat as a floor).

- Baselines, all batteries, both substrates: ≈4.8k greedy generations at
  1.4B (≈1.0-2.0 s/gen, 10-shot prompts slower) ≈ 1.7-2.6 h; 410M ≈3×
  faster ≈ 35-50 min; total ≈ 2.5-3.5 h. Peak RSS ≈ 7 GB. Under the 12 h
  LAW.
- 1.4B lens gate, skip4-only (proposed default): 3 draws × 10 prompts ×
  ~640 s ≈ 5.3 h fits + evals ≈ 1.1-1.5 h (410M eval stage ~23 min × ≈3.4)
  → ≈ 7-8 h wall with 1.15 slack; peak RSS ≈ 7.5-9 GB on 16 GB unified.
  Under the 12 h LAW → Mac-overnight-eligible; placement is Ecaterina's
  ruling. Skip-sweep variants priced in rule 5.
- Launch mode if Mac: detached `nohup` + log monitor (M1 incident
  precedent).

## Deviations

(none yet)
