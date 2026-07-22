# Preregistration — EXP-M5-8: within-species breadth (n>1 per species)

- experiment-id: EXP-M5-8
- status: **RATIFIED** by Ecaterina 2026-07-23 with adjustments (S5 load-bearing
  → ≥3/4; instances named + anti-harvesting clause; draw-stability computed by one
  identical method across machines; S2 = the 3 certified FVs, ≥2/3). FRESH
  measurement, NOT an amendment — budget does not apply; full apparatus does.
  Committing this file is the prereg act. S1/S5 run on the Mac; S2 on the RTX.
- claim: none until run (tests whether the per-species profiles are type-general
  or n=1 artifacts).
- models: EleutherAI/pythia-410m@9879c9b (S1, S5; S2 FVs are 410M-certified).
- author + date: Claude (proposal), 2026-07-23, per Ecaterina.

## Why

Every species profile currently rests on n=1 — one concept direction (S1), one
function vector (S2), one steering vector (S5). So each profile is "this vector",
not "this representation type". This experiment extends each species to multiple
instances under the IDENTICAL instruments and asks whether the profile reproduces
within type, or the species is internally heterogeneous.

## Hypothesis

No claim about the models until run. The experiment tests whether each species'
n=1 profile is TYPE-GENERAL — reproduces across the named instances under the
identical instruments — or an n=1 artifact (internally heterogeneous). Per
species, the reference profile and its reproduction bar are in the Decision
rule; branches (PROFILE-REPRODUCES / HETEROGENEOUS) are fixed there.

## Estimator plan

The extractors are the certified n=1 estimators, UNCHANGED: S1 mean-difference
concept directions; S2 the M2-certified JacobianIclMeanEstimator Todd FVs
(cached tensors, no fresh extraction); S5 mean-difference steering vectors. The
per-axis statistics are identical to the n=1 runs (Axes section): draw-stability
= jtvec.concept_gate.min_pairwise_cosine over the 3 per-draw unit directions;
lens-readout/output-alignment = jvec.evals.fvprobe.decode_vector jlens/logit
label-rank over the cached lens draws; potency = injection (n_shot_eval + vector)
and ablation (project-out), sham-controlled. >= 3 draws per axis; cross-draw
summaries are DrawSet median/IQR. Deterministic given the fixed draws/instances.

## Sample plan

Instances are NAMED and FIXED in the Instances section (S1 n=5, S2 n=3, S5 n=4);
ALL are reported, none added or dropped after seeing results (anti-harvesting).
Held-out probe sets per the n=1 runs; >= 20 records per headline cell (achieved N
reported where a pool is smaller). Lens draws: the 3 cached draws; extraction/FV
draws: the 3 per species as specified in Axes.

## Instances — NAMED NOW, ALL reported (anti-harvesting; Ecaterina 2026-07-23)

The instance lists are FIXED and NAMED in this prereg before the run. ALL named
instances are reported in the result; heterogeneity is reported AS heterogeneity;
an off-profile instance is NEVER curated out by reclassifying it as "not a real
member of the species." No instance is added or dropped after seeing results.

- S1 (concept directions), n=5: **Paris, London, Rome, Berlin, Madrid** —
  mean-difference over each capital's answer states (the certified roster).
- S2 (function vectors), n=3: **capitalize, singular-plural, english-french** —
  the ONLY three M2-certified FVs (checked, certificates.json). No further FV is
  certified; a fresh M2 extraction is NOT run (FV draw-instability makes fresh
  extraction unreliable — VERIFIED), so S2 is n=3, not n≥4.
- S5 (steering vectors), n=4: **sentiment (pos/neg), formality (formal/informal),
  politeness (polite/rude), excitement (exciting/boring)** — mean-difference
  steering, same construction as the M5-6 sentiment vector. LOAD-BEARING (below).

## Axes (identical instruments to the n=1 runs; ≥3 lens draws; sham twins)

Per instance, all four:
1. DRAW-STABILITY — cross-draw MIN-PAIRWISE COSINE of the extracted direction over
   the 3 draws, computed by the IDENTICAL method on both machines (Ecaterina
   2026-07-23, else the extraction/metric confound re-enters inside this
   experiment): jtvec.concept_gate.min_pairwise_cosine over the 3 per-draw unit
   directions — same cosine, same 3 draws, same code path. S1/S5 (Mac): the 3
   fresh extraction draws (seeds 1/2/3). S2 (RTX): the 3 M2-certified FV DRAW
   TENSORS (draws 1/2/3) loaded from the cache, fed to the SAME
   min_pairwise_cosine — NOT M2's internal cosine metric or any other statistic.
   Method stated here once; applied verbatim in both places.
2. LENS-READOUT — decode_vector jlens/logit label-rank over the 3 cached lens
   draws (the EXP-M5-6 A1 statistic); the direction's own content as the label.
3. POTENCY — injection (sham-controlled Δ, the 1b/M5-6 machinery) + ablation
   (sham-controlled Δlogit, the 1d machinery), ≥3 draws, sham twins.
4. OUTPUT-ALIGNMENT — the logit-lens label-rank of the direction's content (low =
   output-aligned; the logit arm of axis 2, reported explicitly).

## Decision rule — profile reproduces vs heterogeneous (bars [proposed])

Each species has a reference profile; an instance MATCHES iff it satisfies the
profile on all measured axes at the same bars used in the n=1 runs:
- S1 profile = draw-stable (converges by the ladder) ∧ lens-dark (jlens > 20)
  ∧ inert (potency below bar). Reproduces iff **≥ 4/5** concepts match.
- S2 profile = draw-UNSTABLE (cross-draw cosine in the M2 0.43–0.61 range) ∧
  lens-dark ∧ potent. Reproduces iff **≥ 2/3** FVs match (n=3: only 3 FVs are
  M2-certified — the honest ceiling, not a chosen bar).
- S5 profile = output-aligned (logit label-rank low) ∧ logit-trivially readable
  (jlens ≈ logit) ∧ potent. Reproduces iff **≥ 3/4** steering vectors match.

S5 LOAD-BEARING CALL (Ecaterina 2026-07-23): S5/output-alignment IS load-bearing —
it is the paper's demonstration that for output-aligned directions the lens-
readability and the potency are ONE fact, and that is a TYPE claim, not a single
sentiment vector. So S5 is raised from ≥2/3 to **≥3/4 of 4 named instances**.
Honest caveat: 410M steering strength is behaviour-dependent. The output-alignment
+ logit-trivial-readability half of the profile reproduces by construction (any
mean-difference-toward-output direction); the POTENCY half depends on whether
410M steers that behaviour. An instance that is output-aligned + logit-trivial
but sub-potent on 410M (its potency positive control gating each) is reported as a
partial match and makes S5 heterogeneous-on-potency — reported, never curated out.

Branches, fixed before running:
- PROFILE-REPRODUCES (per species): the profile is type-general at 410M; the
  species claim upgrades from "this vector" to "this representation type".
- HETEROGENEOUS (per species): the instances do NOT cluster on the profile (e.g.
  some concept directions potent, some inert) → the profile is instance-specific,
  the n=1 result does not generalise, and the species row is reported as
  internally heterogeneous — a HONEST weakening of the taxonomy, published as such.
- Mixed / per-instance reporting always; no forced species call if instances
  straddle a bar (reported instance-by-instance).

## Instruments and controls

- Every axis carries its n=1 controls: injection mechanism (null-check +0.80
  precedent), ablation positive control (unembed-direction, per 1d), sham twins,
  decode_vector A1 controls (positive readable / negative random). require_
  controlled() per axis per instance; a failed control makes that instance's axis
  INCONCLUSIVE, never a match/mismatch.
- Identical statistics + bars to the n=1 runs (no new bars beyond the ≥N/M
  reproduction thresholds above).

## What counts as failure / honesty

- A species whose instances straddle its profile is HETEROGENEOUS — reported, not
  smoothed. This is the load-bearing honest outcome: if the profiles are n=1
  artifacts, the taxonomy's per-species claims are withdrawn to per-vector claims.
- Control failure → that instance-axis inconclusive.
- Post-hoc analyses of stored tensors labelled post-hoc forever.

## Resource estimate + machine split

- S1 (5 concepts) + S5 (2–3 steering): **the Mac** (410M, MPS fp32) — reuses the
  1b/1d/M5-6 machinery + the cached 410M lens draws. Projected ~2–3 h wall, peak
  ~2.5 GB. Under the 12 h LAW.
- S2 (3–5 FVs): **the RTX** — the M2 certified FV tensor cache is on the win32
  machine (not the Mac; the M5-6 blocker), so S2 breadth runs there (410M model +
  FV cache; draw-stability from M2, lens-readout via decode_vector, potency via
  injection/ablation). Projected ~1–2 h on the RTX. Detached + Monitor.
- Draw-stability for S2 is the SAME min_pairwise_cosine (Axes §1) applied to the
  3 cached FV draw tensors — NOT M2's internal cosine metric (the identical-method
  requirement). M2's VERIFIED "FVs draw-unstable" is the expected result,
  re-derived here under the identical statistic so S1/S2/S5 stability is one
  method. Lens-readout for the 3 FVs (E1 on record) is re-derived here for the
  full axis set under decode_vector.

## Deviations

- Section-heading conformance fix (2026-07-23, text-only; D-015 precedent).
  start_run's check_prereg_sections requires the literal headings
  `## Hypothesis`, `## Estimator plan`, `## Sample plan`, `## Resource estimate`;
  the ratified draft carried that content under `## Why` / `## Axes` /
  `## Instances` / `## Machine split + resource estimate`. Added the three
  missing headings (restating content already specified elsewhere) and renamed
  the resource heading to contain `## Resource estimate`. NO threshold, instance
  list, or decision-rule change; not an amendment (fresh experiment, budget n/a).
  Applies to the shared prereg, so it also unblocks the Mac's S1/S5 runs. Flagged
  for Ecaterina.

## Ratification

ratified: EXP-M5-8 2026-07-23 — Ecaterina, with adjustments: reproduction bars
S1 ≥4/5, S2 ≥2/3 (only 3 certified FVs), S5 ≥3/4 (S5 ruled LOAD-BEARING);
instances NAMED + anti-harvesting clause (all reported, none curated);
draw-stability by one identical min_pairwise_cosine across Mac (S1/S5) and RTX
(S2). Fresh experiment; budget does not apply. Machine split: S1/S5 Mac, S2 RTX.
