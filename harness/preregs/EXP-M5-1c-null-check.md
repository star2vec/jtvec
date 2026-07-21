# Preregistration — EXP-M5-1c: recalibrated-instrument null-check

- experiment-id: EXP-M5-1c-NULLCHECK
- status: **RATIFIED** by Ecaterina 2026-07-22 (session instruction "i ratify
  it", following the governance ruling that ordered it, item 1). Committing this
  file is the prereg act; this gate precedes all other Mac runs. Pass conditions
  are fixed below BEFORE the run. One text-only conformance fix folded in after
  ratification (I2 metric aligned to the ratified EXP-M5-1b — Δp not Δ log p; see
  I2 conformance note); no null construction or threshold changed.
- claim: none (a governance gate on the instruments themselves; produces no
  species certificate and asserts nothing about residency).
- models: EleutherAI/pythia-410m@9879c9b — the substrate where we are most
  confident and where the original bars were derived (M1-certified lens, cached
  draws 0/1/2). 410M is the "nothing-to-find" control substrate for both nulls.
- config: configs/m5_1c_nullcheck_pythia410m.yaml (standard Config schema, band
  [4,16]; drafted with this file).
- author + date: Claude (proposal), 2026-07-21, per the governance ruling.

## Why (the risk this gates)

Three consecutive gate failures were each resolved by diagnose-miscalibration →
amend, and each recalibration recovered signal (M1/D-027, 1.4B-Q2/0c-H-CONFOUND,
M5.1/D-033). The live risk (recorded, LABNOTES governance entry): the amendment
process may be harvesting researcher degrees of freedom — the max-over-layers
selection in the max-contrast statistic, and the extend-the-ladder move in the
concept readout, are both selection operators that CAN manufacture apparent
signal from noise. This experiment tests each recalibrated instrument on a
known-null: it must report NULL where there is nothing to find. It does not
re-open the amendments' scientific content; it checks that they do not fire on
noise.

## Instruments under the null-check (two, reported separately)

### I1 — amended-Q5 max-contrast statistic (EXP-M5-0-amendment-Q5)

Statistic (unchanged, applied identically to every arm): ratio_A = max over
source layers L with HMR_A(L) ≤ 5.0 of logit_HMR(L)/HMR_A(L); the J-lens
advantage counts iff median-over-draws ratio_jlens ≥ 5.0 AND every random arm's
ratio < 5.0.

Null construction: a SCRAMBLED-LABEL latent probe on 410M — probe the
intermediate position for a target token that is PERMUTED across prompts (each
prompt's probed target is another prompt's operand, so the probed content has no
genuine alignment with the residual being read). Same lens draws (cached 0/1/2),
same candidate layer set S, same cap 5.0, same statistic on jlens / logit /
random arms. There is no latent content to decode, so a well-behaved statistic
must find no advantage.

Pass condition (fixed before the run): median-over-draws ratio_jlens < 5.0 on
the scrambled-label probe (the recalibrated instrument reports NULL on null).
Pipeline-sanity check (NOT the pass condition, just proof the pipeline can fire):
on the genuine capital-operand latent anchor the same statistic clears 5.0
(EXP-M5-0b saw ~6.98x on fresh operands / 20.35x on capital-recall). If the
scrambled-label ratio_jlens ≥ 5.0, the max-over-layers selection is inflating
noise → the Q5 amendment is WITHDRAWN pending re-spec.

### I2 — D-033 extended-ladder concept readout (EXP-M5-1b)

Statistics (the recalibrations under test — matched to the ratified EXP-M5-1b):
(i) the convergence-ladder min pairwise cosine over the extended rungs
{8,16,32,64,128,256}; (ii) the RESOLVABLE sham-controlled Δp(label) potency
readout — the injection-strength sweep alpha ∈ {1,2,4,8} at N=200 carriers, in
PROBABILITY space (NOT Δ log p).

Conformance note (2026-07-21, text-only, D-015 precedent): the draft first read
"Δ log p" for I2; that contradicted D-033 condition (b), which forbids rescuing
the p-floor by a log/odds reinterpretation and mandates a resolvable Δp (raise
trials / injection until resolvable, or declare unmeasurable). I2 is corrected to
test the SAME resolvable-Δp readout the ratified EXP-M5-1b uses — the null-check
must validate the instrument 1b actually runs. No null construction or pass
threshold changed beyond the metric alignment.

Null construction: SCRAMBLED-LABEL concept directions on 410M — each extraction
context is assigned a label drawn from a fixed random permutation of the roster
capitals, INDEPENDENT of the context's true answer. The mean-difference is then
taken over random groupings; there is no concept to extract. Extended ladder,
alpha sweep, N=200, 3 draws (seeds 1/2/3), the same extractor and injection as
EXP-M5-1b.

Pass condition (fixed before the run), BOTH must hold:
- Convergence null: min pairwise cosine of the scrambled-label direction stays
  BELOW the 0.95 certificate bar at every rung including the ceiling 256 (no
  false convergence — the ladder-extension move does not manufacture a witness
  on noise).
- Potency null: sham-controlled Δp(scrambled label) median within the
  quantization-aware band max(0.005, 1/N) at EVERY alpha (the recalibrated
  readout does not manufacture steering on a random grouping, at any injection
  strength).

If the scrambled-label direction converges (cosine ≥ 0.95 with a witness) OR
shows potency outside the band at any alpha, the D-033 recalibration is WITHDRAWN
and EXP-M5-1b does not run.

## Decision rule

Report per instrument. The null-check PASSES iff BOTH I1 and I2 report null on
their nulls, per the fixed conditions above. On a per-instrument failure, that
instrument's amendment is withdrawn pending re-spec (independently — one may
pass while the other fails). No amendment ratifiable downstream of a failed
null-check (CONSTRAINTS LAW).

## Instruments and controls

- I1 already carries its own random-matrix arm; the scrambled-label probe is the
  added, stronger null (random TARGET, not just random projection matrix).
- I2's sham twins are the potency null; the scrambled LABEL is the convergence
  null. Both reported.
- require_controlled() semantics: a null-check that cannot even fire its
  pipeline-sanity positive (I1 capital-operand ≥ 5.0; I2 genuine-label direction
  converging on the M5.1 evidence) is itself void — a broken null-check, not a
  verdict.

## What counts as failure

- Signal-on-null (either instrument) → that recalibration withdrawn (the point
  of the experiment; a recorded outcome, not an error).
- A pipeline that cannot reproduce the genuine positive → the null-check
  apparatus is broken; investigated, not interpreted.
- Post-hoc analyses of stored tensors labeled post-hoc forever.

## Estimator plan

Forward-only, both instruments, on cached 410M lens draws (I1) / fresh residual
extraction (I2). 3 draws, seeds {1,2,3}; only the scramble permutation +
context-resampling streams vary. Median/IQR over draws (DrawSet). The scramble
permutations are fixed per draw by seed and recorded.

## Sample plan

- I1: the latent-anchor probe set (capital-operand + fresh operand/bridge) with
  permuted targets; N per anchor recorded (≥ the EXP-M5-0b N: capital-operand
  33, fresh1hop-operand 28).
- I2: the 8-capital roster with permuted labels; N_eval ≥ 40; ladder to 256.
- Raw per-item records per cell ≥ 20; scramble permutation logged per draw.

## Resource estimate (Mac tier, 410M, MPS fp32)

I1 evals on cached lenses: ~10–15 min (EXP-M5-0b-scale). I2 extended-ladder
extraction + readout: ~30–50 min (EXP-M5-1b-scale). Total ~1 h wall, peak RSS
~2 GB — under the 12 h LAW, Mac-eligible, no swap. Detached + Monitor.

## Deviations

(none yet)

## Ratification

ratified: EXP-M5-1c-NULLCHECK 2026-07-22 — Ecaterina (session instruction "i
ratify it"). This gate precedes all other Mac runs; EXP-M5-1b stays HELD until
this passes, and no fourth instrument amendment is ratifiable downstream of a
failed null-check (CONSTRAINTS LAW).
