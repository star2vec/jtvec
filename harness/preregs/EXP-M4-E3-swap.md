# Preregistration — EXP-M4-E3-swap: cross-basis FV swap (capitalize → singular-plural)

- experiment-id: EXP-M4-E3-swap
- claim: CLM-004
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m4_e3_swap_pythia410m.yaml
- author + date: Claude (proposal), 2026-07-19; scope ruled by Ecaterina
  2026-07-19 (gated pair only; translation pair deferred); thresholds
  ratified by Ecaterina 2026-07-19 (D-018, session Q&A). Committing this
  file is the prereg act.

## Hypothesis

HYPOTHESIS-adjacent (CONSTRAINTS.md): whether the FV carries transferable, causal task
identity — a component of the execution-vs-verbalization separability
HYPOTHESIS and of Nadaf's steerable-but-not-logit-decodable pattern
(HYPOTHESIS). E3 tests, on the M3-gated capitalize→singular-plural pair,
whether moving the FV_A component onto FV_B redirects the model to task B,
and whether that redirection is specific to the J-lens basis. Scope: the
gated pair only; the translation pair (english-french↔english-spanish,
needs english-spanish certified) is deferred. Nothing is asserted; the run
decides.

## Decision rule

On capitalize (task A) 10-shot prompts + a query valid under both tasks,
edit the final position of band layers 4–16 (norm-preserving) under four
conditions (vendored make_swap_hooks): none; lens_swap (move the h-component
from J·fv_A onto J·fv_B, written back via the truncated pinv, rcond 0.05);
direct_swap (the same move in raw residual space); random_target (lens_swap
toward a norm-matched random vector). Task-B rate = fraction of N=30 shared
queries whose top-1 is the task-B (plural) answer, exact-match case-sensitive
(D-012 answer_first_tokens). Cross-draw: draw k uses fv_A_k and fv_B_k
(k=1,2,3), so each condition's B-rate is a 3-draw DrawSet (median/IQR). One
clean (none) run shared. Context sets sampled once (rng 6363) and reused.

Constants (D-018): let best-swap gain_k = max(lens_b_k, direct_b_k) −
none_b.
- REDIRECTS iff median_k(best gain) ≥ min_b_gain = 0.20 AND median_k(random_b
  − none_b) ≤ max_random_elevation = 0.05 (one-sided: a random swap that
  destroys computation is not redirection — D-012 lesson).
- Cross-draw transfer flag = every draw's best gain ≥ 0.20.
- J-specificity: J-specific iff median(lens_b) − median(direct_b) ≥
  min_j_specificity = 0.15; else basis-agnostic (the raw residual direction
  carries identity as well as the lens basis).
Verdict ∈ {REDIRECTS-J-SPECIFIC, REDIRECTS-BASIS-AGNOSTIC, NO-REDIRECTION}.
CLM-004 moves hypothesis → preliminary iff REDIRECTS (either flavor) with the
transfer flag true; the verdict table is recorded regardless.

[Thresholds await Ecaterina's ratification; constants in jtvec/e3_swap.py and
scripts/m4_e3_swap.py match this file.]

## What counts as failure

- NO-REDIRECTION (best-swap gain < 0.20 or not separable from the random
  target): evidence the FV does not carry transferable task identity for
  this pair at this model/config; CLM-004 stays hypothesis, table recorded.
- Cross-draw transfer failing (median redirects, a draw disagrees): reported
  as non-transferring; not a clean redirection.
- fv-swap instrument not gated at run time: abort at require_controlled.
- Post-hoc analyses of this run's records are labeled post-hoc forever.

## Estimator plan

- FVs: certified fv_todd@capitalize and fv_todd@singular-plural, 3 draws each
  (M2, full-trial), StabilityGatedFV; draw k pairs fv_A_k with fv_B_k. Lens:
  M1 draw 0 (cache/m3, M3-verified), identity-checked at run start; the lens
  is the fixed coordinate system for lens_swap (direct_swap uses none). Each
  condition's B-rate is a 3-draw DrawSet (median/IQR only). N=30 shared
  queries per condition tightens each draw's rate.

## Instruments

fv-swap — M3-gated (results/m3/20260718-174954-instrument-gate), asserted via
require_controlled before measuring. The swap direction (capitalize→
singular-plural) is the one M3 controlled. Execution/B-rate scoring is
deterministic exact-match, not an instrument.

## Interventions and shams

The swap IS the intervention; its control arm is random_target (a
norm-matched random FV_B), reported next to lens_swap/direct_swap in the same
table (sham LAW). random_target's random-direction seed varies per draw so it
is itself a 3-draw DrawSet. Norm-preserving throughout; truncated pinv
rcond 0.05 (VERIFIED: full pinv destroys computation).

## Sample plan

- Pair: capitalize→singular-plural. N=30 queries valid under both tasks
  (distinct A/B targets); task-A 10-shot contexts from capitalize train,
  sampled once and reused across conditions/draws.
- Conditions: none (shared) + {lens_swap, direct_swap, random_target} × 3 FV
  draws. Raw cells (≥20 records each): swap_none, swap_{lens,direct,random}_draw{1,2,3}.

## Resource estimate

~300 forward passes (10 condition-runs × 30 queries), each a single forward
with a simple final-position swap hook (~0.15 s); lens is a cache hit; model
load ~1 min. Projected ~2–5 min wall-clock, bounded 15 min. Peak RSS ≤ ~4 GB.
Under the 10-min flag and far under the 12 h LAW; the run still launches only
after this file's commit and Ecaterina's threshold ratification.

## Deviations

(none at commit time)
