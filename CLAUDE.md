# jtvec — standing instructions (auto-loaded every session)

v2 measurement stack for one claim, currently HYPOTHESIS tier and never to
be stated as a finding: task execution and task verbalization are causally
separable in language models (HYPOTHESIS). v1 lives on in vendored form
only (VENDORING.md); its FV numbers are contaminated by draw instability.

## Read first, in this order

1. `CONSTRAINTS.md` — the constitution. LAW entries are inviolable; the
   VERIFIED/PROVISIONAL/HYPOTHESIS tags bound what may be stated as fact.
   If any instruction conflicts with a LAW, stop and flag.
2. `LABNOTES.md` (end to end) — decisions D-NNN, milestone sign-offs,
   current state. Append-only; sign-offs and `verify:` lines are
   Ecaterina's alone.
3. `VENDORING.md` — what came from v1 and the byte-identical policy:
   vendored v1 code is not rewritten, and vendored != in use.

## Build order and sign-off protocol

M0 skeleton -> M1 lens gate -> M2 FV stability gate -> M3 interventions ->
M4 confirmatory experiments (E1 decodability -> E2 dissociation -> E3 swap
-> E4 confabulation; emergence sweep LAST, only with per-checkpoint AIE and
the M2 gate passing at >= 2 checkpoints). A milestone starts only after
Ecaterina's `sign-off: M<k>` line exists in LABNOTES for the previous one.
Nothing FV-dependent runs before the M2 gate produces a converged trial
count (or a documented non-convergence). Taxonomy phase (2026-07-19,
TAXONOMY_DESIGN.md): M5 species certificates (M5_SPEC.md) -> M6 axis
battery; E4 and the emergence sweep follow the adult-model matrix.

## Hard rules beyond the LAWs (from the project brief)

- Estimate wall-clock + peak memory before any run > 10 min; anything
  projected > 12 h does not run on a laptop — flag it for the A100 and let
  Ecaterina rule.
- Report numbers only with scope: "on <model>, <config>, N=<n>, we
  observed X (sham: Y)" — never "the model can X". Use
  `jtvec.core.reporting`; the validators lint the rest.
- If a result surprises you, the next action is a raw-output replay, not
  an interpretation.
- Any deviation from a prereg or the plan requires a flagged decision that
  Ecaterina answers. Propose, never silently adopt.

## Engineering conventions

- `uv` only; deps stay pinned (torch 2.13.0, transformers 5.13.1,
  python 3.11). Submodules: `third_party/jacobian-lens` @ 581d398,
  `third_party/function_vectors` @ fb9eac7 — never advance them.
- Every scientific run goes through `jtvec.core.runctx.start_run`
  (committed prereg + clean tree enforced; configs copied; raw outputs
  retained). Commit the experiment before running it.
- `uv run pytest -q` and `uv run python -m jtvec.validators` must pass
  before every push; CI runs both.
- Long local runs: launch detached (`nohup ... &`) with a Monitor on the
  log — harness-tracked background Bash tasks get killed after ~1 h
  (learned in M1; incident in LABNOTES).
- MPS quirks the vendored code already handles: fp32 only (fp16 broken on
  MPS for this stack), `torch.linalg.pinv` on CPU, the transformers-5
  FV-hook patch in `jvec/fv.py` (needs its landing test before M2 use).

## Current state (update this line at each milestone boundary)

M0-M3 signed off; M4 partially closed (D-021, 2026-07-20): E1-E3 evidence
closed at recorded verdicts (E1 NOT-DECODABLE 3/3; E2 ONE-WAY, CLM-003
preliminary; E3 REDIRECTS-BASIS-AGNOSTIC, CLM-004 preliminary — both await
Ecaterina's verify: lines), E4 + emergence sweep deferred with entries live
(D-019/D-020 locked). Taxonomy phase (M5+) OPEN per TAXONOMY_DESIGN.md (axes
A1-A5, species S1-S5, H1-H5 in CONSTRAINTS.md) + M5_SPEC.md. Rulings
2026-07-20: D-002 repo PUBLIC (4 housekeeping commits pushed); D-022
evandez/relations vendored @1b9ec3c (third_party/relations submodule, LRE
data); D-023 pythia-1.4b pinned @fedc38a; compute — 1.4B lens gate Mac
overnight skip4-only, binding on Mac, skip-sweep not purchased; D-024 scout
tier authorized AFTER M5.0 baselines + 1.4B lens gate (single-draw,
post_hoc, scratchpad/results-scout only, banned from CLAIMS/findings).
Preregs EXP-M5-0-qualification + EXP-M5-1-concept-gate COMMITTED (thresholds
ratified; both are gates, no CLAIMS entry). Current machine: the original M1
MacBook 16GB (MPS, fp32); heavy tiers remain the win32 RTX laptop and an
A100. 1.4B lens gate RAN (scripts/m5_lens_gate.py, EXP-M5-0 rule 5):
verdict FAIL (Q1/Q3/Q4 pass; Q2 flip 0.56<0.75 though dp 0.48 strong; Q5
J-lens beats logit >=5x on only 1/4 anchors; Q6 dp IQR 0.071 from a draw-1
outlier), evidence results/m5/20260720-024819-p14b-lens-gate + raw replay in
LABNOTES. D-027 RULED (diagnostic-first): EXP-M5-0b (latent-vs-output probe on the
cached draws) returned GAP-RETURNS — the gate FAIL was metric+anchor
miscalibration, not J-lens/logit convergence (fresh latent operand 6.98x /
bridge 15.52x vs matched outputs ~0.9x, all random 0.0, 3 draws; capital-
recall recovered to 20.35x under max-contrast vs the gate's 1.1x). Evidence
results/m5/20260720-215157-p14b-lens-diagnostic. 2.8B NOT triggered. Since
then (2026-07-21): verify-LAW amended (by-hand burden lifted; human
confirmation stays the gate); CLM-003 + CLM-004 VERIFIED (Ecaterina; verify
lines in LABNOTES); EXP-M5-0-amendment-Q5 + EXP-M5-0c RATIFIED. M5.0
qualification RAN (results/m5/20260721-033359-qualification): 410M COMPLETE
(S1 5/8; FV 2/3 — capitalize 0.93 / singplur 0.86 / eng-fr 0.47; LRE 4/12 ->
S3 no; bind2 0.617 -> S4 no) and 1.4B Pass A done (S1 5/8; bind2 0.65 -> S4
no). D-031 RULED: 1.4B Pass B (FV+LRE) killed on the Mac (swap-thrash 21.6/22.5
GB, 8-10+ h ETA) and MOVED to the RTX — orchestrator now substrate-selectable,
configs/m5_0_qual_pythia1p4b_cuda.yaml added; RTX runs
`scripts/m5_0_qualification.py pythia-1.4b=configs/m5_0_qual_pythia1p4b_cuda.yaml`
for one clean 1.4B run dir. FLAG open: binding (S4) clears 0.70 on NEITHER model
at N=60 -> S4/H5 needs a stronger substrate or a battery re-spec (Ecaterina to
rule). Next reading-track: EXP-M5-1 (S1 concept-direction stability gate; code
authorized). RTX queue: EXP-M5-0c swap-decomposition + the 1.4B qualification
Pass B. Standing: D-002 accepted knowingly.
