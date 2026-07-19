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

M0-M3 complete and signed off; M4 E1-E3 evidence closed at their recorded
verdicts (E1 NOT-DECODABLE 3/3 tasks; E2 ONE-WAY, CLM-003 preliminary; E3
REDIRECTS-BASIS-AGNOSTIC, CLM-004 preliminary — both claims await
Ecaterina's verify: lines); E4 + the emergence sweep are deferred by the
2026-07-19 taxonomy pivot (validated sweep pipeline and D-019/D-020 stay
locked, untouched). Taxonomy phase (M5+) is OPEN per TAXONOMY_DESIGN.md
(axes A1-A5, species S1-S5, hypotheses H1-H5 in CONSTRAINTS.md) and
M5_SPEC.md; no `sign-off: M4` line exists yet (D-021 proposed, awaiting
ruling). Current machine: the original M1 MacBook 16GB (MPS, fp32); heavy
tiers remain the win32 RTX laptop and an A100. Next: M5.0 substrate
qualification — preregs EXP-M5-0-qualification / EXP-M5-1-concept-gate
drafted UNCOMMITTED awaiting ratification (the prereg act is the commit).
Open rulings: D-021 (M4 disposition), D-022 (LRE relations vendoring),
D-023 (1.4B revision pin), compute placement for the 1.4B lens gate and
binding battery, D-002 (public repo, standing).
