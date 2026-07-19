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
count (or a documented non-convergence).

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

M0-M3 complete and signed off (M1: lens gate PASS R1-R6, bit-for-bit v1
reproduction; M2: FV stability gate PASS, converged_at=25 on all three
tasks, certificates in `results/m2/20260718-114950-fv-stability-gate`;
M3: intervention-instrument gate, D-011/D-012 — fv-direction-ablation,
jspace-ablation, fv-swap GATED, report-probe gated for capitalize and
english-french, evidence `results/m3/20260718-174954-instrument-gate`).
Platform is a win32 laptop with an RTX 2000 Ada (D-008/D-009). M4 in
progress (E1 decodability -> E2 dissociation -> E3 swap -> E4
confabulation):
- E1 (CLM-001, prereg D-014): NOT-DECODABLE 3/3 tasks — counter-evidence
  to the FV-label HYPOTHESIS; stays hypothesis. Readout variance is
  lens-draw-dominated (marginalize over lens draws); FV readout tops are
  task OUTPUT items not labels (post-hoc). D-015: skip_first is a
  calibration-position parameter, not a layer restriction (text-only
  prereg fix).
- E2 report-gate (D-016 Path A): the D-013-withdrawn singular-plural
  report probe was rebuilt as report-score-prior-corrected@singular-plural
  (prior-corrected log-prob score) and GATED under P3 only — weak signal
  (~+0.22 log-prob mapping margin); P2 showed the D-013 input-leakage,
  caught by the shuffled-arm negative control. landmark-country deferred
  (uncertified in v2).
- E2 dissociation (CLM-002, prereg D-017): verdict ONE-WAY
  (`results/m4/20260719-142007-e2-dissociation`). fv-direction ablation
  is execution-specific and transfers across 3 FV draws (exec
  0.92->0.00, report unhurt/raised); jspace ablation is NOT
  report-specific. CLM-002 stays hypothesis; CLM-003 opened (preliminary)
  for the robust one-way fv execution-specificity.
- E3 swap (CLM-004, prereg D-018): verdict REDIRECTS-BASIS-AGNOSTIC
  (`results/m4/20260719-151956-e3-swap`). Swapping the certified FV_A
  component onto FV_B redirects capitalize->singular-plural (task-B rate
  0.00->0.93, all 3 FV draws; random ~0; task A suppressed to 0), in the
  raw residual basis (lens-direct gap 0.133 < 0.15). CLM-004 ->
  preliminary. With E2 this is the causal complement: ablation removes
  execution, swap redirects it; E1 found the same FV is not lens-readable
  as a label.
- Emergence sweep (the main-track bet; E4 confabulation DEFERRED by this
  pivot): targets the developmental HYPOTHESIS (execution matures early
  while portable stability-gated FVs emerge late), multi-scale on Pythia
  to also kill n=1. Built (jtvec/emergence.py + scripts/m4_emergence_sweep.py,
  reusing the M2 gate at rungs {25,50} per checkpoint) and VALIDATED
  bit-for-bit on a laptop dry-run vs M2 (converged_at=25) and E1 (outvocab
  1302) — after the dry-run caught + fixed a tokenizer-BOS mutation bug
  (build the lens AFTER extraction). Ruled: 3 scales {410M,1B,2.8B};
  D-019 constants ratified; batched-AIE optimization authorized (D-020).
Pending, in order: (1) implement + re-validate the batched-AIE (D-020):
fix the abandoned batched_input addmm->matmul in vendored
replace_activation_w_avg + batch activation_replacement over heads
(~16x, 384->24 forwards/trial), gated by a bit-for-bit re-validation vs
the unbatched AIE; (2) Ecaterina secures the A100 (~4 A100-h/3 scales
batched); (3) commit prereg EXP-M4-emergence + CLM-005 (drafted, in
scratchpad) + 1B/2.8B sibling configs, then launch the sweep; (4) E4;
then writeup. Open notes: v1 cross-code-path FV instability vs
same-pipeline stability (M2 run-1); singular-plural label readable
without task examples (M3 run-3); fv-ablation-raises-report-readout (E2
run) — research questions, not asserted.
