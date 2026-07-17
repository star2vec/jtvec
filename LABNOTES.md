# LABNOTES

Append-only lab notebook. Conventions the validators parse:

- Human verification lines (gate to `verified` in CLAIMS.md):
  `verify: CLM-NNN raw-read: <n> re-derived: yes verified-by: Ecaterina date: YYYY-MM-DD`
- Milestone sign-offs: a `sign-off: M<k>` line by Ecaterina. The next
  milestone does not start before it exists.
- Flagged decisions get ids `D-NNN` with who ruled and when.

---

## 2026-07-17 — M0: repo skeleton (Claude)

Goal (project brief): CONSTRAINTS.md, CLAIMS.md (schema only), LABNOTES.md,
prereg template, results schema, CI running tests; every LAW mapped to a
mechanical enforcement point. The claim under test in v2 stays at
HYPOTHESIS tier (task execution vs. task verbalization, causally separable)
and is not restated as a finding anywhere in this repo.

Decisions:

- D-001 (ruled by Ecaterina, 2026-07-17): v1 code is vendored from commit
  `3bb6d2a` of `~/Developer/jvec-outdated` (which includes
  `third_party/jacobian-lens @ 581d398`, the commit cited by the v1 phase-1
  report). The untracked `scripts/15_fv_stability.py` found in the v1 tree
  is design input for M2 only — it is unvalidated and is not vendored.
- D-002 (setup fact): remote is github.com/star2vec/jtvec (**public** — flag
  to Ecaterina), CI is GitHub Actions: pytest + `python -m jtvec.validators`
  on every push. CI covers unit tests and ledger checks only; model runs
  (gates, experiments) execute locally on the M1 or on the A100 per the
  resource rule, never in CI.
- D-003 (setup fact): identical pinned versions everywhere
  (torch 2.13.0, transformers 5.13.1, python 3.11); Linux CI resolves torch
  from the CPU wheel index, macOS uses the default (MPS) wheels.

LAW → mechanical enforcement point (M0 definition of done):

1. Min 3 draws, median/IQR — `jtvec/core/draws.py` (`DrawSet` raises below
   3 draws or on duplicate seeds; median/IQR are the only summaries);
   `tests/test_draws.py`.
2. Stability gate before use — `jtvec/core/gate.py` (`CertifiedArtifact`
   needs a `GateCertificate`; certificates need an existing evidence run,
   a converged sample size, >= 3 draws); `tests/test_gate.py`. The FV type
   in M2 subclasses `CertifiedArtifact`.
3. Positive + negative control per instrument —
   `jtvec/core/instruments.py` (`require_controlled` called by every eval
   runner; withdrawn instruments hard-banned by name); 
   `tests/test_instruments.py`.
4. Sham twin per intervention — `jtvec/core/intervention.py`
   (`InterventionResult.sham` is non-optional and matched on layers/
   positions/count/norm; the only renderer emits both in one row);
   `tests/test_intervention.py`.
5. Prereg committed before first run; post-hoc labeled forever —
   `jtvec/core/runctx.py` (`start_run` refuses uncommitted or incomplete
   preregs; `post_hoc` stamped into run.json); template field: every
   section of `harness/prereg_template.md`; `tests/test_runctx.py`.
6. Human verification gate — `jtvec/validators/claims.py` blocks
   `verified` status without Ecaterina's `verify:` line in this file and
   >= 20 raw completions per cell; `tests/test_validators.py`. The reading
   itself is human by design; the machine only blocks promotion.
7. One commit per experiment; raw outputs retained; configs copied —
   `start_run` refuses dirty trees (forcing commit-then-run, so the
   recorded hash identifies the experiment commit), copies the config
   unconditionally, and is the only writer of results dirs;
   `jtvec/validators/results_dirs.py` re-checks retention on CI;
   `tests/test_runctx.py`, `tests/test_validators.py`.
8. Language discipline — hard half: `jtvec/core/reporting.py` (scope
   arguments mandatory; intervention strings carry their sham). Lint half:
   `jtvec/validators/language.py` (banned verb stems in prose need a
   `[VERIFIED: ...]` citation or an explicit waiver). Honest limitation:
   the lint is mechanical, not semantic; residual judgment sits with the
   gate in LAW 6.
9. Claims ledger — `CLAIMS.md` machine-readable schema +
   `jtvec/validators/claims.py` (status enum, evidence commit, results-dir
   checks); DRAFT-cites-only-verified arrives as a validator extension
   when DRAFT.md first exists (flagged now, tracked for M4).

Tier discipline (beyond the LAWs): `jtvec/validators/hypotheses.py` — key
phrases of the seven HYPOTHESIS entries may not appear untagged in prose or
source. CONSTRAINTS.md itself is exempt as the defining ledger.

Foreseeable blockers carried forward from planning: M1 tolerance spec
(proposed in the M1 prereg), M2 wall-clock estimate before launch,
HF availability of pythia-410m@9879c9b + pile-10k (cache early),
public-repo visibility (Ecaterina to confirm).

- sign-off: M0 — Ecaterina, 2026-07-18, via session instruction ("go ahead
  with M1"). Recorded by Claude.

---

## 2026-07-18 — M1: lens port + gate on Pythia-410M (Claude)

Vendored the v1 lens pipeline per D-001 — byte-identical from
`jvec-outdated @ 3bb6d2a` via `git show` <!-- lint-ok: git command name -->,
verified by diff; inventory in
VENDORING.md. v1's model-free unit tests (test_skeleton, test_evals) run in
CI alongside the M0 enforcement tests (51 total).

Decisions:

- D-004 (vendoring mechanics): `third_party/jacobian-lens` is a git
  submodule pinned at `581d398` (github.com/anthropics/jacobian-lens, the
  exact commit in v1's lens manifests) rather than a file copy, because
  `jvec.utils.jlens_commit()` reads the checkout and the hash is a
  lens-identity key in every manifest. CI checks out submodules.
- D-005 (proposed, awaiting Ecaterina): M1 tolerances are preregistered in
  `harness/preregs/EXP-M1-lens-gate.md` as rules R1-R6 (9/9 gate PASS;
  swap dp within ±0.05 of v1; sham |dp| <= 0.03; flip rate >= 75%;
  capital-recall band contrast within x2 with >= 5x logit separation; exact
  calibration-hash match; baselines within ±3 pp; draws at seeds 1,2 also
  PASS). Proposed, not silently adopted: the M1 verdict is re-evaluated if
  Ecaterina amends any tolerance at sign-off.
- D-006: task baselines are computed once (draw 0) and shared across draws —
  greedy argmax scoring has no RNG; the lens-draw LAW applies to the lens,
  not to the deterministic baseline stage. Stated in the prereg.

Plan: 3 lens draws (seeds 0/1/2, re-sampled calibration prompts, separate
caches), vendored scripts 01-04 per draw, orchestrated by
`scripts/m1_gate.py` behind `start_run` (prereg + clean-tree enforced).
Resource estimate (prereg): ~2-3.5 h local, peak ~2.6-4 GB — under the 12 h
ceiling; script 01's timing probe re-checks before fitting.

- sign-off: M1 — pending (Ecaterina)
