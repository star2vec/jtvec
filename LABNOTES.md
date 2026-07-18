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

### 2026-07-18 — M1 run complete: PASS on R1-R6

Run: `results/m1/20260718-010559-lens-gate` (prereg + experiment commit
`0065a2a`). Verdict PASS on all six preregistered rules.

- Draw 0 (seed 0) reproduced v1 bit-for-bit: dp(swap_answer) +0.6046 vs
  sham +0.0086, flip 87.5%, capital-recall band contrast 2.49 vs 61.48 at
  L16, calibration sha256 identical to the v1 manifest, all 8 baselines
  equal to the v1 table, gate 9/9.
- Draws 1-2 (fresh calibration prompts): gate PASS each. On
  EleutherAI/pythia-410m@9879c9b, skip4_n10, N=16: dp(swap_answer)
  median +0.6046, IQR 0.025, n_draws=3 (sham: median +0.0086, IQR 0.005).
  Band-min J-lens HMR medians 2.57/2.64/1.32/2.69 (IQR <= 0.47) for
  capital-operand/capital-recall/opposites/word-pairs. The skip4/n10
  defaults are hereby re-derived with draw evidence, not assumed; they
  stay PROVISIONAL pending a second model.
- Instrument control record established for the J-lens readout: positive
  control = 4 known-signal tasks (criterion A/B), negative controls =
  10-seed random-matrix arm + random-direction swap arm. This is the
  ControlRecord pair required by the instruments LAW for M4 use.
- Incident: the first launch was killed externally after ~57 min (harness
  background-task stop; three fits + baselines already done, no scientific
  output lost — fits are manifest-verified in cache). Partial stage records
  preserved under `cache/interrupted-run-1/`; relaunched detached, fits
  loaded as cache hits. No prereg deviation: same configs, same pipeline.
- Wall-clock ~70 min total (3 x ~940 s fits + evals); peak RSS 2.85 GB —
  within the prereg estimate.

- sign-off: M1 — Ecaterina, 2026-07-18, via session instruction ("run m2"),
  ratifying the D-005 tolerances per the note above. Recorded by Claude.

---

## 2026-07-18 — M1->M2 boundary: repo made self-contained (Claude)

- D-007 (ruled by Ecaterina, 2026-07-18: project may move to another
  laptop): all remaining v1 sources vendored byte-identical from `3bb6d2a`
  (fv wrapper, decompose, fv evals, scripts 05-14, phase-2 configs);
  `third_party/function_vectors` added as a submodule pinned at `fb9eac7`
  (clean public checkout). The v1 checkout is no longer required on any
  machine. Vendored != in use: build-order rule unchanged; nothing
  FV-dependent runs before the M2 gate exists. v1's untracked
  `15_fv_stability.py` preserved verbatim under `design_input/` as M2
  design reference per D-001 (not code).
- M2 compute decision is OPEN (flagged per the >12h rule): full ladder
  3 draws x 200 AIE trials x 3 tasks projects ~26-28 h on this MacBook
  (v1 measured 36/38/77 s/trial for capitalize/english-french/
  singular-plural at 25 trials, cost linear in trials; per-trial AIE
  storage confirmed, so lower rungs derive from stored prefixes at no
  extra compute). Options put to Ecaterina: A100 (~3-4 h), local with
  explicit 12h-rule waiver, or trimmed ladder. Awaiting her ruling; M2
  build proceeds, no long run launches until then.

---

## 2026-07-18 — M1->M2: new machine (win32), platform compat D-008 (Claude)

Project moved to the new laptop per D-007. First non-POSIX machine in the
project: Windows 11, NVIDIA RTX 2000 Ada Laptop GPU (8 GB VRAM, driver
595.95), uv 0.11.27, CPython 3.11.15. Setup per the session brief:
submodules initialized at the pinned SHAs (581d398, fb9eac7), `uv sync`
clean. Both verification gates then failed for platform reasons; per
instruction the failures were reported before any fix, and Ecaterina ruled
on each part (session Q&A):

- F1: `import resource` in vendored `jvec/utils.py` is POSIX-only; on
  win32 every `jvec` import died (pytest: 2 collection errors, 0 tests
  collected). F2: eight `read_text()`/`open()` sites in v2-original
  `jtvec/` lacked `encoding="utf-8"`; the win32 locale default (cp1252)
  crashed `check_language` on a UTF-8 curly quote and mis-decodes
  non-ASCII prose where it does not crash. F3: the locked win32 torch
  resolved to `2.13.0+cpu` (default PyPI wheel); CUDA unavailable — win32
  was outside D-003's platform matrix.
- D-008 (ruled by Ecaterina, 2026-07-18): (a) first post-vendor
  modification of a vendored file — try/except ImportError guard plus a
  psapi peak-working-set branch in `jvec/utils.py:peak_rss_gb`, exact
  deviation recorded in VENDORING.md, guard test `tests/test_platform.py`;
  the POSIX code path is unchanged. (b) `encoding="utf-8"` at the eight
  jtvec sites (no behavior change on macOS/Linux/CI, where UTF-8 is
  already the locale default). (c) torch stays pinned at 2.13.0; pyproject
  gains a win32-marker cu130 index (symmetric with D-003's linux-cpu
  index); macOS/Linux resolution unchanged in the regenerated lock.
- Gates after D-008 on this machine: pytest 57 passed (55 prior + 2 guard
  tests), validators 3/3 PASS; `torch.cuda.is_available()` True
  (torch 2.13.0+cu130).

M2 build proceeds next (study code + unit tests, incl. the FV-injection
landing test per CONSTRAINTS). The compute ruling for the full ladder is
still OPEN; a <=10-min timing probe on this machine comes first.

---

## 2026-07-18 — M2: build done, probe done, compute ruled D-009 (Claude)

M2 study code committed at 60b2728: jtvec/fv_stability.py (rung derivation
from stored per-trial AIE tensors, cross-draw agreement stats, witnessed
convergence rule, sham twins, per-task certificate payloads,
StabilityGatedFV), scripts/m2_gate.py (orchestrator behind start_run:
3 tasks x 3 draws, seeds 1/2/3 vary only the extraction stream, fixed eval
contexts, sham arm per (task, draw, rung), in-run instrument controls gate
the verdicts), scripts/m2_probe.py (budget-capped timing probe), and the
landing test required by the CONSTRAINTS VERIFIED entry — the patched
transformers-5 hook lands, unit-tested end to end on a tiny random
GPT-NeoX through the real function_vector_intervention path (zero-vector
identity, logits move, edit-layer delta equals the injected vector).
Gates at commit: pytest 80 passed, validators 3/3 PASS.

Probe (2026-07-18, 81 s GPU + 222 s CPU, inside the 10-min cap), AIE
s/trial for capitalize / singular-plural / english-french:

- win32 GPU (RTX 2000 Ada, D-008 stack): 8.77 / 8.97 / 9.09 — vs the M1
  MacBook's 36 / 77 / 38 (4.1x / 8.6x / 4.2x). Full ladder ~5.5 h
  (extraction 3x5,410 s + eval grid ~1,122 s, x1.15 slack; eval grid
  corrected for per-task test sizes 170/43/987). Peak RSS 3.35 GB at probe
  scale; VRAM 1.8/8.2 GB.
- CPU reference: 69.8 / 72.2 -> ~42 h ladder, out (over the 12 h LAW).
- Probe also surfaced: singular-plural is small — N_test=43 (2.3 pp gain
  granularity) and n_correct_valid=17; the min_correct_valid=50 floor of
  scripts/05 was never applied to the D-007 ladder set.

- D-009 (ruled by Ecaterina, 2026-07-18, session Q&A): (a) the full ladder
  runs on this laptop's GPU (~5.5 h, under the 12 h LAW, no waiver); the
  A100 stays the escalation path if the ladder must extend (>= 400
  trials). (b) EXP-M2 thresholds ratified as drafted (min pairwise cosine
  >= 0.95 AND gain IQR <= 0.05 at T and every larger rung; largest rung
  alone is not convergence; sham |median| <= 0.02; ICL-vs-0-shot
  separation >= 0.10). (c) singular-plural stays, kept-and-flagged in
  prereg and report. (d) commits pushed to origin (public repo per D-002).

Launch: prereg harness/preregs/EXP-M2-fv-stability.md committed in this
commit (the prereg act); run started detached with a log monitor (M1
incident precedent). ETA ~5.5-7 h from launch.

---

## 2026-07-18 — M2 run 1: preregistered control gate fired; D-010 (Claude)

Run results/m2/20260718-051327-fv-stability-gate (prereg e67310a) executed
~5.4 h on the win32 GPU (D-009), probe-accurate throughout (28.4–29.5 min
per extraction, 9 extractions). All 12 (task x rung) agreement cells pass
the ratified convergence rule — min pairwise cosine .959–.997, gain IQR
.000–.023, at every rung including T=25, on all three tasks. The run then
terminated at the instruments gate, exactly as preregistered: verdicts
voided, no certificates, no report.md.

- Control reconstruction from the retained raw cells (the process died
  before writing aggregates; every number here re-derived from
  raw_completions/*.jsonl): positive control PASS on all tasks
  (ICL-vs-0-shot separation +0.9235 / +0.8372 / +0.4671 vs bound 0.10);
  negative control FAIL at exactly one cell — singular-plural T=50, sham
  gains [-0.0233, -0.0233, 0.0], median -0.0233 vs bound 0.02. One flipped
  item out of N_test=43 equals 0.0233: the flat bound sits below the
  readout's own quantum at this N, so any single flip breaches it. Every
  other sham median is <= 0.006 in magnitude.
- Raw-output replay (surprise rule; read-only over cache/m2 and the run
  dir): the three draws differ genuinely and substantially at trial level
  — per-trial indirect_effect tensors correlate 0.31–0.52 across draws
  (max element diff ~0.5), mean_head_activations differ by up to ~0.5 —
  yet the FVs built from them agree at cosine >= 0.959 at every rung. On
  this model/config, same-pipeline re-extraction at 25 AIE trials is
  draw-stable under this protocol. This sits in tension with v1's
  cross-code-path cosines of 0.43–0.61 (VERIFIED, scoped to the
  endpoint-vs-phase-2 comparison); why the two protocols disagree is an
  open question, not resolved here, and v1's entry stands as scoped.
- D-010 (ruled by Ecaterina, 2026-07-18, session Q&A): negative-control
  bound amended post-hoc to |median sham gain| <= max(0.02, 1/N_test) per
  task — the sham may move the median by at most one readout quantum;
  N >= 170 tasks keep the 0.02 bound unchanged. Recorded in the prereg
  Deviations section. The amendment admits run 1's observed value, which
  is why it was put to Ecaterina rather than adopted silently. Run 1's raw
  evidence is committed as-is (unfinalized run.json documents the abort);
  run 2 relaunches on run 1's cached extractions (evals only, ~30 min).

---

## 2026-07-18 — M2 run 2: same cell, rounding artifact; conformance fix (Claude)

Run 2 (results/m2/20260718-113005-fv-stability-gate) cache-hit all nine
extractions, reproduced run 1's eval numbers cell-for-cell (deterministic
0-shot contexts + cached draws), and terminated at the same instruments
gate — this time by 4.4e-5. Cause, verified against the exact arithmetic:
gains were stored as round(x, 4), so the singular-plural T=50 sham median
of exactly -1/43 (= -0.0232558..., one flipped item — precisely the D-010
quantum) was stored as -0.0233, which exceeds the D-010 bound
max(0.02, 1/43) = 0.0232558... The data satisfy D-010's ruled semantics
exactly; the comparison used a decimal-rounded value. Fix: gains and sham
gains are stored unrounded (rounding remains in display formatting only) —
an implementation-fidelity fix to the already-ruled rule, no semantic
change, every reported number unchanged at display precision. Run 2's raw
cells committed as evidence like run 1's. Run 3 relaunches on the same
caches.

---

## 2026-07-18 — M2 run 3: gate PASS; certificates issued; awaiting sign-off (Claude)

Run 3 (results/m2/20260718-114950-fv-stability-gate; prereg e67310a with
the D-010 deviation; code at e3cc606) completed end to end: 9/9 extraction
cache hits from run 1, eval grid reproduced runs 1-2 cell-for-cell,
instrument controls PASS (positive: ICL-vs-0-shot separation
+0.92/+0.84/+0.47 vs bound 0.10; negative: every |median sham gain| within
max(0.02, 1/N_test) — the singular-plural T=50 cell now compared unrounded
at exactly one quantum). Wall-clock 1,072 s (evals only; extraction hours
live in run 1), peak RSS 3.34 GB, device cuda.

Verdicts under the ratified rule (min pairwise cosine >= 0.95 AND gain IQR
<= 0.05 at T and every larger rung; largest rung alone is not
convergence): capitalize converged_at=25, singular-plural converged_at=25,
english-french converged_at=25 — M2 gate PASS on all three tasks. On
EleutherAI/pythia-410m@9879c9b, m2_pythia410m.yaml, n_draws=3, induction
gain @T=25: +0.394 median (sham +0.000, N=170) / +0.209 (sham +0.000,
N=43) / +0.125 (sham +0.002, N=987). Certificates issued per task
(estimator fv_todd@<task>, converged_at=25, n_draws=3, evidence_run = the
run dir) in certificates.json. The CONSTRAINTS known-unknown — the AIE
trial count at which FV extraction converges on 410M — has a measured
answer on this model/config/task-set: 25, with witness rungs to 200.

For the record: (1) the tension with v1's cross-code-path instability
(cosines 0.43-0.61 there vs same-pipeline >= 0.959 here) is documented in
the run 1 entry and stays an open question; v1's VERIFIED entry stands as
scoped. (2) Hendel vectors: descriptive 3-draw flat cosines 0.996-0.998 at
fixed n_trials_mean=100; no certificate (prereg scope). (3) Raw evidence
for all three runs is committed; the per-draw FV caches under
cache/m2/draw*/ are the certified artifacts' backing store.

M3 does not start before Ecaterina's `sign-off: M2` line. Verification
material: report.md, stability.json, and 30 raw per-item cells in the run
3 dir, plus the runs 1-2 evidence dirs.

- sign-off: M2 — Ecaterina, 2026-07-18, via session instruction ("do m3"),
  ratifying the gate outcome including the D-010 deviation and the run-2
  conformance fix. Recorded by Claude, per the M0/M1 precedent.

---

## 2026-07-18 — M3: scope ruled (D-011), build (Claude)

Repo state at M3 start: local == origin at 6a3a00b (+ sign-off record);
tree clean; gates green.

- D-011 (ruled by Ecaterina, 2026-07-18, session Q&A): (a) M3 = the
  intervention-instrument gate, mirroring M1 (lens readout) and M2 (FV
  estimator): landing tests + positive/negative ControlRecord pairs for
  fv-direction-ablation, jspace-ablation, the forced-choice report probe,
  and the fv-swap, all on M2-certified FVs; the lens is re-materialized on
  this machine first and verified against M1 draw 0's committed manifest
  (identity keys incl. exact calibration sha256) plus a capital-recall
  band-min HMR spot-check — E1-E4 stay in M4. (b) The report-probe
  positive control is the explicit-rule context (rule sentence naming the
  label + label-shuffled pairs): a detection-ceiling control that does not
  presuppose what ICL contexts carry; negative stays the shuffled-context
  prior baseline; all three phrasings P1-P3 reported. (c) The swap
  control pair is capitalize->singular-plural (both certified; the same
  query is valid under both tasks). The v1 translation pair waits for
  M4-E3's own prereg, where certifying english-spanish belongs.
- Build: jtvec/m3_instruments.py (control rules with quantization-aware
  bounds max(base, 1/N) built in from the start — D-010 lesson; certified-
  FV loading through the manifest-checked vendored loader; lens-manifest
  verification on the vendored IDENTITY_KEYS; explicit-rule context
  builder), scripts/m3_gate.py (orchestrator behind start_run), config
  configs/m3_pythia410m.yaml (calibration/fit mirror M1 draw 0 exactly),
  tests/test_m3_instruments.py (landing/property contracts for the four
  vendored hook classes — plain-tensor AND tuple outputs, final-position-
  only, idempotence/orthogonality, norm preservation, swap component move
  — plus rule/verdict logic). Prereg drafted UNCOMMITTED
  (EXP-M3-intervention-instruments.md); committing it is the prereg act
  and waits for Ecaterina's ruling on the resource estimate
  (~10-25 min projected: lens refit 2-15 min + ~700 control forwards).
