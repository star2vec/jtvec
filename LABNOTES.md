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

### 2026-07-18 — M3 run 1 aborted on a code bug (not a gate); fixed, relaunch

Prereg committed a5b87dc; gate launched. Prerequisites PASSED and are worth
recording as the cross-machine reproduction result they are: the lens
refit on this win32/CUDA machine reproduced M1 draw 0's identity exactly
(calibration sha256 + all fit hyperparameters equal the committed
manifest), and the capital-recall functional spot-check gave band-min
J-lens HMR 2.50 at L16 vs M1's 2.49 at L16 (logit 61.5, 24.6x separation)
— the MPS->CUDA move left the lens numerically the same instrument.
fv-direction-ablation controls PASSED on both tasks (exec 0.933 -> 0.000
under FV ablation; sham 0.900 / 0.933; bound 0.05).

Then a KeyError('target') aborted the jspace stage: the swap-capitals task
uses the swap schema (keys answer/swap_to/swap_answer), not the completion
schema's target, and m3_gate scored jspace execution against item["target"].
Pure code bug, not a preregistered event: no verdict, no certificates
issued, deterministic crash. Fix: jtvec/m3_instruments.execution_answer()
resolves target|answer across schemas (raises if neither), used by the
jspace stage; pinned by tests/test_m3_instruments against the real task
files. The aborted partial dir (results/m3/20260718-140704-instrument-gate,
fv-ablation + spot-check cells + unfinalized run.json) is removed rather
than committed: unlike M2 run 1 (which aborted at a *preregistered gate* —
a scientific event worth retaining), this abort carries no scientific
content and every number in it is reproduced bit-for-bit by the relaunch
(context rng 9090 seeded once; the fix touches only a later stage). No
prereg change — the fix implements the committed jspace-on-swap-capitals
rule as written. Relaunch on the cached lens.

### 2026-07-18 — M3 run 2 complete: 1/4 instruments gated; replayed

Run results/m3/20260718-141523-instrument-gate finalized (report.md,
controls.json, run.json). Prerequisites reproduced run 1 (lens identity +
spot-check HMR 2.50@L16). Verdict: fv-direction-ablation GATED; the other
three not gated, for three distinct reasons. Raw-output replay of each
(read-only) before any interpretation, facts only:

1. fv-direction-ablation GATED: exec 0.933 -> 0.000 under FV projection,
   sham_fv 0.900/0.933 within bound 0.05, both certified tasks. Clean.
2. jspace-ablation NOT gated (positive arm): top-10-atom J-space projection
   on swap-capitals moved exec 15/16 -> 13/16 (drop 0.125 < 0.15). Replay:
   two items broke cleanly (Oslo -> " Tr", Havana -> " San"); a third
   (Tehran) changed token " Tehran" -> "Te" but still scored a hit under the
   first-token surface relaxation. The measured drop is sensitive to that
   relaxation at N=16. Real, directional, but below the preregistered bar;
   substantive, not a code error.
3. report-probe NOT gated (negative arm): explicit-rule detection 1.0 on all
   three tasks x three phrasings (positive strong). Negative fails on
   singular-plural: shuffled-context accuracy 1.0 (capitalize 0.0,
   english-french 0.25). Replay confirms the shuffled-context baseline does
   not null a task whose outputs are all one morphological class: shuffling
   plural outputs among inputs still yields (singular -> plural) pairs, so a
   "plural" readout survives. The label-vs-output-vocabulary separability
   this touches is a CONSTRAINTS HYPOTHESIS and is NOT asserted here; the
   operational fact is that the v1 shuffled baseline is not a valid null for
   morphological-output tasks.
4. fv-swap NOT gated (negative arm): positive strong — lens_swap moved the
   task-B (plural) rate 0.467 -> 0.867, direct_swap -> 0.767 (gain 0.40).
   Two artifacts drive the negative failure, both mine, not the swap:
   (a) random_target output was newline on all 30 trials (B=0) — a
   norm-matched random swap destroys computation, which CONSTRAINTS already
   records for random directions; my |random - none| <= bound criterion is
   the wrong (two-sided) test, since "random breaks the model" is expected.
   (b) the none B-rate 0.467 is inflated by a scoring collision:
   surface_token_ids case-expansion makes some clean capitalize outputs
   (e.g. " K" for "Kettle") share a first token with the plural target's
   capitalized variant (" Kettles"), so task-A outputs count as task-B hits
   for case-ambiguous words. The swap itself produces genuine plural
   outputs; the control criterion and the cross-task scoring are what fail.

Net: the gate did its job — it caught two substantive instrument weaknesses
(jspace power at the chosen config/N; the report-probe null for
morphological tasks) and two of my own design defects (swap negative-control
criterion; case-collision scoring across the swap pair) BEFORE M4 built on
any of them. The fv-swap and report-probe fixes both change preregistered
control CRITERIA, so they are deviations for Ecaterina to rule, not adopt.
Run 2 evidence committed; options put to Ecaterina next.

### 2026-07-18 — D-012: four ruled fixes, re-run (Claude)

Ecaterina ruled all four (session Q&A); applied as one deviation commit
with tests (prereg Deviations D-012 has the full spec):

1. Exact-match, case-sensitive execution scoring
   (m3_instruments.answer_first_tokens) across the three execution
   controls, replacing the vendored surface_token_ids relaxation that
   caused the jspace "Te"/"Tehran" edge and the swap " K"/" Kettles"
   cross-task collision; surface scoring kept only for the M1 lens
   spot-check.
2. jspace anchored on capital-recall (36 items) not swap-capitals (16).
3. report-probe negative arm is a random-word-output null
   (m3_instruments.random_word_null_context) drawing from the other tasks'
   outputs, replacing the shuffled baseline that did not null
   morphological-output tasks.
4. fv-swap negative control is one-sided (random must not ELEVATE the
   task-B rate); a random swap that destroys computation no longer counts
   as a failure.

Gates: pytest 97 passed (+ answer_first_tokens / random_word_null_context /
one-sided-swap / null-not-removed tests), validators 3/3. Re-run is
evals-only on the cached lens. Expected on the run-2 raw evidence: swap
gates (with the collision removed the none B-rate should fall and the
one-sided negative should pass), report-probe gates (random-word null
should read ~prior), jspace re-measured cleanly on 36 items decides on its
own numbers — no thumb on the scale.

### 2026-07-18 — M3 run 3 (D-012): 3/4 instruments gated

Run results/m3/20260718-174954-instrument-gate finalized. Lens reproduced
(identity + HMR 2.50@L16). Verdicts:

- fv-direction-ablation: GATED. exec 0.933 -> 0.000 under exact scoring,
  sham 0.900/0.933; unchanged by the D-012 scoring switch (within-task, no
  collision to fix).
- jspace-ablation: GATED. On capital-recall (36 items, exact scoring), the
  top-10-atom J-space projection moved exec 0.861 -> 0.583 (drop 0.278 >=
  0.15); sham 0.833 within bound. The run-2 sub-threshold reading (0.125 at
  N=16 on swap-capitals with relaxed scoring) was under-powered and
  scoring-fragile, not a weak instrument: re-measured on the anchored task
  at finer granularity it clears the bar. The ruling was to measure cleanly
  and let the numbers decide; they did.
- fv-swap: GATED. With case-sensitive scoring the clean none B-rate fell
  0.467 -> 0.000 (confirming the run-2 0.467 was entirely the " K"/
  " Kettles" first-token collision); lens_swap 0.900, direct_swap 0.933,
  random_target 0.000; one-sided negative passes. gain 0.933.
- report-probe: GATED for capitalize (null 0.000) and english-french (null
  0.000); NOT gated for singular-plural (null 0.722). Replay (read-only):
  the random-word null pool for singular-plural (3546 words = capitalize +
  english-french outputs) contains zero plural outputs, yet the
  forced-choice probe reads the singular-plural label 26/36 in the null.
  The readout is therefore not coming from the in-context output mapping;
  it is consistent with the label being readable from the singular-noun
  inputs/query independent of the task examples. Stated operationally
  only: WHY the label is readable without the task is a research question,
  adjacent to a CONSTRAINTS HYPOTHESIS (label vs output-vocabulary
  decodability), and is NOT asserted here.

Net: three instruments fully gated (fv-direction-ablation, jspace-ablation,
fv-swap); the report probe gated for 2 of 3 tasks. The negative control
correctly refused to certify the report probe on singular-plural, where on
this model it cannot separate task-present from task-absent. This bears
directly on E2 (double dissociation, HYPOTHESIS, specified on
singular-plural), which needs a report measure on that task — flagged for
the E2 design. Disposition + M3 sign-off put to Ecaterina. Run 3 evidence
committed.

### 2026-07-18 — D-013: M3 disposition ruled; M3 signed off (Claude)

Ecaterina held to read the run 3 evidence (review map: the five headline
cells in results/m3/20260718-174954-instrument-gate/raw_completions, each
re-derivable by counting exact-match hits per line), then ruled:

- D-013 (ruled by Ecaterina, 2026-07-18, session instruction after the
  evidence hold): the report probe is certified per task — gated for
  capitalize and english-french; on singular-plural it is WITHDRAWN per
  the instruments LAW unless rebuilt and re-controlled. Mechanical
  enforcement: BANNED_INSTRUMENTS entry
  `report-probe-forced-choice@singular-plural` in
  jtvec/core/instruments.py (the @task scoping follows the M2
  fv_todd@<task> certificate convention), pinned by a test that the same
  probe name on the gated tasks still admits. The E2 prereg must resolve
  the singular-plural report measure explicitly — a redesigned probe/null
  re-controlled on that task, or a respecified task pair. Carried as an
  E2 design requirement.
- M3 deliverable stands: ControlRecord pairs (controls.json) for
  fv-direction-ablation, jspace-ablation, fv-swap, and
  report-probe-forced-choice@{capitalize,english-french}; lens
  re-materialized on this machine with M1-identical identity; evidence
  run results/m3/20260718-174954-instrument-gate (+ runs 1-2 history).

- sign-off: M3 — Ecaterina, 2026-07-18, via session instruction
  ("sign off") after the evidence hold, ratifying the D-013 disposition.
  Recorded by Claude, per the M0/M1 precedent.

Next per build order: M4 confirmatory experiments, E1 (decodability)
first. No E1 run before its own prereg is committed and its resource
estimate ruled.

---

## 2026-07-19 — M4-E1: scope ruled (D-014), build, prereg, launch (Claude)

E1 re-tests the FV-label HYPOTHESIS (v1 Exp-1, contaminated single-draw)
on M2-certified FVs: decode fv_todd through the M1-gated J-lens transport
vs the logit lens, task-label full-vocab rank over v1's registered label
sets, 3 certified tasks x 3 FV draws x 3 lens draws (+ v1's two robustness
variants at lens draw 0). Claim ledger entry CLM-001 (hypothesis) opened;
prereg harness/preregs/EXP-M4-E1-decodability.md.

- D-014 (ruled by Ecaterina, 2026-07-18/19, session Q&A): (a) decision
  constants ratified as drafted — C1 jlens median rank <= 20, C2
  jlens<logit ordering in every grid cell, C3 logit median rank >= 200,
  C4 >= 95/100 norm-matched random vectors beaten per draw; readout
  positive control (pinv-constructed label vector) rank <= 10, negative
  control random median >= 100. (b) Hendel vectors excluded entirely (no
  M2 certificate; estimator LAW). (c) Standing launch go once gates are
  green on the build and the prereg is committed (~25-50 min projected,
  bounded 75). (d) skip16_n10 kept with a proportional inclusion rule
  (>= ceil(0.75 x n) of an instance's band-overlap layers; identical to
  the ratified 10/13 for the primary instances; a variant failing its
  control voids only its own C2 cells) — ruled after the build surfaced
  that skip16's source layers overlap the M1-gated band at L16 only,
  where v1 read that variant over ungated layers 16-22.
- Build: jtvec/e1_decodability.py (registered label sets pinned to the
  vendored v1 definitions by test, rank statistics, control rules,
  C1-C4 decision rule), scripts/m4_e1_gate.py (orchestrator behind
  start_run; lens draws identity-checked per draw against M1's committed
  manifests; in-run readout controls run before any FV is read),
  configs/m4_e1_pythia410m.yaml (calibration/fit mirror M1 draw 0; lens
  draws get per-draw cache_dirs — the lens cache key excludes the seed),
  tests/test_e1_decodability.py. New instrument surface named
  jlens-label-rank-readout@<task> (the M1 gate covers the lens on
  residual streams; E1's rank readout of static vectors carries its own
  in-run ControlRecord pair).
- E1 is read-only: no interventions, hence no sham twins; the
  norm-matched random-vector arm is the reading-level null (and rank
  statistics are provably invariant to the norm rescaling — pinned by a
  unit test; the companion norm cells are recorded anyway per the
  prereg's sample plan).

### 2026-07-19 — E1 run complete: 0/3 tasks decodable at the bar; replayed (Claude)

Run results/m4/20260719-021823-e1-decodability (prereg + code 6f9798f)
finalized end to end: wall 2,299 s (inside the 25-50 min estimate), peak
RSS 3.34 GB, device cuda. Prerequisites: lens draws 1-2 refit on this
machine matched their committed M1 manifests exactly (identity incl.
calibration sha256; draw 0 cache-hit from the M3-verified fit) — all
three M1 lens draws now reproduce cross-machine. Readout instrument
controls all PASS before any FV was read: positive 13/13 band layers on
every (instance x task); negative random-vector medians 628-1,328 vs
bound 100. ControlRecord pairs issued for
jlens-label-rank-readout@{capitalize,singular-plural,english-french}.

Verdicts under the D-014 constants: NOT-DECODABLE on all three tasks —
0/3 DECODABLE-AND-SEPARATED, zero INSTRUMENT-VOID. Per the preregistered
rule this is evidence against the FV-label HYPOTHESIS at this
model/config; CLM-001 stays hypothesis with the counter-evidence recorded
in its entry.

- capitalize: jlens label-rank median 278 (IQR 485 over the 9-cell grid)
  vs C1 <= 20; random beaten 79/80/80 vs C4 >= 95; logit median 6,563
  (C3 pass); jlens < logit ordering 33/33 cells (C2 pass).
- singular-plural: 436 (IQR 261); beaten 80/79/77; logit 3,203 (C3
  pass); C2 33/33.
- english-french: 56 (IQR 31); beaten 96/97/95 (C4 PASS); logit median
  114 (C3 FAIL, < 200); C1 fail; C2 33/33.

Raw-output replay (read-only; surprise rule) before anything else, facts:

1. Every headline number re-derives by hand from the committed grids
   (9-cell and 3-draw medians; C4 counts recomputed from the
   random_*/decode_* raw cells: 79/80/96 for FV draw 1, equal to stored).
2. Grid variance is lens-draw-dominated, not FV-draw-dominated: within a
   lens draw the three certified FV draws agree tightly (capitalize at
   lens draw 2: ranks 66/64/74) while across lens draws the same
   statistic moves ~10x (capitalize 64-74 / 256-282 / 559-619 by lens
   draw; english-french 17-30 at lens draw 1 vs 61-65 at draw 2). The
   M2-certified FVs are draw-stable through this readout; the lens draw
   is the dominant nuisance at these ranks. Design consequence flagged
   for E2-E4: any readout-based number must marginalize over lens draws
   (the 3x3 grid did; v1's single-lens single-FV Exp-1 could not).
3. v1-parity: under v1's Exp-1 criteria (ordering + random control, no
   absolute bar) english-french would have counted decodable here
   (ordering 33/33, random 96/97/95). The v2 absolute bar C1 <= 20 and
   logit floor C3 >= 200 are what refuse it — the stricter preregistered
   rule ruling as written, not an instrument failure (all controls
   passed).
4. Top-token content of the jlens readouts (decode_* cells; post-hoc
   replay observation, labeled as such), stated operationally: the top
   of the readout is dominated by task OUTPUT items, not label words —
   singular-plural: literal plural nouns ("stations", "names",
   "objects") and the "s"/"es" morpheme; english-french: French tokens
   ("é", "de", "que"); capitalize: punctuation/sentence-start tokens.
   Label words never enter the top-16 except english-french at lens
   draw 1 (ranks 17-30). The preregistered descriptive arm points the
   same way: output-cloud mean rank jlens 1.3k/2.9k/11.9k vs logit
   13.8k/13.3k/20.2k (best layer, lens draw 0). Whether label and
   output-vocabulary readability are separable FV properties is a
   CONSTRAINTS HYPOTHESIS; nothing beyond the readings above is asserted.

- D-015 (proposed, awaiting Ecaterina): correction of D-014(d)'s premise.
  skip_first in the vendored jlens is a calibration-position parameter
  (leading prompt positions excluded from the Jacobian average;
  jlens/fitting.py valid_position_mask), NOT a source-layer restriction;
  every lens variant has all 13 band layers (the run's positive
  controls: 13/13 on every instance). My "skip16 overlaps the band at
  L16 only" claim behind the D-014(d) question was wrong. Operational effect nil:
  ceil(0.75 x n) equals the ratified 10/13 at n=13 for every instance —
  no criterion or verdict differs. Proposal: (a) this entry stands as
  the LABNOTES correction; (b) one prereg Deviations note correcting the
  false parenthetical in the Instruments section ("skip16_n10's single
  band layer L16 must pass") — text-only, no criterion change.
  Ecaterina rules on (b).
- D-015 ruled by Ecaterina, 2026-07-19 (session Q&A): correction
  ratified; the Deviations note is recorded in the prereg. In the same
  session Ecaterina ruled the next step: proceed to E2 design (folding
  in the E1 lens-draw-marginalization lesson and the D-013
  singular-plural report-measure requirement).

---

## 2026-07-19 — M4-E2: scope ruled (D-016), report-gate build (Claude)

E2 design surfaced a hard apparatus constraint: v1's Exp-3 tested a double
dissociation on singular-plural and a one-way dissociation on
landmark-country, but v2's certified set is {capitalize, singular-plural,
english-french} and the M3 instruments do not cover the v1 pair. Per-task
apparatus after M2/M3: capitalize has both a gated fv-direction-ablation
(execution) and a gated report probe; singular-plural has the gated
fv-ablation but its report probe was WITHDRAWN (D-013); english-french has
the report probe but no controlled fv-ablation; landmark-country has no
certified FV and no gated instruments at all. The verbalization half of the
double dissociation (HYPOTHESIS) is thus unmeasurable on singular-plural,
and the whole landmark-country arm has no v2 apparatus.

- D-016 (ruled by Ecaterina, 2026-07-19, session Q&A): Path A — unblock
  singular-plural first. Rebuild + control-gate a report instrument on
  singular-plural, then run the singular-plural
  double dissociation (HYPOTHESIS) cross-draw over the 3 certified FVs;
  defer landmark-country's certification (its own FV stability gate +
  instrument controls) to a separate later decision. Chosen over B
  (respecify the double dissociation (HYPOTHESIS) to capitalize — the only
  fully-gated task today, but a different task than CONSTRAINTS specifies)
  and C (certify the full v1 pair up front — most faithful, hours of
  prerequisite gated runs).

- Report-gate build (this commit): the rebuild is v1's prior-corrected
  report SCORE (vendored scripts/11 protocol), NOT the withdrawn
  forced-choice argmax. jtvec/report_instruments.py (bootstrap-CI control
  rule; positive = coherent report_score CI-low > 0; negative = coherent
  CI-low > shuffled CI-high AND > other CI-high), scripts/m4_e2_reportgate.py
  (model-only orchestrator behind start_run — no lens, no FV),
  configs/m4_e2_reportgate_pythia410m.yaml, tests/test_report_instruments.py.
  New instrument name report-score-prior-corrected@singular-plural (the
  D-013 rebuild-under-a-new-name path; the banned forced-choice name is
  untouched).

- Why the prior-corrected score can succeed where the forced-choice probe
  failed: D-013's failure was that " plural" is legible from the
  singular-noun INPUTS (null read 26/36 with singular-noun inputs kept).
  The negative control here is the shuffled arm — same singular-noun
  inputs, scrambled output mapping — so "coherent >> shuffled" is exactly
  the test that the readout comes from the task mapping, not the inputs.
  Not assumed to pass: if coherent≈shuffled the gate refuses to certify
  (measure cleanly, let the numbers decide — D-012 lesson), and that
  refusal blocks E2's singular-plural verbalization measure, a scope
  result put back to Ecaterina. Prereg drafted UNCOMMITTED
  (EXP-M4-E2-reportgate.md); committing it is the prereg act, after the
  threshold ratification.

### 2026-07-19 — E2 report-gate run: GATED under P3 only; replayed (Claude)

Thresholds ratified (session Q&A); prereg committed 9dd7ff0; run
results/m4/20260719-053911-e2-reportgate finalized (wall 18.4 s, peak RSS
3.32 GB, cuda). Verdict: report-score-prior-corrected@singular-plural
GATED, best_phrasing P3. Per-phrasing (bootstrap 95% CIs, N=40/cell):

- P1: coherent +0.07 [-0.03,+0.17] — CI straddles 0, positive FAILS (no
  detection); shuffled -0.26.
- P2: coherent +0.25 [+0.17,+0.33] vs shuffled +0.27 [+0.20,+0.34] —
  positive passes but negative FAILS: coherent ~ shuffled, the D-013
  input-leakage exactly (the mapping adds nothing beyond the singular-noun
  inputs). Caught, not certified.
- P3: coherent +0.36 [+0.29,+0.44] vs shuffled +0.14 [+0.05,+0.23] vs
  other -0.06 [-0.21,+0.08] — coherent CI-low +0.29 clears shuffled
  CI-high +0.23 AND other CI-high +0.08. Positive AND negative PASS.

Raw-output replay (read-only; surprise rule — the phrasing-split was not
expected), facts:
1. Every cell mean re-derives by hand from raw report_score (P3 coherent
   +0.364, shuffled +0.142; P2 coherent +0.248, shuffled +0.271).
2. The signal is a CONTINUOUS log-prob elevation of " plural", NOT a
   forced-choice win: the free-running argmax is "first" (P3, 39/40) or
   "answer" (P2, 34/40) — the probe tail's grammatical completion — and
   " plural" sits at median full-vocab rank ~9 (P3 coherent) / ~13 (P3
   shuffled). The gated quantity is the MAPPING-specific margin: under P3
   the coherent mapping lifts log p(" plural") ~+0.22 above the input-only
   (shuffled) baseline. So E2's verbalization measure is this continuous
   report_score under P3, not a report "accuracy".
3. The >=3-phrasing discipline earned its keep (CONSTRAINTS PROVISIONAL
   "single phrasing" risk): 2 of 3 phrasings would have misled — P2
   false-positive on detection while leaking, P1 no detection. Only P3
   isolates the mapping.

Consequences carried to the E2 dissociation design: (a) the report
measure is fixed to P3, report_score continuous; (b) the signal is WEAK
(margin ~+0.22 log-prob, rank ~9), so E2 must have the power to detect an
ablation-induced REDUCTION in it — an N / effect-size question for the E2
prereg, flagged now; (c) the instrument is GATED (ControlRecord in the run
controls.json); E2's prereg cites it. No BANNED_INSTRUMENTS change — this
is a pass under a new name, the D-013 forced-choice entry stands. Disposition
+ the E2 dissociation design go to Ecaterina next. Run evidence committed.

### 2026-07-19 — M4-E2 dissociation build (D-017); high-N ruled (Claude)

With the report measure unblocked, the E2 dissociation is designed on
singular-plural (D-016 Path A). Ecaterina ruled "build as designed,
high-N" over strengthening the report measure first or hedging with
capitalize (session Q&A). CLM-002 opened (hypothesis).

- Design: two measures (execution = greedy exact-match accuracy; report =
  report_score under P3, the gated instrument), two M3-gated ablations
  (fv-direction, jspace) each vs matched sham, at final position of band
  layers 4–16. Cross-draw ablation transfer (CONSTRAINTS): the fv ablation
  re-derived from each of the 3 M2-certified FV draws, jspace from each of
  the 3 M1 lens draws (jspace reads the lens — the E1 nuisance axis). Every
  effect is a 3-draw DrawSet (clean − ablated_k). Context sets sampled once
  and reused across conditions (paired, to fight the weak P3 signal).
- Decision (D-017, thresholds await ratification): an ablation hurts a
  measure iff effect_median − sham_median ≥ δ (δ_exec 0.15, δ_report 0.10
  log-prob). Direction 1 = fv hurts execution not report; Direction 2 =
  jspace hurts report not execution; DOUBLE-DISSOCIATION (HYPOTHESIS) iff
  both, with per-arm cross-draw transfer flags (every draw clears
  sham+δ). N_exec 50, N_report 80.
- Build: jtvec/e2_dissociation.py (effect DrawSet + DissociationRule),
  scripts/m4_e2_dissociation.py (orchestrator behind start_run; asserts all
  three consumed instruments gated via require_controlled against their
  committed ControlRecords before measuring), configs/m4_e2_dissociation_
  pythia410m.yaml, tests/test_e2_dissociation.py. Reuses the ablation hooks
  M3 exercised (exp3.make_hooks / final_logits_under), exact-match scoring
  (D-012 answer_first_tokens), and the certified-FV / lens loaders.
- Setup smoke (off-run, not committed): lens draw 0 (cache/m3) identity
  matches M1 draw 0; FV loads; hooks fire; on one item fv ablation breaks
  execution (hit True→False) while jspace leaves it (hit True) — wiring
  validated (M3-run-1 precedent: catch code bugs before the gate).
- Weak-signal caveat carried from the report-gate: the jspace→report arm
  must detect a reduction in a ~+0.22 margin; it may return inconclusive,
  which the prereg pre-commits to reporting rather than engineering around.
  Prereg drafted UNCOMMITTED (EXP-M4-E2-dissociation.md); committing it is
  the prereg act, after threshold ratification.

### 2026-07-19 — E2 dissociation run: ONE-WAY; replayed (Claude)

Thresholds ratified (D-017, session Q&A); prereg + CLM-002 committed
3ba3903; run results/m4/20260719-142007-e2-dissociation finalized (wall
86 s, peak RSS 3.37 GB, cuda). All three consumed instruments asserted
gated before measuring. Clean: execution 0.920 (N=50); report_score
+0.389 (P3, N=80, baseline -4.625). Effects (clean - ablated; median over
3 draws, vs matched sham):

| ablation x measure | effect med [IQR] | sham med | effect-sham | hurts? |
|---|---|---|---|---|
| fv x exec    | +0.920 [0.000] | +0.020 | +0.900 | YES |
| fv x report  | -0.638 [0.017] | +0.037 | -0.675 | no (report ROSE) |
| jspace x exec   | +0.440 [0.160] | +0.000 | +0.440 | YES |
| jspace x report | +0.341 [0.155] | +0.295 | +0.046 | no (~= sham) |

Verdict: ONE-WAY. Direction 1 (fv hurts execution NOT report) holds and
transfers across all 3 FV draws; Direction 2 (jspace hurts report NOT
execution) fails on two independent counts — jspace hurts execution, and
its report effect does not beat its own sham.

Raw-output replay (read-only; surprise rule), facts:
1. Every effect re-derives by hand from the raw cells (clean exec 0.920,
   report +0.389; fv-ablated report +1.05/+1.02/+1.03; jspace exec drops
   0.44/0.36/0.68 by lens draw).
2. fv-ablation execution: 0.920 -> 0.000 on all 3 FV draws (IQR 0). The
   outputs collapse from real plural forms (clean top1: "markers",
   "beaches", "refriger[ators]") to generic function words ("a" 36/50,
   "the" 4/50). Projecting out the certified FV surgically removes the
   singular->plural computation; the effect is FV-draw-invariant.
3. fv-ablation report: the free-running argmax is UNCHANGED ("first"
   77/80 clean -> 80/80 ablated); yet log p(" plural") rises ~+0.66
   (report_score +0.389 -> ~+1.03). Removing the FV RAISES the abstract
   label readout without changing what the model says. Operational
   reading only: the FV's presence suppresses the " plural" label token
   relative to the plural-FORM tokens it carries (E1 found the FV's top
   tokens are the output vocabulary). This is adjacent to the quarantined
   CONSTRAINTS HYPOTHESIS about ablation raising report accuracy, but is
   distinct (readout log-prob, not accuracy; fv not jspace) and is NOT
   asserted.
4. jspace hurts execution (median 0.44): outputs become truncated/mangled
   ("h", "beach", "Mix"), a noisier degradation than fv's clean collapse.
   So jspace is a broad residual-stream perturbation, not an
   execution-sparing one.
5. jspace report effect (0.341) does NOT exceed its matched 10-random-
   direction sham (0.295): the P3 report readout is fragile to ANY 10-dim
   final-position projection — the sham alone flips the argmax
   ("first" -> "answer", 53/80) and drops log p(" plural") comparably.
   This is the weak-signal risk (report-gate: ~+0.22 margin) realized: at
   10 dimensions the report measure cannot separate a structured jspace
   ablation from random noise of the same rank.

Net: a ONE-WAY dissociation on singular-plural — fv-direction ablation is
execution-specific and robust (spares/raises report), while jspace is not
report-specific (it also degrades execution, and its report effect is
sham-indistinguishable). This is one direction of the v1 double
dissociation (HYPOTHESIS), not both; CLM-002 stays hypothesis per the
prereg (only DOUBLE-DISSOCIATION with both transfer flags promotes it).
The robust, cross-draw fv execution-specificity is recorded as the
substantive sub-result. Disposition + next step (E3 per build order) go
to Ecaterina. Run evidence committed.

- Disposition ruled by Ecaterina, 2026-07-19 (session Q&A): (a) open
  CLM-003 at preliminary for the robust one-way finding (fv-direction
  ablation removes singular-plural execution while sparing/raising report,
  cross-draw), evidence commit a1c7cb1 — promotion to verified still needs
  her raw-read verify: line; CLM-002 (the double dissociation, HYPOTHESIS)
  stays at hypothesis tier. (b) Proceed to E3 (swap) per build order.

---

## 2026-07-19 — M4-E3: scope ruled, swap build (D-018) (Claude)

E3 tests whether the FV causally carries transferable task identity: on
capitalize (task A) prompts, swap the FV_A component onto FV_B
(singular-plural) and measure the task-B answer rate, vs a random-target
control; lens_swap vs direct_swap asks whether that identity is specific to
the J-lens basis or lives in the raw residual direction. Same apparatus
constraint as E2: v1's headline swap pair was the translation pair
english-french<->english-spanish, but english-spanish is uncertified in v2.

- Scope ruled by Ecaterina, 2026-07-19 (session Q&A): the gated
  capitalize->singular-plural pair only (M3 fv-swap ControlRecord),
  cross-draw over the 3 certified FV draws; the translation pair
  (english-spanish certification) is deferred to a follow-up, as D-011
  anticipated. Chosen over certifying english-spanish now (the richer
  label-vs-output-vocabulary contrast, more prerequisite work) or doing
  both. CLM-004 opened (hypothesis).
- Build: jtvec/e3_swap.py (SwapRedirectionRule: redirects iff best-swap
  B-gain median >= 0.20 and random gain <= 0.05, one-sided; J-specific iff
  lens-direct median >= 0.15, else basis-agnostic; per-draw B-rate
  DrawSets + cross-draw transfer flag), scripts/m4_e3_swap.py (orchestrator
  behind start_run; asserts fv-swap gated; reuses the vendored swap hooks
  and exact-match D-012 scoring; paired contexts), configs/m4_e3_swap_
  pythia410m.yaml, tests/test_e3_swap.py. D-018 thresholds await
  ratification; constants in code match. Prereg drafted UNCOMMITTED
  (EXP-M4-E3-swap.md).
- Prior context: the M3 fv-swap control (single draw) already moved the
  B-rate 0.000 -> 0.900/0.933 (lens/direct) with random 0.000; E3 adds the
  3-draw discipline, the CI/threshold verdict, and the lens-vs-direct
  J-specificity result as a registered claim. Given M3's direct_swap ~=
  lens_swap, a basis-agnostic outcome is plausible — the run decides.

### 2026-07-19 — E3 swap run: REDIRECTS-BASIS-AGNOSTIC; replayed (Claude)

Thresholds ratified (D-018); prereg + CLM-004 committed 0c523d2; run
results/m4/20260719-151956-e3-swap finalized (wall 17.5 s, peak RSS 3.37
GB, cuda). fv-swap asserted gated. Task-B answer rate by condition
(median over 3 FV draws):

| condition | draw1 | draw2 | draw3 | median |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 |
| lens_swap | 0.933 | 0.933 | 0.933 | 0.933 |
| direct_swap | 0.800 | 0.767 | 0.867 | 0.800 |
| random_target | 0.033 | 0.000 | 0.000 | 0.000 |

Verdict REDIRECTS-BASIS-AGNOSTIC: best-swap B-gain median +0.933 (random
+0.000) clears 0.20 and separates from the control; transfers (every draw
0.933); lens-direct gap +0.133 is below the 0.15 J-specificity bar, so the
task identity is carried by the raw residual direction, not specifically
the J-lens basis. CLM-004 -> preliminary per the prereg (REDIRECTS +
transfer).

Raw-output replay (read-only), facts:
1. B-rates re-derive by hand (none B 0.000 / A 0.933; lens_swap B 0.933 /
   A 0.000; direct_swap B 0.800 / A 0.000; random B 0.033 / A 0.000). Under
   both swaps task A is fully suppressed (A -> 0.000), so this is a genuine
   redirection, not additive noise.
2. Swap outputs are real plural forms ("laptops", "shirts", "airplanes"),
   including the irregular "mouse" -> " mice" under lens_swap — the swap
   engages morphological pluralization, not a bare "+s". random_target
   outputs empty/newline (23/30) — a norm-matched random target destroys
   computation (as CONSTRAINTS records), scoring B ~ 0.
3. The lens-direct gap (0.133): on 5/30 queries lens_swap yields the full
   plural while direct_swap yields a truncated first token ("book" ->
   " books" vs " b"; "mouse" -> " mice" vs " m"). The lens basis is
   marginally cleaner but not necessary — direct redirects on 80% and both
   null task A. Hence basis-agnostic (below the bar), not J-specific.

Net: on capitalize->singular-plural the certified FV carries transferable,
causal task identity — swapping it redirects execution to task B (0.00 ->
0.93, all 3 draws) above a random control, in the raw residual basis. With
E2 (fv ablation removes execution) this is the causal complement:
ablation removes, swap redirects. E1 found the same FV is not lens-readable
as a label. All scoped to Pythia-410M; the separability HYPOTHESIS stays at
tier. Disposition + next step (E4, the confabulation HYPOTHESIS) to
Ecaterina. Evidence committed.

- Disposition ruled by Ecaterina, 2026-07-19: open CLM-004 at preliminary
  for the REDIRECTS-BASIS-AGNOSTIC result (evidence commit 0d8b278);
  promotion to verified still needs her raw-read verify: line.

---

## 2026-07-19 — Strategic pivot to the main-track emergence sweep (Claude)

Ecaterina set the target to an **EACL main-track** paper and, after a
brutally-honest assessment of the current results, ruled the next
investment (session Q&A). The honest read: E1 (FV not label-decodable,
counter to v1), E2 (ONE-WAY, not the full double dissociation, HYPOTHESIS),
E3 (swap redirects — close to known Todd/Hendel function-vector results) are
a coherent *mechanism foundation* but, on a single model with the headline
only partly replicated, are workshop-tier, not main-track. The
main-track-carrying result is developmental: whether ICL execution matures
early while portable, stability-gated FVs emerge late (HYPOTHESIS) — the
emergence sweep, uniquely enabled by Pythia's dense checkpoints, with the
multi-scale axis (kills the n=1 objection) sharing the same infrastructure.

- Rulings: (a) sweep ambition = emergence × MULTIPLE Pythia scales
  {160M..2.8B} (not single-scale); (b) compute posture = "confirm A100
  availability first" — produce concrete per-option A100-hour estimates for
  Ecaterina to secure the allocation before any launch; (c) E4
  (confabulation, HYPOTHESIS) deferred by this pivot (still on the docket as
  supporting work). Plan file: .claude/plans/quizzical-floating-emerson.md.

### 2026-07-19 — M4-emergence build (D-019 constants pending ratification)

Two Explore analyses grounded the plan: (1) extraction is inference-only,
cost = n_trials_aie·(L·H+1)·t_fwd, and head selection is already
per-checkpoint automatic in compute_function_vector (fixes v1's fixed-head
confound); the cheapest gate that reproduces converged_at=25 WITH a witness
rung is n_trials_aie=50 (rungs {25,50}), 1/4 the M2 ladder (~65 min/ckpt on
the laptop). (2) The extraction/cache/certificate layer is already fully
revision-keyed — set cfg.model.revision and lens/FV caches, manifests, and
certificates re-key with no collision — so the sweep is a thin driver +
reuse, not new gate code, and M2's signed-off code is untouched.

- Compute (A100 ≈ 5× laptop, 50-trial gate, 12 ckpts + 2 full-ladder
  anchors/scale): ~65 A100-h for 3 scales {410M,1B,2.8B} (2.8B ≈18×/ckpt
  dominates ~70%), ~90-110 for 5 scales — i.e. ~3-5 days on one A100.
  Highest-value lever: batching the batch-1 AIE head loop could cut this
  ~10-50× but edits vendored Todd code (byte-identity deviation → a ruling).
- Build: jtvec/emergence.py (model-free onset detection + per-scale
  developmental classification: DISSOCIATION / CO-EMERGENCE /
  INCONCLUSIVE-FV / NO-EXECUTION; scale-interaction roll-up),
  tests/test_emergence.py, scripts/m4_emergence_sweep.py (per-revision M2
  gate at rungs {25,50} reusing jtvec.fv_stability primitives + the
  scripts/13 loop/teardown/too-weak/jsonl skeleton; records execution,
  induction, E1 decodability; emergence.json roll-up),
  configs/m4_emergence_pythia410m.yaml.
- D-019 (proposed, awaiting ratification): exec onset = first ckpt at 0.8×
  the scale's max 10-shot accuracy; FV-stability onset = first gate PASS;
  DISSOCIATION iff log10 gap >= 0.5; FV numbers count only where the gate
  passes at >= 2 checkpoints (CONSTRAINTS). P-E1/2/3 in the prereg.
- Next: laptop wiring dry-run (capitalize @ the cached final revision,
  {25,50} gate — reproduce converged_at=25), then draft
  EXP-M4-emergence.md + CLM-005 with the A100-hour table, then PAUSE for
  Ecaterina's A100 allocation + the batching ruling. No scientific run
  before the prereg commit and her compute ruling (>12 h LAW). Gates at
  build: pytest 143 passed, validators 3/3.

### 2026-07-19 — dry-run caught a tokenizer-mutation bug; fixed (Claude)

The wiring dry-run (capitalize @ the cached final revision 9879c9b,
n_trials_aie=50) did NOT reproduce M2: converged_at=None (M2: 25),
induction gain +0.018 (M2: +0.394), cross-draw cosine 0.879 (M2: 0.991),
outvocab rank 11002 (E1: ~1302). Raw replay (read-only) traced it: the
extracted FVs were genuinely weak/different — the stored fv_todd cosine to
M2's was 0.64, and even the mean_head_activations differed by up to 5.58.
Root cause: jlens.from_hf(model, tokenizer) sets add_bos_token=True on the
SHARED tokenizer (verified: tokenize("dog") [21428] -> [0, 21428]). The
orchestrator built the lens (for decodability) BEFORE FV extraction, so
every extraction prompt got a spurious leading BOS -> corrupted FVs. M2
never builds a jlens model, so its extraction is clean.

Fix (this commit): run_checkpoint reordered into two phases — Phase A does
all raw-model work (execution evals, per-draw AIE extraction, the rung
gate) on the CLEAN tokenizer, exactly as M2; Phase B then builds the jlens
model + lens and decodes the decodability arm (the E1 rank statistic is
add_special_tokens=False, so BOS-independent — consistent with E1, which
also used from_hf). The BOS-corrupted cache/m4emerge and the buggy post_hoc
dry-run dir were deleted (non-scientific artifacts; M3-run-1 precedent).
This is exactly the M3-run-1 lesson: a cheap laptop dry-run caught a bug
that would have silently poisoned the entire (expensive) A100 sweep.

Re-run after the fix (capitalize @ 9879c9b, {25,50} gate) reproduces the
ground truth BIT-FOR-BIT: T=25 min_cos 0.9910 / gain +0.3941, T=50 min_cos
0.9971 / gain +0.3882, converged_at=25, gate PASS — identical to M2's
stability.json; and decodability outvocab 1302 / label 282 — identical to
E1. The cheap {25,50} revision-keyed sweep pipeline is validated against
both prior gates. The post_hoc dry-run dir is removed (reproducible wiring
artifact); the fix is committed at d795c48. PAUSE: the multi-scale A100
sweep awaits Ecaterina's allocation, the scale-set choice, D-019
ratification, and the batched-AIE optimization ruling; prereg
EXP-M4-emergence + CLM-005 commit and launch follow her go.

### 2026-07-19 — emergence-sweep decisions ruled; batched-AIE scoped (Claude)

Ecaterina ruled (session Q&A): (a) scale set = 3 {410M, 1B, 2.8B}; (b)
D-019 emergence constants ratified as drafted (exec onset 0.8x-max, FV
onset first gate PASS, DISSOCIATION iff log10 gap >= 0.5, >= 2 gate passes
required); (c) implement + re-validate the batched-AIE optimization BEFORE
the A100 run. Prereg EXP-M4-emergence + CLM-005 remain drafted (scratchpad)
pending the prereg act at launch.

Batched-AIE spec (D-020, to record at implementation): the AIE inner loop
(third_party/function_vectors/src/compute_indirect_effect.py:71-88) does
384 batch-1 forwards/trial (one per layer x head), each patching one head's
final-position activation. The vendored replace_activation_w_avg
(intervention_utils.py:50-54) already has a batched_input (batch-by-head)
path but it is abandoned/broken for pythia: the projection
`torch.addmm(bias, inputs.squeeze(), W.T)` (intervention_utils.py:79)
requires 2-D and fails on a batched (n_heads, tokens, resid) input. Fix:
(1) in the pythia/gpt-neox branch, use a batched `torch.matmul(inputs,
W.T) + bias` when the input is 3-D/batched (keep the exact addmm path for
the unbatched case); (2) wire activation_replacement_per_class_intervention
to build sentences = [prompt]*n_heads, layer_head_token_pairs =
[(layer, h, last_tok) for h in range(n_heads)], batched_input=True, one
forward per LAYER (24 instead of 384), extracting per-head logits from the
batch. ~16x fewer forwards (~65 -> ~4 A100-h for 3 scales). RE-VALIDATION
GATE (the safety net for this core-computation vendored edit): a test that
the batched AIE reproduces the unbatched AIE fp-close (allclose atol ~1e-4)
AND that the resulting FV reproduces M2's converged_at=25 / cosine / gain
on capitalize. Not shipped unless that passes. Deviation recorded in
VENDORING.md + D-020 at implementation. Deferred from this session's tail
to a focused implementation (delicate scientific-core surgery; the
validated sweep pipeline + all decisions are locked and committed).

---

## 2026-07-19 — Taxonomy phase opened (M5+): residency pivot (Claude)

Ecaterina ruled (session kickoff instruction): the project pivots from the
"task vectors in the J-space" framing to the workspace-residency taxonomy —
five axes A1–A5, species S1–S5, hypotheses H1–H5 — per TAXONOMY_DESIGN.md
and M5_SPEC.md (committed fd56653). The Aug-3 deadline is dropped; the
target is main-track quality on the science's own schedule. The emergence
sweep is deferred to after the adult-model matrix; its validated pipeline
and the D-019/D-020 rulings stay locked, untouched.
E4 (confabulation, HYPOTHESIS tier) remains deferred on the docket. H1–H5 are
HYPOTHESIS-tier entries in
CONSTRAINTS.md (9b01ab0), with superseded-by-taxonomy annotations appended
to the recast v1-framing entries (append-only; nothing deleted).

Machine + gates: the kickoff described a fresh clone on a third platform;
in fact this session runs on the original M1 MacBook (the M0/M1 machine),
existing checkout at origin/main c07f4d1, clean tree, submodules at the
pinned SHAs (jacobian-lens 581d398, function_vectors fb9eac7), with the
untracked cache/draw{0,1,2} (three certified 410M lens draws) intact from
M1. Platform verification on this machine, this date: `uv sync --frozen`
clean, `uv run pytest -q` 143 passed, validators 3/3 PASS. No platform
issues to report (D-008 precedent: none needed fixing).

Proposed decisions put to Ecaterina (awaiting her ruling; nothing adopted):

- D-021 (proposed): M4 disposition. No `sign-off: M4` line exists and the
  build-order protocol conditions M5 start on the previous milestone's
  sign-off. Proposal: a partial-closure line (E1–E3 evidence closed at
  their recorded verdicts; E4 + emergence sweep deferred by this pivot) —
  or her alternative ruling.
- D-022 (proposed): LRE relation data vendoring. M5.0 needs >= 8 relations
  from the Hernandez et al. set; no such data exists in the repo or the
  vendored Todd datasets. Proposal: vendor github.com/evandez/relations at
  a pinned commit (submodule, mirroring D-004 mechanics) with attribution
  in VENDORING.md; battery membership specified in EXP-M5-0 at
  ratification.
- D-023 (proposed): Pythia-1.4B revision pin for M5 configs — resolve the
  current EleutherAI/pythia-1.4b main SHA and pin it, mirroring the 410M
  @9879c9b pattern.
- Compute placement for the 1.4B lens gate (Mac-overnight vs GPU tier) and
  the binding battery: held until the pre-prereg probe numbers are in
  (recorded in a follow-up entry).

Standing open items surfaced at session start, per the kickoff: (i) D-002 —
the repo is public, still unruled; (ii) CLM-003 and CLM-004 are at
preliminary awaiting Ecaterina's verify: lines (raw dirs
`results/m4/20260719-142007-e2-dissociation/raw_completions/` and
`results/m4/20260719-151956-e3-swap/raw_completions/`).

Session discipline note: preregs EXP-M5-0-qualification and
EXP-M5-1-concept-gate are being drafted UNCOMMITTED; committing is the
prereg act and happens only after Ecaterina ratifies thresholds. No
scientific run starts before that commit and her compute rulings.

### 2026-07-19 — pre-prereg probes: 1.4B on the M1 Mac (non-scientific)

M2-probe/D-009 precedent; outputs in the session scratchpad only, nothing
scientific. pythia-1.4b@fedc38a (D-023 proposed pin) fp32 on MPS:

- Smoke: load 11.1 s; first forward 1.08 s; greedy 8-token generation
  1.02 s/gen; capital-recall sample answered correctly.
- Jacobian lens-fit probe (1 calibration prompt, 23 source layers,
  d=2048, dim_batch=8, skip4): 661.6 s/prompt probe pass, 621.6 s fit
  pass; peak RSS 7.39 GB (ru_maxrss floor; MPS-side allocations partly
  uncounted). The in-fit max_d_mean=nan is definitional at n_prompts=1
  (running-mean statistic needs n_done > 0), not an anomaly.
- Projections (this Mac): M5.0 baseline batteries both substrates
  ≈ 2.5-3.5 h; 1.4B lens gate skip4-only 3 draws ≈ 7-8 h wall, RSS
  ≈ 7.5-9 GB — under the 12 h LAW, so Mac-overnight-eligible; a {2,4,8}
  skip sweep on all draws ≈ 16 h (over the LAW → GPU tier if wanted);
  draw-0-only sweep ≈ 11 h (no margin, not recommended on this Mac).
  Compute placement awaits Ecaterina with the EXP-M5-0 ratification.

Prereg drafts EXP-M5-0-qualification / EXP-M5-1-concept-gate and the five
m5 configs carry these numbers; all remain UNCOMMITTED awaiting her
thresholds ruling. The 410M side needs no re-fit: cache/draw{0,1,2} lens
draws from M1 are intact on this machine.

---

## 2026-07-20 — Taxonomy-phase rulings D-021..D-024; preregs committed (Claude)

Ecaterina ruled the open items (session instruction; attribution confirmed
for this session's rulings — she types milestone sign-off and verify: lines
herself from now on):

- D-002 RULED: the repo stays PUBLIC. The four housekeeping commits
  (fd56653, 9b01ab0, 628b205, 5dec2cc) are pushed to origin. The
  TAXONOMY_DESIGN.md commit (fd56653) is thereby the publicly timestamped
  registration of the prediction matrix; the scooping risk of a public
  prediction registry is accepted knowingly.
- D-021 RULED: partial M4 closure as proposed — E1, E2, E3 are closed at
  their recorded verdicts (E1 NOT-DECODABLE 3/3; E2 ONE-WAY, CLM-003
  preliminary; E3 REDIRECTS-BASIS-AGNOSTIC, CLM-004 preliminary); E4 and
  the emergence sweep stay deferred with their CONSTRAINTS entries live.
  This disposition authorizes the taxonomy phase (M5+) to proceed. The
  formal `sign-off:` line remains Ecaterina's to type (ruling 9); this
  entry records the disposition, not a sign-off in her name.
- D-022 RULED: vendor github.com/evandez/relations (Hernandez et al. ICLR
  2024) as a git submodule pinned at 1b9ec3c, D-004 mechanics; data only,
  code not imported (VENDORING.md updated). The M5.0 LRE bar reading is
  RATIFIED: few-shot CAPABILITY >= 0.60 at qualification, with faithfulness
  >= 0.60 as the M5.2 operator positive-control bar. 12-relation battery
  pinned in EXP-M5-0 and the qualification configs by dataset path.
- D-023 RULED: pin EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79
  (main resolved 2026-07-19) as proposed.
- Compute RULED: the 1.4B lens gate runs on the Mac overnight, skip4-only
  (the recommended option, ~7-8 h, under the 12 h LAW); the binding battery
  runs on the Mac; the {2,4,8} skip sweep is NOT purchased now (a later
  need for it becomes a flagged GPU-tier run).
- Prereg thresholds RULED: ratified as drafted. Both preregs committed as
  the prereg act — EXP-M5-0-qualification.md and EXP-M5-1-concept-gate.md,
  with the five m5 configs and the m5_1 concept config. Neither opens a
  CLAIMS.md entry (both are gates). EXP-M5-1's RUN still waits on EXP-M5-0
  to produce the admitted-substrate set and re-derived band; its prereg is
  committed early to lock predictions.
- D-024 RULED (new — scout tier authorized): AFTER the M5.0 baselines and
  the 1.4B lens gate complete, a SCOUT MATRIX runs before any further M5
  gate. Constraints: single-draw, uncertified, post_hoc-stamped,
  scratchpad/results-scout only, HARD-BANNED from CLAIMS.md and from
  LABNOTES findings language. Coverage: LRE extraction for the qualifying
  relations + quick A1 (decode) and A3 (lens-vs-direct application) reads
  per species, on whichever substrate qualifies. Purpose: a cheap preview
  of the prediction matrix to guide investment; wall-clock estimated before
  starting; target <= 2 sessions. Every scout cell showing signal earns the
  full gate treatment afterward; no scout number is ever quoted outside the
  scout report. (Discipline note: the scout report is not a findings
  document — its language stays at the preview/scan register, never the
  assertive findings verbs the language validator guards.)

Standing reminders still open (unchanged): CLM-003 and CLM-004 stay at
preliminary awaiting Ecaterina's verify: lines
(`results/m4/20260719-142007-e2-dissociation/raw_completions/` and
`results/m4/20260719-151956-e3-swap/raw_completions/`); she does these
before any paper text cites them.

Next (this session, no long run without its estimate + detached launch):
build the M5.0 qualification orchestrator (scripts/m5_0_qualification.py,
reusing scripts/02 baselines + the emergence exec_top1 patterns + the new
LRE/binding evals) with landing tests, and a generalized lens-gate
orchestrator for 1.4B (m1_gate.py is 410M/v1-reproduction-specific); commit
each before its run (start_run discipline); run the Mac baselines, then
launch the 1.4B lens gate detached overnight with a Monitor.

### 2026-07-20 — M5.0 lens-gate build + two prereg findings (Claude)

Built the 1.4B lens gate (EXP-M5-0 rule 5, ruled Mac-overnight skip4-only):
scripts/m5_lens_gate.py generalizes scripts/m1_gate.py off the 410M/v1
anchors (drops R3 capital-recall exact contrast, R4 calibration-hash
identity, R5 the v1 baseline table — a fresh substrate has no v1 reference)
and keeps the model-agnostic content as preregistered Q1-Q6. evaluate_gate
is a pure function with a model-free landing test (tests/test_m5_lens_gate.py,
10 cases: all-pass + each Q failure mode incl. the quant-aware Q3 bound and
the Q4 random-arm breach). The fit/eval pipeline (vendored scripts 01-04) is
unchanged and already validated on this model's one-prompt fit probe. Gates
at build: pytest 153 passed, validators 3/3.

Prereg conformance fix (text-only, D-015 precedent; FLAGGED for
acknowledgement): both preregs as first committed (113d04f) omitted the
`## Estimator plan` heading that start_run's check_prereg_sections requires,
so neither could run. Added the section to each (describing the estimators
already specified elsewhere in the file); no threshold or decision rule
changed. Recorded in-section and here.

Two prereg-vs-data findings surfaced while designing the M5.0 baseline
orchestrator (these BLOCK the baseline run, not the lens gate; proposed
decisions for Ecaterina):

- D-025 (proposed) — S1 qualification shot count. EXP-M5-0 rule 1 says
  "10-shot", but the 8 tasks/*.json are self-contained fixed-shot probes
  (capital-recall/operand/opposites/word-pairs ~3-shot, multihop ~2-shot,
  context-binding ~6-shot, typo-robustness 0-shot); the M1 anchor table
  scored them as-authored via jvec.evals.baseline.score_task. Proposal:
  score as-authored (built-in exemplars), report the per-task shot count,
  keep the 0.80 bar — a text-only rule-1 correction that keeps the anchor
  comparable. Alternative: rebuild the tasks as true 10-shot ICL (breaks
  anchor comparability, more construction).
- D-026 (proposed) — binding battery data does not exist. EXP-M5-0 rule 4
  describes Feng & Steinhardt bind2/bind3 (N=60 each), but tasks/ has only
  context-binding.json (a ~6-shot repetition-completion probe, not the
  entity-attribute binding format). Proposal: build a scripts/make_tasks.py
  extension generating bind2/bind3 from a template Ecaterina approves, run
  on the Mac; until then S4 admission is unmeasurable and S4 stays
  un-admitted (the matrix carries the hole explicitly).

Next: launching the 1.4B lens gate detached overnight (nohup + Monitor; the
~1 h background-kill lesson). Results + Q1-Q6 verdict get their own commit
(one commit per experiment). The M5.0 baseline orchestrator
(scripts/m5_0_qualification.py) is deferred pending D-025/D-026 — it needs
both rulings to fix its S1 and binding batteries before it is built + run.

---

## 2026-07-20 — M5.0 1.4B lens gate: verdict FAIL; raw replay (Claude)

Run results/m5/20260720-024819-p14b-lens-gate (prereg EXP-M5-0 rule 5, commit
cf1da2b; skip4-only, 3 draws seeds 0/1/2). m5_0_lens_verdict = FAIL. On
EleutherAI/pythia-1.4b@fedc38a, band [4,16], N_swap=16. Q1 PASS (all 3 draws
pass the vendored 9-check sanity gate), Q3 PASS (sham median 0.0004 <= 0.0625),
Q4 PASS; Q2, Q5, Q6 FAIL.

Raw replay (read-only, per the surprise->replay rule; nothing below asserted as
a finding):

- Q2 positive control FAIL, but on flip not effect. dp(swap_answer) median
  0.483 (draws 0.495/0.353/0.483) clears the 0.30 bar; swap_top1_rate median
  0.5625 (0.562/0.375/0.562) misses the 0.75 bar. Per-item dp is bimodal:
  strong on most capital pairs (0.70-0.94) with a few near-zero (England->Spain
  0.005, Greece->Egypt 0.009, Poland->Ireland 0.05). The swap moves answer
  probability; top-1 flips ~56%.
- Q5 probing contrast FAIL: 1 of 4 anchor tasks clears (needs 2). capital-
  operand band-min L13 jlens HMR 1.54 vs logit 19.73 (12.8x) PASS; capital-
  recall L15 1.38 vs 1.53 (1.1x), opposites L14 1.00 vs 1.00, word-pairs L16
  1.60 vs 1.51 (0.9x) -- the logit lens reads those three as well as the
  J-lens. Scanning all 23 layers (not only the band), the logit-to-jlens HMR
  ratio never exceeds ~1.3 for those three tasks.
- Q6 draw stability FAIL: dp IQR 0.0707 > 0.05 (band-min HMR IQR <= 0.065,
  fine); driven by draw1 as a low outlier (dp 0.353 / flip 0.375 vs ~0.49 /
  0.56 on draws 0/2).

Not a pipeline artifact: all 3 draws pass Q1, the swap effect is strong and
per-item structured, and J-lens HMRs are low (~1-2) where expected. Contrast
with 410M (M1 VERIFIED: capital-recall logit HMR 61.5 at L16 vs J-lens 2.5):
on 1.4B the logit lens itself reaches HMR ~1-2 on capital-recall/opposites/
word-pairs in the upper-mid layers. Whether that is a genuine scale effect
(the J-lens advantage over the logit lens narrowing as the residual stream
becomes more logit-readable with scale) is OPEN and HYPOTHESIS-tier -- one gate
run, not concluded. It touches the pre-registered deflation branch in
TAXONOMY_DESIGN (whether the J-lens indexes output-proximity rather than
representation type); the A1/A3 axes require the J-lens to separate from the
logit lens, which 1.4B did not exhibit here on 3 of 4 anchors.

Deviations / notes:
- Band not re-derived (conformance gap). EXP-M5-0 rule 5 specifies the band is
  re-derived from the probing profile per model; scripts/m5_lens_gate.py used
  the config's fixed [4,16] (the 410M PROVISIONAL band). Verified above (full-
  profile scan) that this did NOT cause the FAIL -- the logit lens is
  competitive at every layer for the three failing tasks, so a re-run with a
  re-derived band would still FAIL Q5. Flagged for the record.
- Resource reality (retracts the earlier D-027 "12h-LAW breach" alarm I raised
  mid-run): actual compute was fits 6877/5385/4817 s (1.91/1.50/1.34 h) +
  evals ~16 min each + baselines ~0.2 h ~= 5.75 h total, within the ~7-8 h
  prereg estimate and under the 12 h LAW. Wall-clock was 14.5 h (02:48->17:16)
  only because the laptop slept ~07:12-13:25 while unattended; caffeinate
  (bound to the run PID) held it awake afterward. No compute-budget deviation.

Disposition PUT TO ECATERINA (D-027, proposed; the prereg failure clause makes
1.4B inadmissible for the lens-readout axes A1/A3 and calls for an escalation
ruling). Options, none adopted:
(a) escalate the primary substrate to Pythia-2.8B (TAXONOMY_DESIGN scope
    allows; compute flag ~2.8x the 1.4B per-fit cost -> GPU tier likely);
(b) treat the J-lens vs logit convergence at 1.4B as a first-class question and
    design a dedicated measurement before escalating (it may be a deflation or
    a scale-effect result in its own right);
(c) revisit the Q-rule thresholds for larger models -- flip 0.75 and the 5x
    logit-contrast were set from 410M anchors -- a prereg amendment, her call;
(d) 410M stays lens-admitted (M1-certified) and can anchor S1/S2 while the
    1.4B question is resolved.
Non-lens axes (A2 potency vs sham) are unaffected by a lens-gate FAIL. No
further lens-dependent M5 work on 1.4B until she rules.

---

## 2026-07-20 — Scout: multi-hop latent-composition variance pilot (D-028) (Claude)

D-028 (ruled by Ecaterina via the session instruction; second scout-tier
authorization on the D-024 pattern): single-run, post_hoc-stamped,
results-scout/ only, HARD-BANNED from CLAIMS.md and from findings language.
Purpose — a go/no-go input for the (not-yet-in-repo) "relgraph" spinoff, not a
claim: price whether *latent* (no-CoT) 2-hop success is VARIABLE at a feasible
Pythia scale, conditional on both constituent 1-hops passing. If latent
multi-hop were uniformly failed or uniformly passed, relgraph's key assumption
dies and the taxonomy proceeds alone. This session priced that assumption; it
built no relgraph.

Setup: scripts/scout_multihop.py (this session, uncommitted->committed with
this entry) builds 2-hop compositions from third_party/relations (D-022 vendor)
with a latent bridge entity — 7 relation pairs across 3 bridge types (country
x4, company x2, person x1), 265 items; every item also tests both constituent
1-hops on the same entities, plus a no-bridge paraphrase (bridge stated) format
control. Measures: greedy exact-match (word-prefix; lenient window flag logged
too) + correct-answer log-prob; zero-shot and 4-shot; a shuffled-gold frequency
control (re-score of the greedy 2-hop gens). Substrates pythia-1.4b@fedc38a
(D-023 pin) and pythia-2.8b@2a259cdd96a4beb1cdf467512e3904197345f6a9 (main
resolved 2026-07-20 via HfApi; recorded scout-only, NOT a ratified pin — a pin
decision stays Ecaterina's), fp16 inference at scout tier (dtype in each
manifest). RTX 2000 Ada laptop; ~16 min total compute, peaks 2.88 / 5.64 GB,
under the 12 h LAW. Evidence:
results-scout/20260720-200644-multihop-variance/ (battery.json, per-item
results.jsonl, raw completions, verdict.json + verdict_table.md, SCOUTLOG.md).
Parent commit d7f4f2b.

Scan verdict (scout label, not a finding): VARIANCE-EXISTS at 4-shot on both
scales (strict and lenient matcher); MIXED-INCONCLUSIVE zero-shot on both.
Among items where both 1-hops pass, the four country-bridge landmark pairs are
admissible (n_both 22-30, frequency control ok) and land inside [0.2,0.8]:
cond-2hop capital 0.375/0.28, currency 0.39/0.43, language 0.68/0.80,
largest_city 0.50/0.46 (1.4b/2.8b). Ordering is stable across scale (most
compositional = language; least = capital). The no-bridge control stays high
(0.65-1.0, incl. zero-shot) on the same items, so the drop tracks the LATENT
bridge, not prompt format. The shuffled-gold control excluded product->hq
(golds 92% one value; real==shuffled EM) and product->ceo; father->mother had
no both-pass items at either scale. Zero-shot is thin because obscure-landmark
hop-A mostly fails there.

Disposition for Ecaterina: this scout scan is a go-signal input for relgraph's
precondition, confined to the country bridge; it is not evidence for any
taxonomy claim and opens no CLAIMS entry. No number here is quoted outside the
scout report. Any relgraph follow-up (or an admissible non-country bridge)
would earn its own prereg + gate treatment. [scout tier; no sign-off implied]
### 2026-07-20 — EXP-M5-0b diagnostic: GAP-RETURNS; D-027 resolves to miscalibration (Claude)

Run results/m5/20260720-215157-p14b-lens-diagnostic (prereg c9f2acd; probe-only
on the 3 cached M5.0 lenses, no refit; wall ~1h40m, peak RSS 5.64 GB).
diagnostic_verdict = GAP-RETURNS (2/2 fresh matched pairs dissociate). Metric =
max-contrast layer ratio applied identically to jlens / logit / random arms
(ratified amendment); median over 3 draws.

Fresh matched battery (decision-bearing; latent vs output on identical prompts):

| fresh task | probe | jlens ratio median | N | clears 5x bar |
|---|---|---|---|---|
| fresh1hop-operand | latent (country) | 6.98x | 28 | yes |
| fresh1hop-answer | output (capital) | 0.89x | 28 | no |
| fresh2hop-bridge | latent (country) | 15.52x | 6 | yes |
| fresh2hop-answer | output (capital) | 0.98x | 6 | no |

Every random-arm control ratio = 0.0 (the identical statistic gives noise no
advantage; the amendment guard holds). The 2-hop pair is underpowered (N=6, the
1.4B model does 6/12 fresh 2-hop) but the effect is large enough to be
unambiguous; the 1-hop pair (N=28) carries the decision on its own.

Post-hoc context (existing anchors, labelled post-hoc; jlens ratio median):
capital-operand 27.06x, capital-recall 20.35x, multihop-scaled 4.27x,
opposites 1.00x, word-pairs 1.45x. Note capital-recall (an OUTPUT-token probe)
is recovered to 20.35x here vs the gate's 1.1x under the withdrawn
band-min-jlens metric — the max-contrast metric surfaces the ~2-layer J-lens
lead the gate stepped past. opposites/word-pairs stay ~1x under either metric
(simple associative tasks, no J-lens-specific lead).

Raw replay (read-only): fresh1hop-operand draw0 hand-recomputes to 6.81x @L22
(jlens HMR 2.86 vs logit 19.46); at the same L22 the matched output answer is
jlens 1.15 vs logit 1.12 (ratio 0.97). Same prompts, same layer: the J-lens
reads the held operand far better than the logit lens, the answer no better.
Pipeline verified.

Interpretation (scoped; a methodological diagnostic, NOT a residency finding):
on Pythia-1.4b@fedc38a, the M5.0 lens-gate FAIL was substantially a metric +
anchor miscalibration, not a J-lens/logit convergence. Under the max-contrast
metric the J-lens has a robust, 3-draw-replicated advantage on
latent-intermediate probes that is absent on matched output probes. This is
D-027 outcome (c): amend the EXP-M5-0 Q5 probing-contrast (max-contrast metric
+ latent-intermediate anchors), then admit 1.4B. Amendment drafted UNCOMMITTED
(EXP-M5-0 amendment below / harness) for Ecaterina's ratification — the prereg
amendment is the prereg act; 1.4B admission follows her ruling. Non-lens axes
were never in question. 2.8B escalation is NOT triggered (the deflation branch
did not fire).

### 2026-07-21 — Q5 amendment ratified; latent-vs-output specificity (A1); Q2/Q6 blocker (Claude)

Ecaterina ruled (session, 2026-07-21):

- Q5 amendment RATIFIED with two riders, folded into
  harness/preregs/EXP-M5-0-amendment-Q5.md and committed: (a) the
  fair-statistic clause (all baselines under the identical max-contrast
  statistic) is a PERMANENT part of Q5 on every substrate; (b)
  latent-intermediate probes are, by definition, the A1-relevant anchor class
  (an output-token probe cannot answer A1, since the logit lens reads the
  emitted token by construction).

- Substantive diagnostic outcome (ruling 4; HYPOTHESIS tier, flagged for the
  paper's A1-definition section): on Pythia-1.4b@fedc38a, EXP-M5-0b, the
  J-lens's max-contrast advantage over the logit lens is SPECIFIC to
  latent-intermediate readout. On identical prompts, the held operand/bridge is
  read by the J-lens far better than by the logit lens (operand 6.98x, bridge
  15.52x; jlens HMR ~2-3 vs logit ~15-20), while the emitted answer is read no
  better by the J-lens than the logit lens (~0.9x); random baselines 0.0 under
  the identical statistic. This latent-vs-output specificity is the proposed
  operational definition of A1 (decodability = the workspace holds content the
  logit lens does not surface). Stays HYPOTHESIS-tier: one substrate, one gate,
  a methodological diagnostic — not a residency finding (that is M6 on certified
  species). Not to be stated as a finding in the paper until confirmed.

- Formal re-run ORDERED (~40 min, EXP-M5-0-labelled, amended criterion, proper
  anchor N): 1.4B admitted on that PASS, not on the re-graded diagnostic.

- D-029 (proposed; BLOCKS the re-run's PASS): the original gate
  (results/m5/20260720-024819-p14b-lens-gate) failed on Q2 and Q6 as well as
  Q5. The amendment fixes Q5 only. Amended Q5 now passes, but Q2 (swap dp 0.483
  clears the 0.30 dp bar, but top-1 flip 0.5625 misses the 0.75 bar) and Q6 (dp
  IQR 0.0707 > 0.05, draw-1 outlier) still fail — both swap-intervention rules,
  untouched by the probing-contrast amendment. So a re-run today returns FAIL
  and does NOT admit 1.4B. Options put to Ecaterina, none adopted: (i)
  recalibrate the Q2 flip / Q6 IQR bars per substrate on the same evidence-based
  footing as Q5 (the swap moves probability strongly, dp 0.48; a 410M-derived
  argmax-flip bar and a 0.05 IQR bar may be too brittle for 1.4B — parallel to
  Q5's band-min brittleness); (ii) investigate the swap's per-draw weakness on
  1.4B (real, and then A2/A3-relevant, or artifact); (iii) scope this admission
  to A1 (Q5) and hold the swap-dependent A2/A3 admission pending Q2/Q6. The
  formal re-run is built-ready and HELD pending this ruling — running it now
  only re-confirms the predetermined Q2/Q6 FAIL.

### 2026-07-21 — D-029 ruled (scoped admission); 2.8B pin; EXP-M5-0c drafted (Claude)

- D-029 RULED (Ecaterina, 2026-07-21) — option (iii) + gated diagnostic:
  EXP-M5-0 gate admission is scoped PER AXIS CLASS. Amended-Q5 PASS admits
  Pythia-1.4b@fedc38a for A1 and A4 work only (lens-readout / decodability +
  report-coupling). Q2 and Q6 remain BLOCKING for A2/A3 admission (swap-potency
  / basis-mediation). No Q2/Q6 bar change without an EXP-M5-0c verdict on record
  first; if 0c finds a genuinely reduced swap gap-shift at 1.4B (vs a
  flip-rate/base-margin confound), that is a potency-scaling observation,
  HYPOTHESIS tier, NOT a gate failure. Amendment recorded in
  harness/preregs/EXP-M5-0-amendment-Q5.md; committed.

- D-030 (2.8B pin RATIFIED, Ecaterina 2026-07-21): pythia-2.8b@2a259cd is the
  pinned 2.8B checkpoint for ALL tiers (supersedes the D-028 scout's scout-only
  use of the same SHA). Use this revision for every 2.8B config going forward,
  mirroring the 1.4B @fedc38a (D-023) pattern.

- EXP-M5-0c drafted (harness/preregs/EXP-M5-0c-swap-decomposition.md) as
  DRAFT-AWAITING-RATIFICATION (per Ecaterina's explicit instruction to commit
  the draft so the RTX session can access it — NOT the prereg act; her
  ratification line makes it active). Design: swap-intervention decomposition,
  410M vs 1.4B, matched swap tasks, sham twins, >= 3 lens draws median/IQR;
  separates the sham-controlled logit-gap-shift distribution from the top-1
  flip rate; includes a margin-normalized flip statistic as the candidate
  recalibrated Q2 metric; positive + negative instrument controls; identical
  statistic across substrates; tiered for the RTX (~1.5-2.5 h). Configs
  m5_0c_swap_pythia{410m,1p4b}.yaml drafted. The RTX session builds + runs 0c
  after Ecaterina's ratification.

- Parallel-tier note: the RTX session (D-028 scout, and 0c next) runs
  independently on the win32 RTX laptop; this Mac session integrated the D-028
  scout commit (rebased, both LABNOTES entries kept). The RTX 0c run needs only
  these commits pushed to origin — no Mac dependency.

### 2026-07-21 — verify-LAW amended; CLM-003/004 verified; EXP-M5-0c ratified (Claude)

Ecaterina ruled (session, 2026-07-21):

- Verify-LAW amended (CONSTRAINTS.md): the by-hand re-derivation burden is
  lifted. The AI lays out the raw completions (>= 20/cell) and its
  re-derivation; Ecaterina reviews and confirms; the AI may transcribe the
  verify line on her explicit confirmation. The human confirmation remains the
  gate; sign-off lines stay Ecaterina's own to type. The prior informal
  CLAIMS.md edit (verified-by "me :3"/"^-^") was stashed and dropped in favour
  of conformant verification below.

- CLM-003 and CLM-004 CONFIRMED verified by Ecaterina ("im good", 2026-07-21),
  on the AI re-derivation laid out this session (both re-derive from raw:
  CLM-003 exec 0.920->0.000 effect +0.920 vs sham +0.020 / report effect -0.638
  vs sham +0.037, N_exec=50/N_report=80; CLM-004 task-B 0.000 -> lens 0.933 /
  direct 0.800, random 0.000, task-A -> 0.000, N=30, lens-direct gap 0.133).
  Both promoted to verified in CLAIMS.md.

verify: CLM-003 raw-read: 50 re-derived: yes verified-by: Ecaterina date: 2026-07-21
verify: CLM-004 raw-read: 30 re-derived: yes verified-by: Ecaterina date: 2026-07-21

- EXP-M5-0c RATIFIED (Ecaterina, 2026-07-21, via session instruction): the
  swap-decomposition prereg is now the prereg act; the DRAFT marker is cleared.
  The RTX session may build + run it. Thresholds ratified as drafted.

### 2026-07-21 — M5.0 qualification: 410M complete; 1.4B FV/LRE -> RTX (D-031, Claude)

EXP-M5-0 qualification run `results/m5/20260721-033359-qualification` (start_run,
prereg committed, clean tree; raw retained; both substrate configs copied in).
410M ran to completion; on 1.4B, Pass A (S1 + binding) completed and Pass B
(FV + LRE) was killed mid-run.

D-031 RULED (Ecaterina, session, 2026-07-21): the 1.4B qualification Pass B was
thrashing swap on the 16 GB Mac (vm.swapusage 21.6/22.5 GB used, ~0.9 GB free;
per-item eval time climbing ~5 s -> 30 s; projected 8-10+ h remaining and rising,
with a live OOM/stall risk). This is the swap-risk the M5-kickoff plan flagged
for 1.4B fp32 on 16 GB. Presented as a flagged decision; Ecaterina chose "kill
it, move 1.4B to RTX" (heavy 1.4B compute is the RTX/GPU tier per the standing
compute rulings). No decision-rule or threshold change — a compute-placement
execution of the existing rulings.

Actions taken:
- Killed the run (python + uv + the bound caffeinate); swap released.
- `qualification.json` reconstructed from the retained on-disk raw cells (not the
  log; script in scratchpad), covering 410M (complete) + 1.4B Pass A, with 1.4B
  FV/LRE marked `deferred-to-rtx`. A `note` field records the kill + provenance.
- Orchestrator `scripts/m5_0_qualification.py` made substrate-selectable
  (`parse_selection`: no args = both Mac substrates; `name=config_path` overrides
  the config) so the RTX runs the FULL 1.4B qualification on cuda in one clean
  self-contained run dir. Added `configs/m5_0_qual_pythia1p4b_cuda.yaml`
  (device: cuda, dtype float32, D-023 pin, no lens cache needed — qualification
  is forward-pass only). Landing test + full suite green (166 passed),
  validators 3/3.

Qualification results on the Mac (scope: greedy exact-match, seeds per config;
these are admission GATES, not CLAIMS — no verify line):

- pythia-410m@9879c9b (COMPLETE):
  - S1 5/8 admitted (>=0.80): capital-recall, capital-operand, swap-capitals,
    opposites, word-pairs; below bar: typo-robustness 0.70, context-binding 0.533,
    multihop-scaled 0.50.
  - FV 2/3 (>=0.80, 10-shot): capitalize 0.929, singular-plural 0.861 admitted;
    english-french 0.474 below.
  - LRE 4/12 passing (>=0.60): country-capital 0.643, adj-antonym 0.60,
    verb-past 0.76, word-first-letter 0.98 -> S3 NOT admitted (needs >=8). 410M
    is too weak for the LRE operator battery, as expected at this scale.
  - Binding: bind2 0.617, bind3 0.45 -> S4 NOT admitted (bar 0.70).
- pythia-1.4b@fedc38a (PASS A only; FV/LRE on RTX):
  - S1 5/8 admitted (same five as 410M); below bar: typo-robustness 0.767,
    context-binding 0.533, multihop-scaled 0.625.
  - Binding: bind2 0.65, bind3 0.667 -> S4 NOT admitted (bar 0.70).

Observation worth recording (not over-read): the binding battery (S4) clears its
0.70 admission bar on NEITHER model at N=60 (410M bind2 0.617, 1.4B bind2 0.65).
If the RTX 1.4B FV/LRE confirms the expected S3 admission but binding stays sub-bar,
S4 admission (and thus H5) needs either a stronger substrate (2.8B, D-030) or a
binding-battery re-spec before it can be measured. Flagged for Ecaterina; no action
without a ruling.

Next: RTX runs `scripts/m5_0_qualification.py pythia-1.4b=configs/m5_0_qual_pythia1p4b_cuda.yaml`
after these commits land -> one clean 1.4B qualification run dir (supersedes the
Mac Pass A partial). The Mac session picks it up and folds it into the matrix.
