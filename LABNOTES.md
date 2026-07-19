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
