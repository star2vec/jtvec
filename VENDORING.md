# Vendored code manifest

Per LABNOTES decision D-001: working validated v1 code is vendored, not
rewritten. Source: `~/Developer/jvec-outdated` at commit `3bb6d2a`
(files taken via `git show 3bb6d2a:<path>`, byte-identical — verified by
diff at vendor time; the package keeps its original name `jvec` and repo
layout so `REPO_ROOT`-relative paths and the jlens submodule path resolve
unchanged).

## Vendored at M1 (lens pipeline)

- `jvec/`: `__init__.py`, `calibration.py`, `config.py`, `lens_cache.py`,
  `modeling.py`, `report.py`, `utils.py`
- `jvec/evals/`: `__init__.py`, `baseline.py`, `controls.py`, `probe.py`,
  `swap.py`, `tasks.py`
- `tasks/`: all 8 task JSONs
- `scripts/`: `01_fit_lens.py`, `02_task_baselines.py`, `03_run_evals.py`,
  `04_report.py`, `make_tasks.py`
- `tests/`: `test_skeleton.py`, `test_evals.py`
- `configs/`: `gpt2_phase1.yaml`, `pythia410m_phase1_v1reference.yaml`
  (v1's `configs/pythia410m_phase1.yaml`, renamed)
- `third_party/jacobian-lens`: git **submodule** pinned at `581d398`
  (github.com/anthropics/jacobian-lens) — the exact commit in v1's lens
  manifests; kept as a git checkout because `jvec.utils.jlens_commit()`
  reads it and the hash is a lens-identity key (decision D-004).

## Vendored at the M1->M2 boundary (D-007: portability for laptop switch)

All remaining v1 sources were vendored byte-identical from `3bb6d2a` so v2
is self-contained and the v1 checkout is no longer needed on any machine.
**Vendored does not mean in use**: each item below stays inert until its
milestone is signed off (build-order rule unchanged).

- `jvec/fv.py` (Todd wrapper + transformers-5 hook patch),
  `jvec/decompose.py`, `jvec/evals/{fvprobe,fvswap,exp3}.py` — M2/M3/M4
- `scripts/05`-`14` (incl. `11b`) — M2/M3/M4
- `tests/test_decompose.py` — runs in CI now (model-free)
- `configs/*_v1reference.yaml` — v1 configs kept for reference
- `third_party/function_vectors`: git **submodule** pinned at `fb9eac7`
  (github.com/ericwtodd/function_vectors, clean public checkout —
  `jvec.fv.todd_commit()` reads it, same rationale as D-004)
- `design_input/15_fv_stability_v1_untracked.py`: v1's UNTRACKED stability
  script, preserved verbatim as M2 design reference per D-001 — NOT
  vendored code, never imported (sha256 prefix 70b140d003a15981)

## Vendored at the taxonomy phase (D-022: LRE relation data)

- `third_party/relations`: git **submodule** pinned at `1b9ec3c`
  (github.com/evandez/relations, Hernandez et al. ICLR 2024; MIT, © 2022
  Evan Hernandez). Ruled by Ecaterina 2026-07-20 (D-022, session
  instruction). Supplies the LRE relation battery for EXP-M5-0
  qualification and the S3 operator extractor (M5.2). Data only
  (`data/{factual,linguistic,commonsense,bias}/*.json`: name,
  prompt_templates, samples of subject/object pairs); the repo's own code
  (`src/`, `experiments.py`) is NOT imported — v2 reads the JSON directly,
  same posture as the Todd dataset files. Same pin rationale as D-004: a
  submodule so the data provenance is a recorded commit hash. Never
  advanced.

`jvec/decompose.py` (k=25 gradient-pursuit J-space-fraction) is the
instrument CONSTRAINTS bans (VERIFIED negative: failed its positive
control). It is vendored for provenance and its unit tests, but
`jtvec.core.instruments.BANNED_INSTRUMENTS` blocks its use as evidence
unless rebuilt and re-controlled.

## Post-vendor deviations (each requires a ruled decision; keep this list short)

- `jvec/utils.py` (D-008, ruled by Ecaterina 2026-07-18): the byte-identity
  exception for win32. v1's `import resource` is POSIX-only, so every import
  of `jvec` aborted on Windows (pytest collection error before a single test
  ran). Change, in full: the import is wrapped in try/except ImportError
  (binding `resource = None`), and `peak_rss_gb()` gained a win32 branch
  reading the psapi peak working set via ctypes (new module-level helper
  `_win32_peak_working_set_bytes`). POSIX takes the identical code path v1
  took. No other vendored file is modified. Guard test:
  `tests/test_platform.py`.

## v2-original files (not from v1)

`jtvec/` (all of it), `scripts/m1_gate.py`, `configs/m1_pythia410m_draw*.yaml`,
`harness/`, CI, and the ledgers.
