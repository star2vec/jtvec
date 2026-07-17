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

## Deliberately NOT vendored yet (build order)

- `jvec/fv.py`, `jvec/evals/fvprobe.py`, `jvec/evals/fvswap.py`,
  `third_party/function_vectors` — M2, behind the stability gate
- `jvec/decompose.py`, `jvec/evals/exp3.py`, `tests/test_decompose.py`,
  scripts 05-15 — M2/M3 material
- v1's untracked `scripts/15_fv_stability.py` — design input for M2 only,
  never vendored (D-001)

## v2-original files (not from v1)

`jtvec/` (all of it), `scripts/m1_gate.py`, `configs/m1_pythia410m_draw*.yaml`,
`harness/`, CI, and the ledgers.
