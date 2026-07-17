# CLAIMS.md — claims ledger

Machine-validated by `jtvec/validators/claims.py` on every CI run. Rules
(from the CONSTRAINTS.md LAWs):

- Every claim has a status in {hypothesis, preliminary, verified, withdrawn}
  and an evidence commit.
- The paper (DRAFT.md) may only cite claims at `verified`.
- Promotion to `verified` is blocked unless all of the following hold:
  - `evidence-commit` is a real commit hash;
  - `results-dir` passes the results-directory check (config copy, run
    record, raw completions on disk);
  - every raw-completion cell holds >= 20 records;
  - LABNOTES.md contains Ecaterina's verification line for this claim, of the
    exact form:
    `verify: CLM-NNN raw-read: <n> re-derived: yes verified-by: Ecaterina date: YYYY-MM-DD`

## Entry schema (CLM-000 is the template; validators ignore it)

### CLM-000
- status: hypothesis
- statement: <one observation with scope, e.g. "on Pythia-410M, skip4_n10, N=16, dp(swap) median=…, IQR=… (sham: …)">
- scope: <model@revision, config, N>
- evidence-commit: none
- prereg: none
- results-dir: none
- raw-completions: none
- verified-by: none

## Claims

<!-- none yet; the first entries arrive with M4 preregs -->
