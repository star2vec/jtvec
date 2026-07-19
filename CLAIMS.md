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

### CLM-001
- status: hypothesis
- statement: On the three M2-certified tasks, Todd FVs (fv_todd@task, converged_at=25, n_draws=3) carry a task-label component decodable through the M1-gated J-lens arm and separated from the logit-lens arm, per criteria C1-C4 of EXP-M4-E1. Tests the CONSTRAINTS FV-label HYPOTHESIS. E1 counter-evidence (2026-07-19, run below): NOT-DECODABLE on 3/3 tasks — jlens label-rank medians 278/436/56 vs the C1 bar 20; english-french alone passed the random anchor (96/97/95) and failed the logit floor (logit median 114 < 200); jlens < logit ordering held in 33/33 cells on every task. Status stays hypothesis per the preregistered rule.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e1_pythia410m.yaml, 3 tasks x 3 FV draws x 3 lens draws
- evidence-commit: none
- prereg: harness/preregs/EXP-M4-E1-decodability.md
- results-dir: results/m4/20260719-021823-e1-decodability
- raw-completions: results/m4/20260719-021823-e1-decodability/raw_completions
- verified-by: none

### CLM-002
- status: hypothesis
- statement: On singular-plural, fv-direction ablation and jspace ablation dissociate task execution from task verbalization per EXP-M4-E2-dissociation (fv cuts execution not report; jspace cuts report not execution), each vs its matched sham, cross-draw over the 3 M2-certified FV draws and 3 M1 lens draws. Tests the CONSTRAINTS execution-vs-verbalization HYPOTHESIS; verdict table recorded from the run.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e2_dissociation_pythia410m.yaml, singular-plural, 3 FV draws x 3 lens draws, N_exec=50 / N_report=80
- evidence-commit: none
- prereg: harness/preregs/EXP-M4-E2-dissociation.md
- results-dir: none
- raw-completions: none
- verified-by: none
