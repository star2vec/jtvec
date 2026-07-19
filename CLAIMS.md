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
- statement: On singular-plural, fv-direction ablation and jspace ablation dissociate task execution from task verbalization per EXP-M4-E2-dissociation (fv cuts execution not report; jspace cuts report not execution), each vs its matched sham, cross-draw over the 3 M2-certified FV draws and 3 M1 lens draws. Tests the CONSTRAINTS execution-vs-verbalization HYPOTHESIS. E2 result (2026-07-19, run below): verdict ONE-WAY, NOT the double dissociation. Direction 1 holds and transfers (fv-ablation execution effect +0.920 vs sham +0.020 all 3 draws; report NOT hurt — it rose, effect -0.638 vs sham +0.037). Direction 2 fails: jspace hurts execution (effect +0.440 vs sham +0.000) and its report effect (+0.341) does not beat its sham (+0.295). Status stays hypothesis per the preregistered rule (only DOUBLE-DISSOCIATION promotes).
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e2_dissociation_pythia410m.yaml, singular-plural, 3 FV draws x 3 lens draws, N_exec=50 / N_report=80
- evidence-commit: none
- prereg: harness/preregs/EXP-M4-E2-dissociation.md
- results-dir: results/m4/20260719-142007-e2-dissociation
- raw-completions: results/m4/20260719-142007-e2-dissociation/raw_completions
- verified-by: none

### CLM-003
- status: preliminary
- statement: On Pythia-410M, projecting out the M2-certified singular-plural function vector at the final position of band layers 4-16 removes task execution (accuracy 0.920 -> 0.000) while NOT reducing the P3 report readout (report_score rose, effect -0.638 vs sham +0.037), robust and near-identical across all 3 certified FV draws (execution effect +0.920, IQR 0, vs sham +0.020). One direction of the execution-vs-verbalization dissociation (Direction 1 of EXP-M4-E2); the reverse (jspace report-specific) was not established.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e2_dissociation_pythia410m.yaml, singular-plural, 3 FV draws, N_exec=50 / N_report=80
- evidence-commit: a1c7cb1
- prereg: harness/preregs/EXP-M4-E2-dissociation.md
- results-dir: results/m4/20260719-142007-e2-dissociation
- raw-completions: results/m4/20260719-142007-e2-dissociation/raw_completions
- verified-by: none

### CLM-004
- status: preliminary
- statement: On capitalize (task A) prompts, swapping the certified FV_A component onto FV_B (singular-plural) at band layers 4-16 redirects the model to produce task-B (plural) outputs above a random-target control, per EXP-M4-E3-swap, cross-draw over the 3 certified FV draws. E3 result (2026-07-19): REDIRECTS-BASIS-AGNOSTIC — task-B rate 0.000 -> lens_swap 0.933 / direct_swap 0.800 (random 0.000), task A suppressed to 0.000, transfers across all 3 FV draws; lens-direct gap 0.133 < 0.15 so the identity is carried by the raw residual direction, not specifically the J-lens basis. Promotion to verified needs Ecaterina's raw-read verify line.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e3_swap_pythia410m.yaml, capitalize->singular-plural, 3 FV draws, N=30 shared queries
- evidence-commit: 0d8b278
- prereg: harness/preregs/EXP-M4-E3-swap.md
- results-dir: results/m4/20260719-151956-e3-swap
- raw-completions: results/m4/20260719-151956-e3-swap/raw_completions
- verified-by: none
