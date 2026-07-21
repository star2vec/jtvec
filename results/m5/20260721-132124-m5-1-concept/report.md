# EXP-M5-1 report: S1 concept-direction stability gate

- model: EleutherAI/pythia-410m@9879c9b; band layers [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; N_eval=40
- prereg: harness/preregs/EXP-M5-1-concept-gate.md (thresholds ratified)
- rule: min pairwise cosine >= 0.95 AND effect IQR <= 0.05; witness rung required

## Per-concept convergence + controls

| concept | converged_at | pos ctrl | neg ctrl | certificate |
|---|---|---|---|---|
| Paris | None | FAIL | pass | — |
| London | None | FAIL | pass | — |
| Rome | None | FAIL | pass | — |
| Berlin | None | FAIL | pass | — |
| Madrid | None | FAIL | pass | — |
| Vienna | None | FAIL | pass | — |
| Athens | None | FAIL | pass | — |
| Cairo | None | FAIL | pass | — |

## Instrument lines (effect with sham)

- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Paris Δp(answer) @rung64 = median=0.00333, IQR=0.00189, n_draws=3 (sham: median=-0.00061, IQR=5e-06, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 London Δp(answer) @rung64 = median=-0.00125, IQR=0.000325, n_draws=3 (sham: median=-0.00176, IQR=2e-05, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Rome Δp(answer) @rung64 = median=-0.00043, IQR=8.5e-05, n_draws=3 (sham: median=-0.00072, IQR=5e-06, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Berlin Δp(answer) @rung64 = median=-0.00084, IQR=1e-05, n_draws=3 (sham: median=-0.00103, IQR=4e-05, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Madrid Δp(answer) @rung64 = median=0.0011, IQR=0.00358, n_draws=3 (sham: median=-0.00012, IQR=0, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Vienna Δp(answer) @rung64 = median=-0.00011, IQR=0, n_draws=3 (sham: median=-0.0001, IQR=1e-05, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Athens Δp(answer) @rung64 = median=3e-05, IQR=1.5e-05, n_draws=3 (sham: median=-1e-05, IQR=5e-05, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, EXP-M5-1 (m5_1_concept_pythia410m.yaml), N=40: S1 Cairo Δp(answer) @rung64 = median=0.0001, IQR=0.000285, n_draws=3 (sham: median=5e-05, IQR=0.0001, n_draws=3)

**S1 species certificate: NOT issued** (0/8 concepts converged, 0/8 controlled)

wall-clock 785.4 s; peak RSS 1.98 GB; device mps; grid in concept_gate.json; raw cells under raw_completions/.
