# M1 report: lens port + 9-check gate reproduction

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/configs)
- prereg: harness/preregs/EXP-M1-lens-gate.md
- v1 reference: jvec-outdated results/phase1_report/20260714-051555
- draws: lens fits at seeds [0, 1, 2] (independently re-sampled calibration prompts); task baselines are deterministic and shared
- controls: per-layer 10-seed Frobenius-matched random-matrix arm (probing), random-unit-direction swap arm with matched edit energy (10 seeds/item), logit-lens comparator

## Decision rules (tolerances preregistered)

| rule | outcome |
|---|---|
| R1_draw0_gate_pass | PASS |
| R2_swap | PASS |
| R3_capital_recall_contrast | PASS |
| R4_calibration_hashes_exact | PASS |
| R5_baselines | PASS |
| R6_draws_stable | PASS |

**M1 verdict: PASS (R1-R6)**

## Headline numbers, median/IQR over the 3 draws

- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=16: dp(swap_answer), swap-capitals = median=0.6046, IQR=0.02549, n_draws=3 (sham: median=0.00862, IQR=0.00466, n_draws=3)
- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=16: swap top-1 flip rate = median=0.875, IQR=0.0625, n_draws=3
- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=36: band-min J-lens HMR, capital-operand = median=2.57, IQR=0.47, n_draws=3
- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=36: band-min J-lens HMR, capital-recall = median=2.64, IQR=0.075, n_draws=3
- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=16: band-min J-lens HMR, opposites = median=1.32, IQR=0.095, n_draws=3
- on EleutherAI/pythia-410m@9879c9b, skip4_n10 (configs/m1_pythia410m_draw*.yaml), N=24: band-min J-lens HMR, word-pairs = median=2.69, IQR=0.145, n_draws=3

## Draw-0 vs v1 reference

| quantity | v1 | draw 0 |
|---|---|---|
| dp(swap_answer) | +0.6046 | +0.6046 |
| dp random ctrl | +0.0086 | +0.0086 |
| top-1 flip rate | 87.5% | 87.5% |
| capital-recall band-min J-lens HMR | 2.5 | 2.49 (L16) |
| logit HMR at that layer | 61.5 (L16) | 61.48 |
| calibration sha256 (10 prompts) | — | identical to v1 |

| task | v1 baseline | draw 0 baseline | included |
|---|---|---|---|
| capital-operand | 86.1% (in) | 86.1% | True |
| capital-recall | 86.1% (in) | 86.1% | True |
| context-binding | 53.3% (out) | 53.3% | False |
| multihop-scaled | 50.0% (out) | 50.0% | False |
| opposites | 100.0% (in) | 100.0% | True |
| swap-capitals | 93.8% (in) | 93.8% | True |
| typo-robustness | 70.0% (out) | 70.0% | False |
| word-pairs | 91.7% (in) | 91.7% | True |

| task | v1 band-min J-lens HMR | draws 0/1/2 |
|---|---|---|
| capital-operand | 2.6 | 2.57/1.9/2.84 |
| capital-recall | 2.5 | 2.49/2.64/2.64 |
| opposites | 1.3 | 1.32/1.2/1.39 |
| word-pairs | 2.7 | 2.69/2.44/2.73 |

## Provenance

| draw | seed | fit wall-clock (s) | peak RSS (GB) | gate |
|---|---|---|---|---|
| 0 | 0 | 919.3 | 1.8 | PASS |
| 1 | 1 | 945.8 | 2.85 | PASS |
| 2 | 2 | 944.9 | 2.85 | PASS |

jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e (submodule); per-draw reports, eval JSONs (per-item records), and lens manifests are under draws/; raw per-item cells under raw_completions/.
