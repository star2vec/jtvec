# EXP-M4-E2 report-gate: prior-corrected report instrument on singular-plural

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M4-E2-reportgate.md
- instrument: report-score-prior-corrected@singular-plural (rebuild of the D-013-withdrawn report-probe-forced-choice@singular-plural, under a new name)
- report_score(ctx) = log p(' plural' | ctx+probe) - neutral baseline; context rng 7070; N=40/cell; neutral pool N=40

## Control arms (bootstrap 95% CIs; gated iff one phrasing does both)

| phrasing | coherent | shuffled | other | positive | negative |
|---|---|---|---|---|---|
| P1 | +0.07 [-0.03,+0.17] | -0.26 [-0.36,-0.15] | +0.01 [-0.14,+0.16] | fail | fail |
| P2 | +0.25 [+0.17,+0.33] | +0.27 [+0.20,+0.34] | +0.03 [-0.08,+0.14] | PASS | fail |
| P3 | +0.36 [+0.29,+0.44] | +0.14 [+0.05,+0.23] | -0.06 [-0.21,+0.08] | PASS | PASS |

## Verdict

- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E2-reportgate (m4_e2_reportgate_pythia410m.yaml), N=40: report instrument GATED on singular-plural under P3: coherent report_score +0.36 [+0.29,+0.44] vs shuffled CI-high +0.23 and other CI-high +0.08 (coherent CI-low +0.29 clears both) = 0.3638

**report-gate verdict: report-score-prior-corrected@singular-plural GATED**

wall-clock 18.4 s; peak RSS 3.32 GB; device cuda; ControlRecord in controls.json; raw per-item cells under raw_completions/.
