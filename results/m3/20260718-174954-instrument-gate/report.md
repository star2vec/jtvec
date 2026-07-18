# M3 report: intervention-instrument gate

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M3-intervention-instruments.md
- certified FVs: M2 run 20260718-114950-fv-stability-gate, draw 1 (n_trials_aie=200 >= converged_at=25)
- lens: re-materialized on this machine; identity equals M1 draw 0's committed manifest (incl. calibration sha256); functional spot-check band-min jlens HMR 2.50 (L16) vs M1 2.49 (bound 1.5x), logit 61.5 (>= 5.0x separation)
- context rng 9090; sham seeds 0; bounds are quantization-aware, max(base, 1/N) (D-010)

## Instrument verdicts

| instrument | positive | negative | gated |
|---|---|---|---|
| fv-direction-ablation | PASS | PASS | YES |
| jspace-ablation | PASS | PASS | YES |
| report-probe-forced-choice | PASS | FAIL | NO |
| fv-swap | PASS | PASS | YES |

## Control readings (every ablated/swapped number next to its sham)

- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=30: fv-ablation capitalize: exec none 0.933 -> fv 0.000 (sham_fv 0.900, bound 0.050) = 0.9333
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=30: fv-ablation singular-plural: exec none 0.933 -> fv 0.000 (sham_fv 0.933, bound 0.050) = 0.9333
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=36: jspace-ablation capital-recall: exec none 0.861 -> jspace 0.583 (sham_jspace 0.833, bound 0.050) = 0.2778
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=36: report-probe capitalize: explicit-rule detection [P1 1.00, P2 1.00, P3 1.00], random-word null 0.000 vs prior 0.125 (margin 0.150) = 1
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=36: report-probe singular-plural: explicit-rule detection [P1 1.00, P2 1.00, P3 1.00], random-word null 0.722 vs prior 0.125 (margin 0.150) = 1
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=36: report-probe english-french: explicit-rule detection [P1 1.00, P2 1.00, P3 1.00], random-word null 0.000 vs prior 0.125 (margin 0.150) = 1
- on EleutherAI/pythia-410m@9879c9b, m3 instrument gate (m3_pythia410m.yaml), N=30: fv-swap capitalize->singular-plural: b-rate none 0.000, direct 0.933, lens 0.900 (random_target 0.000, bound 0.050) = 0.9333

**M3 verdict: NOT all instruments gated**

wall-clock 31.0 s; peak RSS 3.37 GB; device cuda; ControlRecord pairs in controls.json; raw per-item cells under raw_completions/.
