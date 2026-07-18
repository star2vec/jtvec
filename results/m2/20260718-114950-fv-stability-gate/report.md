# M2 report: FV extraction-stability gate

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M2-fv-stability.md
- draws: 3 independent extraction streams (seeds [1, 2, 3]); datasets, weights, and eval contexts held fixed (0-shot evals are context-deterministic)
- ladder: extraction at n_trials_aie=200; rungs [25, 50, 100, 200] from stored per-trial AIE prefixes
- sham twins: norm-matched random directions at the same layer/position, seeds 9000+10k+rung_index
- instrument controls (fv-induction-readout): positive = 10-shot ICL vs 0-shot separation >= 0.1 per task -> PASS; negative = |median sham gain| <= max(0.02, 1/N_test) at every rung (D-010) -> PASS

## Verdicts (rule: min pairwise cosine >= 0.95 AND gain IQR <= 0.05 at T and every larger rung; largest rung alone does not converge)

| task | converged | converged_at | max-rung-only pass |
|---|---|---|---|
| capitalize | YES | 25 | False |
| singular-plural | YES | 25 | False |
| english-french | YES | 25 | False |

## Cross-draw agreement per rung

### capitalize (zero-shot top-1 0.006, 10-shot 0.929, N=170)

| rung | min cos | min overlap | gain median | gain IQR | sham median | sham IQR | pass |
|---|---|---|---|---|---|---|---|
| 25 | +0.991 | 9/10 | +0.3941 | 0.0059 | +0.0000 | 0.0059 | PASS |
| 50 | +0.997 | 10/10 | +0.3882 | 0.0029 | +0.0000 | 0.0029 | PASS |
| 100 | +0.997 | 10/10 | +0.3882 | 0.0029 | -0.0059 | 0.0029 | PASS |
| 200 | +0.997 | 10/10 | +0.3882 | 0.0029 | +0.0059 | 0.0118 | PASS |

Hendel (descriptive, fixed n_trials_mean=100): flat cosines {'draw1|draw2': 0.9977887868881226, 'draw1|draw3': 0.9982115030288696, 'draw2|draw3': 0.9983659982681274}, edit-layer cosines {'draw1|draw2': 0.9993402361869812, 'draw1|draw3': 0.9994276762008667, 'draw2|draw3': 0.9995895624160767}

### singular-plural (zero-shot top-1 0.023, 10-shot 0.860, N=43)

| rung | min cos | min overlap | gain median | gain IQR | sham median | sham IQR | pass |
|---|---|---|---|---|---|---|---|
| 25 | +0.959 | 8/10 | +0.2093 | 0.0233 | +0.0000 | 0.0233 | PASS |
| 50 | +0.969 | 8/10 | +0.2093 | 0.0000 | -0.0233 | 0.0116 | PASS |
| 100 | +0.992 | 9/10 | +0.2093 | 0.0233 | +0.0000 | 0.0233 | PASS |
| 200 | +0.994 | 9/10 | +0.2093 | 0.0116 | +0.0000 | 0.0000 | PASS |

Hendel (descriptive, fixed n_trials_mean=100): flat cosines {'draw1|draw2': 0.9976858496665955, 'draw1|draw3': 0.997329592704773, 'draw2|draw3': 0.9978551864624023}, edit-layer cosines {'draw1|draw2': 0.9996876120567322, 'draw1|draw3': 0.9996296763420105, 'draw2|draw3': 0.9995211958885193}
Caveat (D-009): N=43 makes the gain criterion coarse (top-1 granularity 0.023); ruled kept-and-flagged.

### english-french (zero-shot top-1 0.007, 10-shot 0.474, N=987)

| rung | min cos | min overlap | gain median | gain IQR | sham median | sham IQR | pass |
|---|---|---|---|---|---|---|---|
| 25 | +0.982 | 8/10 | +0.1246 | 0.0096 | +0.0020 | 0.0041 | PASS |
| 50 | +0.989 | 9/10 | +0.1266 | 0.0035 | -0.0030 | 0.0041 | PASS |
| 100 | +0.989 | 9/10 | +0.1266 | 0.0035 | -0.0041 | 0.0025 | PASS |
| 200 | +0.983 | 9/10 | +0.1307 | 0.0051 | +0.0000 | 0.0030 | PASS |

Hendel (descriptive, fixed n_trials_mean=100): flat cosines {'draw1|draw2': 0.9957625269889832, 'draw1|draw3': 0.9956168532371521, 'draw2|draw3': 0.995875895023346}, edit-layer cosines {'draw1|draw2': 0.9993767738342285, 'draw1|draw3': 0.9995691776275635, 'draw2|draw3': 0.9993826746940613}

## Headline numbers (median/IQR over draws; sham in the same line)

- on EleutherAI/pythia-410m@9879c9b, m2 fv gate (m2_pythia410m.yaml), N=170: capitalize induction gain @T=25 = median=0.3941, IQR=0.005882, n_draws=3 (sham: median=0, IQR=0.005882, n_draws=3)
  | inject | layers=[8] | median=0.3941, IQR=0.005882, n_draws=3 | sham: median=0, IQR=0.005882, n_draws=3 |
- on EleutherAI/pythia-410m@9879c9b, m2 fv gate (m2_pythia410m.yaml), N=43: singular-plural induction gain @T=25 = median=0.2093, IQR=0.02326, n_draws=3 (sham: median=0, IQR=0.02326, n_draws=3)
  | inject | layers=[8] | median=0.2093, IQR=0.02326, n_draws=3 | sham: median=0, IQR=0.02326, n_draws=3 |
- on EleutherAI/pythia-410m@9879c9b, m2 fv gate (m2_pythia410m.yaml), N=987: english-french induction gain @T=25 = median=0.1246, IQR=0.009625, n_draws=3 (sham: median=0.002026, IQR=0.004053, n_draws=3)
  | inject | layers=[8] | median=0.1246, IQR=0.009625, n_draws=3 | sham: median=0.002026, IQR=0.004053, n_draws=3 |

- on EleutherAI/pythia-410m@9879c9b, m2 fv gate (m2_pythia410m.yaml), N=3: M2 gate verdict (1=all tasks converged) = 1

## Provenance

| task | draw | seed | mean-acts (s) | AIE (s) | hendel (s) |
|---|---|---|---|---|---|
| capitalize | 1 | 1 | 4.1 | 1728.3 | 3.2 |
| capitalize | 2 | 2 | 3.8 | 1715.4 | 3.3 |
| capitalize | 3 | 3 | 3.9 | 1717.4 | 3.2 |
| singular-plural | 1 | 1 | 4.0 | 1700.1 | 3.1 |
| singular-plural | 2 | 2 | 3.9 | 1700.5 | 3.3 |
| singular-plural | 3 | 3 | 3.9 | 1697.3 | 3.1 |
| english-french | 1 | 1 | 4.0 | 1751.7 | 3.5 |
| english-french | 2 | 2 | 4.0 | 1749.8 | 3.2 |
| english-french | 3 | 3 | 4.3 | 1734.5 | 3.2 |

wall-clock 1072.4 s; peak RSS 3.34 GB; device cuda; todd_commit fb9eac7b6dc707ea1475a717379916007fe448d5; per-draw FV caches under cache/m2/draw*/fvs/; raw per-item cells under raw_completions/.
