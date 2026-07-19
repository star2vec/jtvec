# EXP-M4-E2 report: execution/verbalization dissociation on singular-plural

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M4-E2-dissociation.md (constants D-017)
- clean: execution 0.920 (N=50); report_score +0.389 (P3, N=80, baseline -4.625)
- ablations: fv-direction (3 certified FV draws), jspace (3 M1 lens draws); each vs matched sham; paired context sets across conditions
- decision (D-017): an ablation hurts a measure iff effect_median - sham_median >= delta (exec 0.15, report 0.1)

## Effects (clean - ablated; median [IQR] over 3 draws, vs sham)

| ablation x measure | effect med | effect IQR | sham med | effect-sham | hurts? |
|---|---|---|---|---|---|
| fv x exec | +0.920 | 0.000 | +0.020 | +0.900 | YES |
| fv x report | -0.638 | 0.017 | +0.037 | -0.675 | no |
| jspace x exec | +0.440 | 0.160 | +0.000 | +0.440 | YES |
| jspace x report | +0.341 | 0.155 | +0.295 | +0.046 | no |

## Verdict

- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E2-dissociation (m4_e2_dissociation_pythia410m.yaml), N=50: E2 singular-plural: fv-ablation hurts execution=True report=False; jspace-ablation hurts report=False execution=True; verdict ONE-WAY (cross-draw transfer fv_exec=True, jspace_report=False) = 0.9

**E2 verdict: ONE-WAY** (direction1 fv-exec-not-report=True, direction2 jspace-report-not-exec=False)

wall-clock 86.0 s; peak RSS 3.37 GB; device cuda; grids in e2_results.json; raw per-item cells under raw_completions/.
