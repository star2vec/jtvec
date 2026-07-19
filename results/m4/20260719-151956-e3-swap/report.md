# EXP-M4-E3 report: cross-basis FV swap (capitalize -> singular-plural)

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M4-E3-swap.md (constants D-018)
- clean task-B rate 0.000; swap moves fv_A component onto fv_B at band layers 4-16, final position; cross-draw over 3 certified FV draws
- decision (D-018): redirects iff best-swap gain median >= 0.2 and random gain <= 0.05; J-specific iff lens-direct median >= 0.15

## Task-B answer rate by condition (per FV draw + median)

| condition | draw1 | draw2 | draw3 | median |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 |
| lens_swap | 0.933 | 0.933 | 0.933 | 0.933 |
| direct_swap | 0.800 | 0.767 | 0.867 | 0.800 |
| random_target | 0.033 | 0.000 | 0.000 | 0.000 |

## Verdict

- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E3-swap (m4_e3_swap_pythia410m.yaml), N=30: E3 capitalize->singular-plural: best-swap B-gain median +0.933 (random +0.000); lens-direct gap +0.133; verdict REDIRECTS-BASIS-AGNOSTIC (transfer True) = 0.9333

**E3 verdict: REDIRECTS-BASIS-AGNOSTIC** (redirects=True, J-specific=False, cross-draw transfer=True)

wall-clock 17.5 s; peak RSS 3.37 GB; device cuda; grid in e3_results.json; raw cells under raw_completions/.
