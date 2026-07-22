# EXP-M5-1b S1 concept-gate diagnostic (410M)

- model EleutherAI/pythia-410m@9879c9b; ladder [8, 16, 32, 64, 128, 256]; alphas [1.0, 2.0, 4.0, 8.0]; N_eval 200
- issues NO certificate (diagnostic); per-concept convergence x potency below

| concept | convergence | conv_at | cos@256 | potency | best Δp | neg |
|---|---|---|---|---|---|---|
| Paris | ceiling-limited | None | 0.9737 | unmeasurable-potency | +0.003 | FAIL |
| London | crossed | 128 | 0.9848 | unmeasurable-potency | +0.001 | ok |
| Rome | ceiling-limited | None | 0.9739 | unmeasurable-potency | +0.002 | ok |
| Berlin | ceiling-limited | None | 0.9759 | unmeasurable-potency | +0.000 | ok |
| Madrid | ceiling-limited | None | 0.9738 | sub-bar | +0.006 | ok |
| Vienna | crossed | 128 | 0.9832 | unmeasurable-potency | +0.000 | ok |
| Athens | crossed | 128 | 0.9825 | unmeasurable-potency | +0.000 | ok |
| Cairo | crossed | 128 | 0.9849 | unmeasurable-potency | +0.000 | ok |

outcome tally: {'ceiling-limited/unmeasurable-potency': 3, 'crossed/unmeasurable-potency': 4, 'ceiling-limited/sub-bar': 1}

wall 7113.4 s; peak 2.57 GB. raw under raw_completions/.