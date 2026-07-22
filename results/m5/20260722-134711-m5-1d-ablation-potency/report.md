# EXP-M5-1d S1 ablation-potency (410M) — REMOVE arm (Δlogit, Option B)

- model EleutherAI/pythia-410m@9879c9b; band [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; N_eval 30; 3 draws; E2 project-out, sham-ctrl Δlogit
- bars (logit): potent g>=1.0+transfer | inert g<=0.3 | 0.3-1.0 weak | pos-ctrl>=1.0
- **roster verdict: MIXED** (potent 0, inert 1, weak 3, inconclusive 4) — NO certificate

| concept | clean logit | sham-ctrl Δlogit g | transfer | pos-ctrl g | class |
|---|---|---|---|---|---|
| Paris | 18.6497 | +0.537 | n | +2.027 ok | weak-ambiguous |
| London | 15.8914 | +0.370 | n | +2.097 ok | weak-ambiguous |
| Rome | 17.2629 | -0.263 | n | +1.466 ok | ablation-inert |
| Berlin | 17.0433 | +0.344 | n | +1.685 ok | weak-ambiguous |
| Madrid | 19.1744 | +0.786 | n | +0.370 FAIL | inconclusive |
| Vienna | 18.8124 | -0.446 | n | +0.780 FAIL | inconclusive |
| Athens | 15.5291 | -0.752 | n | +0.356 FAIL | inconclusive |
| Cairo | 16.6412 | -0.164 | n | +0.277 FAIL | inconclusive |

mechanism positive control (on record): null-check unembed('Paris') injection Δp=+0.80.

wall 341.4 s; peak 2.4 GB. raw under raw_completions/.