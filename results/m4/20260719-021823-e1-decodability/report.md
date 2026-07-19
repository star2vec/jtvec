# EXP-M4-E1 report: FV label decodability (J-lens vs logit lens)

- model: EleutherAI/pythia-410m@9879c9b (full sha in run.json/config)
- prereg: harness/preregs/EXP-M4-E1-decodability.md (constants D-014)
- certified FVs: M2 run 20260718-114950-fv-stability-gate, draws 1-3 (n_trials_aie=200 >= converged_at=25); Hendel out of scope (no certificate)
- lens instances: M1 draws 0-2 (identity == committed M1 manifests) + skip16_n10 + skip4_n5 (in-run controls only)
- random seeds 1000-1099; rank statistics are invariant to the norm-matching rescaling (companion norm cells recorded)

## Per-task verdicts (C1-C4)

| task | C1 jlens med | C3 logit med | C2 cells | C4 beaten/100 | verdict |
|---|---|---|---|---|---|
| capitalize | 278 | 6563 | 33 (0 n/e) | 79/80/80 | NOT-DECODABLE |
| singular-plural | 436 | 3203 | 33 (0 n/e) | 80/79/77 | NOT-DECODABLE |
| english-french | 56 | 114 | 33 (0 n/e) | 96/97/95 | NOT-DECODABLE |

## Scoped readings

- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E1 (m4_e1_pythia410m.yaml), N=9: E1 capitalize: jlens label-rank median=278, IQR=485, n_draws=9 vs logit median=6563, IQR=137.5, n_draws=3; random beaten 79/80/80/100 per draw; verdict NOT-DECODABLE = 278
- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E1 (m4_e1_pythia410m.yaml), N=9: E1 singular-plural: jlens label-rank median=436, IQR=261, n_draws=9 vs logit median=3203, IQR=930, n_draws=3; random beaten 80/79/77/100 per draw; verdict NOT-DECODABLE = 436
- on EleutherAI/pythia-410m@9879c9b, EXP-M4-E1 (m4_e1_pythia410m.yaml), N=9: E1 english-french: jlens label-rank median=56, IQR=31, n_draws=9 vs logit median=114, IQR=22.5, n_draws=3; random beaten 96/97/95/100 per draw; verdict NOT-DECODABLE = 56

**E1 outcome: 0/3 tasks DECODABLE-AND-SEPARATED; CLM-001 stays hypothesis per the preregistered rule (pending the post-run evidence commit)**

wall-clock 2299.3 s; peak RSS 3.34 GB; device cuda; grids in e1_results.json; ControlRecords in controls.json; raw cells under raw_completions/.
