# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-410m (revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f)
- device: mps, dtype: float32, seed: 1
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-18T00:39:53, 945.8s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [8, 12, 13]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.6074 vs random +0.0068

## Task baseline gate

Included = in-context top-1 accuracy >= 80%.

| task | protocol | accuracy | items | verdict |
|---|---|---|---|---|
| capital-operand | completion | 86.1% | 36 | **INCLUDED** |
| capital-recall | completion | 86.1% | 36 | **INCLUDED** |
| context-binding | completion | 53.3% | 30 | dropped |
| multihop-scaled | completion | 50.0% | 24 | dropped |
| opposites | completion | 100.0% | 16 | **INCLUDED** |
| swap-capitals | swap | 93.8% | 16 | **INCLUDED** |
| typo-robustness | typo | 70.0% | 30 | dropped |
| word-pairs | completion | 91.7% | 24 | **INCLUDED** |

## Probing eval (rank of the intermediate token in the lens readout)

HMR = harmonic mean rank over items (lower is better); pass@10 = fraction of items with rank <= 10. `random` = mean over Frobenius-matched Gaussian matrices (10 seeds).

### skip4 / capital-operand

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 445.9 | 3234.6 | 2282.0 | 0.00 | 0.00 | 0.00 |
| 1 | 683.9 | 2129.3 | 3630.9 | 0.00 | 0.00 | 0.00 |
| 2 | 276.8 | 3305.3 | 3222.6 | 0.03 | 0.00 | 0.01 |
| 3 | 293.4 | 3182.7 | 3087.2 | 0.03 | 0.00 | 0.00 |
| 4 | 209.0 | 4239.2 | 2965.6 | 0.03 | 0.00 | 0.00 |
| 5 | 235.0 | 3500.7 | 4648.0 | 0.00 | 0.00 | 0.00 |
| 6 | 72.4 | 3626.2 | 4163.4 | 0.08 | 0.00 | 0.00 |
| 7 | 103.2 | 2994.3 | 2686.8 | 0.03 | 0.00 | 0.01 |
| 8 | 182.3 | 1193.3 | 3787.6 | 0.03 | 0.00 | 0.00 |
| 9 | 167.4 | 3013.9 | 2699.9 | 0.00 | 0.00 | 0.00 |
| 10 | 18.0 | 1859.4 | 2765.8 | 0.11 | 0.00 | 0.00 |
| 11 | 175.5 | 2604.2 | 3701.5 | 0.00 | 0.00 | 0.00 |
| 12 | 172.4 | 2670.6 | 3114.4 | 0.00 | 0.00 | 0.00 |
| 13 | 3.0 | 30.5 | 2544.7 | 0.56 | 0.03 | 0.01 |
| 14 | 1.9 | 158.1 | 3952.5 | 0.78 | 0.00 | 0.00 |
| 15 | 10.1 | 91.2 | 1716.6 | 0.22 | 0.03 | 0.00 |
| 16 | 9.4 | 58.8 | 1830.4 | 0.25 | 0.06 | 0.00 |
| 17 | 1.2 | 1.1 | 2573.2 | 1.00 | 1.00 | 0.00 |
| 18 | 1.6 | 1.5 | 3950.4 | 1.00 | 1.00 | 0.00 |
| 19 | 2.3 | 2.5 | 1964.3 | 1.00 | 0.94 | 0.00 |
| 20 | 3.3 | 5.0 | 2024.2 | 0.94 | 0.67 | 0.01 |
| 21 | 4.8 | 30.9 | 1897.1 | 0.83 | 0.11 | 0.00 |
| 22 | 8.0 | 51.6 | 1774.4 | 0.56 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.1 / pass@10 1.00; logit HMR 1.1 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 999.5 | 3864.1 | 2759.8 | 0.00 | 0.00 | 0.00 |
| 1 | 680.1 | 1696.3 | 3217.8 | 0.00 | 0.00 | 0.00 |
| 2 | 673.1 | 3473.4 | 3239.4 | 0.00 | 0.00 | 0.00 |
| 3 | 1622.5 | 3572.8 | 2149.7 | 0.00 | 0.00 | 0.00 |
| 4 | 1716.2 | 3320.2 | 2449.2 | 0.00 | 0.00 | 0.00 |
| 5 | 1181.0 | 2106.8 | 3020.3 | 0.00 | 0.00 | 0.00 |
| 6 | 1740.5 | 4384.9 | 4432.9 | 0.00 | 0.00 | 0.00 |
| 7 | 1352.3 | 2021.2 | 4182.6 | 0.00 | 0.00 | 0.00 |
| 8 | 639.6 | 2897.9 | 3029.3 | 0.00 | 0.00 | 0.01 |
| 9 | 586.9 | 2274.9 | 2670.8 | 0.00 | 0.00 | 0.01 |
| 10 | 650.7 | 203.5 | 2392.5 | 0.00 | 0.03 | 0.00 |
| 11 | 296.4 | 1377.2 | 3105.4 | 0.00 | 0.00 | 0.01 |
| 12 | 28.5 | 562.0 | 3413.4 | 0.03 | 0.00 | 0.00 |
| 13 | 8.3 | 318.5 | 1982.8 | 0.33 | 0.00 | 0.00 |
| 14 | 6.8 | 537.4 | 2357.8 | 0.33 | 0.00 | 0.01 |
| 15 | 4.6 | 231.2 | 2327.7 | 0.28 | 0.00 | 0.01 |
| 16 | 2.6 | 61.5 | 1740.5 | 0.56 | 0.08 | 0.00 |
| 17 | 1.7 | 1.9 | 2131.7 | 0.89 | 0.94 | 0.00 |
| 18 | 1.4 | 1.4 | 2889.0 | 0.89 | 0.92 | 0.01 |
| 19 | 1.3 | 1.2 | 2122.4 | 0.89 | 0.97 | 0.00 |
| 20 | 1.2 | 1.1 | 1753.2 | 0.89 | 0.97 | 0.00 |
| 21 | 1.2 | 1.1 | 1999.4 | 0.92 | 1.00 | 0.00 |
| 22 | 1.2 | 1.1 | 2134.0 | 0.92 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 0.92; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 1177.1 | 9312.7 | 3486.5 | 0.00 | 0.00 | 0.00 |
| 1 | 899.2 | 6446.2 | 5285.2 | 0.00 | 0.00 | 0.00 |
| 2 | 1032.4 | 8306.8 | 5368.1 | 0.00 | 0.00 | 0.00 |
| 3 | 379.8 | 5045.2 | 5986.2 | 0.00 | 0.00 | 0.00 |
| 4 | 333.9 | 7144.9 | 4160.5 | 0.00 | 0.00 | 0.00 |
| 5 | 235.4 | 4939.0 | 3520.7 | 0.00 | 0.00 | 0.00 |
| 6 | 485.9 | 5752.8 | 5641.4 | 0.00 | 0.00 | 0.00 |
| 7 | 163.1 | 4110.1 | 6305.6 | 0.00 | 0.00 | 0.00 |
| 8 | 127.3 | 2062.7 | 6211.6 | 0.00 | 0.00 | 0.00 |
| 9 | 92.0 | 6150.7 | 4471.3 | 0.00 | 0.00 | 0.01 |
| 10 | 10.8 | 216.0 | 5865.6 | 0.19 | 0.00 | 0.00 |
| 11 | 52.1 | 212.8 | 3452.3 | 0.06 | 0.00 | 0.00 |
| 12 | 16.8 | 23.9 | 3492.3 | 0.12 | 0.19 | 0.00 |
| 13 | 1.6 | 1.3 | 5915.1 | 0.88 | 0.94 | 0.00 |
| 14 | 1.4 | 1.4 | 4140.7 | 0.88 | 1.00 | 0.00 |
| 15 | 1.2 | 1.3 | 4687.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.3 | 1.0 | 4950.8 | 1.00 | 1.00 | 0.00 |
| 17 | 1.2 | 1.0 | 5793.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.0 | 1.0 | 4840.9 | 1.00 | 1.00 | 0.00 |
| 19 | 1.0 | 1.0 | 4740.7 | 1.00 | 1.00 | 0.00 |
| 20 | 1.0 | 1.0 | 3693.3 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 4328.3 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 4541.5 | 1.00 | 1.00 | 0.01 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 895.0 | 933.5 | 3186.5 | 0.00 | 0.00 | 0.00 |
| 1 | 1420.6 | 1090.8 | 3879.9 | 0.00 | 0.00 | 0.00 |
| 2 | 690.9 | 1566.1 | 2771.8 | 0.00 | 0.00 | 0.01 |
| 3 | 866.3 | 168.7 | 3388.1 | 0.00 | 0.04 | 0.00 |
| 4 | 459.2 | 159.4 | 2487.8 | 0.00 | 0.00 | 0.00 |
| 5 | 72.1 | 23.1 | 3501.5 | 0.04 | 0.04 | 0.00 |
| 6 | 70.2 | 22.6 | 3684.3 | 0.04 | 0.04 | 0.00 |
| 7 | 46.8 | 22.7 | 3858.8 | 0.04 | 0.04 | 0.00 |
| 8 | 20.1 | 20.5 | 2830.0 | 0.08 | 0.08 | 0.00 |
| 9 | 40.7 | 27.3 | 4453.4 | 0.04 | 0.08 | 0.00 |
| 10 | 20.1 | 13.7 | 3113.0 | 0.17 | 0.12 | 0.00 |
| 11 | 21.6 | 9.4 | 2971.3 | 0.08 | 0.17 | 0.00 |
| 12 | 8.0 | 12.9 | 2125.5 | 0.21 | 0.17 | 0.00 |
| 13 | 3.5 | 4.4 | 2645.9 | 0.42 | 0.33 | 0.00 |
| 14 | 3.6 | 3.5 | 3625.3 | 0.42 | 0.42 | 0.00 |
| 15 | 2.7 | 2.4 | 2938.7 | 0.50 | 0.58 | 0.00 |
| 16 | 2.4 | 1.8 | 2650.1 | 0.58 | 0.67 | 0.00 |
| 17 | 2.3 | 1.9 | 1853.7 | 0.62 | 0.79 | 0.00 |
| 18 | 2.1 | 1.7 | 3189.3 | 0.67 | 0.75 | 0.00 |
| 19 | 1.8 | 1.4 | 2861.3 | 0.79 | 0.83 | 0.00 |
| 20 | 1.7 | 1.4 | 1975.8 | 0.83 | 0.83 | 0.00 |
| 21 | 1.5 | 1.4 | 1352.8 | 0.88 | 0.79 | 0.00 |
| 22 | 1.2 | 1.2 | 2831.1 | 0.96 | 0.96 | 0.00 |

min-over-layers: J-lens HMR 1.2 / pass@10 0.96; logit HMR 1.1 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.6074 | +0.0068 | -0.7058 | 93.8% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `...; the other compounds were decomposed extensively. Butene was the`)

**skip4** (model's actual next token: `' only'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `  ` ` 
` `
` ` p` ` name` ` table` ` put` ` place` ` entry` ` present` | `manuel` `izabeth` `PI` `imab` `ocardi` `slant` `)\[` `uthan` `urther` `aul` |
| 1 | ` p` ` entry` ` above` `
` ` house` ` table` ` put` ` 
` ` field` ` place` | `PI` `manuel` `ocardi` `)\[` `imab` `uthan` ` thereof` `izabeth` ` Bold` `PR` |
| 2 | ` “` ` p` ` all` ` new` ` access` ` a` ` way` ` place` ` area` ` name` | `PI` `manuel` ` thereof` `extension` `)\[` `urther` `bage` `imab` `,&` `ocardi` |
| 3 | ` tax` ` sales` ` new` ` products` ` following` ` presentation` ` place` ` Tax` ` product` ` places` | `PI` `manuel` `uese` `urther` `)}}{\` `.\"` `-------------------------------------------------` `,&` ` thereof` `;\|` |
| 4 | ` trade` ` sales` ` "` ` tax` ` Pennsylvania` ` public` ` property` ` relationship` ` Jones` ` new` | `PI` `iae` `manuel` ` leukocyte` `rosse` `enes` `itti` `asone` `ritz` `/*!` |
| 5 | ` "` ` public` ` same` ` new` ` development` ` other` ` “` ` American` ` "[` ` unauthorized` | `PI` `same` `ros` ` same` `uds` `ERTY` `eli` `ruck` `IFIC` ` Roche` |
| 6 | ` “` ` $` ` contract` ` same` ` relationship` ` "` ` public` ` general` ` so` ` permanent` | `unnumbered` `PI` `uds` `chief` `ents` `rase` `cci` `ajor` `ros` ` sole` |
| 7 | ` “` ` $` ` sales` ` principal` ` method` ` new` ` only` ` money` ` permanent` ` possession` | `uds` `PI` ` sole` ` preferred` ` only` ` chief` `tec` ` same` ` mean` `ussed` |
| 8 | ` only` ` exception` ` present` ` method` ` most` ` contact` ` same` ` available` ` persons` ` except` | `uds` ` dominant` `domin` `ajor` ` obser` `ussed` ` predominant` ` largest` `/*!` `PI` |
| 9 | ` only` ` most` ` "` ` object` ` not` ` best` ` a` ` exception` ` present` ` person` | ` chief` ` dominant` `domin` ` largest` `uds` ` strongest` `Chief` `те` `inally` `chief` |
| 10 | ` only` ` most` ` same` ` also` ` not` ` difference` ` exception` ` preferred` ` choice` ` following` | ` strongest` ` dominant` ` fastest` ` brightest` ` largest` `arily` `largest` ` preferred` ` biggest` ` predominant` |
| 11 | ` only` ` most` ` choice` ` preferred` ` "` ` chosen` ` form` ` best` ` all` ` use` | ` strongest` ` fastest` ` brightest` ` largest` ` dominant` ` longest` ` biggest` `inally` ` exception` `essages` |
| 12 | ` only` ` most` ` preferred` ` best` ` chosen` ` choice` ` predominant` ` result` ` not` ` all` | ` strongest` ` fastest` `inally` ` largest` ` answ` `essages` ` lapt` ` brightest` ` obser` ` longest` |
| 13 | ` only` ` most` ` exception` ` preferred` ` fastest` ` highest` ` least` ` result` ` next` ` lowest` | ` strongest` ` fastest` ` highest` ` longest` ` preferred` ` largest` ` least` ` exception` ` most` ` lapt` |
| 14 | ` preferred` ` most` ` least` ` fastest` ` only` ` highest` ` preponder` ` longest` ` predominant` ` exception` | ` fastest` ` strongest` ` highest` ` largest` ` longest` ` lowest` ` least` ` exception` ` most` ` easiest` |
| 15 | ` exception` ` only` ` preferred` ` fastest` ` longest` ` predominant` ` ONLY` ` strongest` ` highest` ` preponder` | ` fastest` ` strongest` ` lapt` ` exception` ` preferred` ` longest` ` highest` ` least` ` largest` ` only` |
| 16 | ` strongest` ` fastest` ` longest` ` largest` `least` ` most` ` highest` ` least` ` greatest` ` hardest` | ` most` ` fastest` ` least` ` strongest` ` largest` ` longest` `most` ` greatest` ` highest` ` hardest` |
| 17 | ` longest` ` strongest` ` fastest` `least` ` cheapest` ` largest` ` least` ` hardest` ` poorest` ` smallest` | ` most` ` least` ` strongest` ` fastest` ` only` ` largest` ` highest` ` easiest` ` longest` ` hardest` |
| 18 | ` strongest` ` largest` ` fastest` `least` ` longest` ` poorest` ` hardest` ` least` ` cheapest` ` smallest` | ` most` ` least` ` highest` ` largest` ` only` ` best` ` fastest` ` strongest` ` worst` ` preferred` |
| 19 | ` strongest` ` fastest` ` poorest` ` largest` ` most` ` smallest` ` cheapest` ` hottest` ` hardest` ` highest` | ` most` ` least` ` only` ` best` ` strongest` ` largest` ` highest` ` preferred` ` greatest` ` fastest` |
| 20 | ` strongest` ` most` ` fastest` ` largest` ` least` ` poorest` ` smallest` ` hardest` ` predominant` ` highest` | ` most` ` least` ` only` ` best` ` largest` ` greatest` ` strongest` ` biggest` ` fastest` ` highest` |
| 21 | ` strongest` ` most` ` predominant` ` fastest` ` largest` ` least` ` worst` ` greatest` ` only` ` smallest` | ` most` ` only` ` least` ` major` ` best` ` main` ` predominant` ` strongest` ` greatest` ` preferred` |
| 22 | ` most` ` predominant` ` only` ` strongest` ` dominant` ` exception` ` toxic` ` least` ` fastest` ` major` | ` most` ` only` ` least` ` major` ` best` ` predominant` ` main` ` preferred` ` sole` ` principal` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... late 1960s, when the latter used to deliver beefburg`)

**skip4** (model's actual next token: `'ers'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` ` Burg` `”.` `”,` `ers` `”).` `burg` ` Burk` ` Bund` `,”` | `s` `hes` `ly` `rys` `erate` `ers` `asy` `sky` `head` ` \|` |
| 1 | `s` ` Burg` ` Bund` `”.` ` Burk` `”,` ` basket` `ers` `burg` `sburg` | `rys` `s` `hes` `erate` `hart` `ling` `ers` `elijk` `ydrate` `acles` |
| 2 | `s` ` Burg` `sburg` `burg` ` clot` ` Burk` ` Bur` ` Tus` ` volley` ` thromb` | `ers` `s` `erate` `ellular` `ая` `suit` `heres` `hers` `lei` `th` |
| 3 | `s` ` Burg` `sburg` `burg` `”.` `ers` ` ”` `”` ` Springfield` ` Ravens` | `ers` `s` `sky` `sm` `sx` ` single` `ай` `地` `selling` `sf` |
| 4 | `s` `ers` `”.` `’` `burg` `sburg` ` Burg` `ings` `ingly` ` clot` | `ers` `s` `,` ` successful` ` Cor` ` [` ` sa` `-` `rule` ` rule` |
| 5 | `s` `sburg` `burg` `ings` `ers` `ons` `undry` `ged` `ingly` `ous` | `ers` `s` `ings` `ish` `up` `'s` ` up` `sm` `man` `er` |
| 6 | `ings` `s` `ers` `sburg` `ingly` `burg` `ry` `ons` `’` `ary` | `ers` `s` `'s` `ish` `-` `地` `,` `man` ` when` `ings` |
| 7 | `s` `ings` `’` `ers` `burg` `sburg` `master` `ons` `ed` `ous` | `ers` `s` `-` `'s` `hers` `,` `an` `ets` ` and` `ish` |
| 8 | `s` `ers` `ings` `burg` `master` `ged` `ous` `ary` `sp` `sburg` | `s` `ers` `'s` `-` `an` `hers` `as` `er` `,` `ings` |
| 9 | `s` `ers` `ings` `as` `master` `ets` ` and` ` v` `ons` `-` | `ers` `s` `an` `-` `ings` `er` `hers` `'s` `as` `sh` |
| 10 | `s` `ers` ` and` `ings` `.` `-` `(` `ses` `ets` `/` | `ers` `er` `s` `de` `an` `ings` `hers` `-` `et` `ward` |
| 11 | `s` `.` ` and` `-` `ers` `ings` `ite` `ets` `(` `ses` | `ers` `s` `ward` `ings` `ht` `ets` `er` `ards` `-` `an` |
| 12 | `s` `ers` `ings` `ite` `ed` `ets` `ged` `er` ` bb` `)` | `ers` `s` `ings` `as` `er` `ht` `ward` `em` `erate` `ues` |
| 13 | `ers` `er` `s` `”.` `ings` `ets` `ged` `.”` `”` `wards` | `ers` `s` `ward` `er` `ings` `as` `st` `h` `-` `ad` |
| 14 | `ers` `ite` `ings` `ons` `haus` `er` `town` `ged` `ards` `ingly` | `ers` `s` `er` `st` `ings` `as` `h` `em` `t` `ne` |
| 15 | `ers` `ings` `haus` `erate` `ite` `aries` `ary` `gers` `ard` `?”` | `ers` `s` `as` `ard` `ings` `em` `er` `h` `are` `-` |
| 16 | `ers` `ery` `ings` `haus` `ues` `hol` `hs` `agh` `gers` `ering` | `ers` `s` `,` `er` `are` `h` `-` `as` `w` `ard` |
| 17 | `ers` `ery` `ings` `agh` `ards` `ues` `ard` `erville` `ache` `hs` | `ers` `s` `as` `-` `ings` `ed` `h` `st` `,` `age` |
| 18 | `ers` `ings` `hs` `agh` `gers` `ues` `ery` `aries` `ards` `erated` | `ers` `s` `-` `,` `h` `ing` `ed` `as` `st` ` and` |
| 19 | `ers` `hs` `ues` `ings` `gers` `ards` `agh` `ling` `ery` `hol` | `ers` `s` `ing` `-` `ed` `,` `h` `in` `er` `t` |
| 20 | `ers` `hs` `ues` `ings` `ards` `aries` `les` `gers` `ies` `ling` | `ers` `s` `-` `,` `ed` `h` `ing` `in` `as` `er` |
| 21 | `ers` `hs` `ues` `agh` `ings` `les` `aries` `ling` `ary` `gers` | `ers` `s` `-` `h` `ed` `,` `as` ` and` `st` `.` |
| 22 | `ers` `hs` `ues` `ings` `ies` `ling` `les` `ary` `agh` `ht` | `ers` `s` `h` `as` `-` `hs` `ing` `ed` `­` ` on` |

### Held-out prompt 3

(128-token window; last 12 tokens: `...ptidases and hydrolyze any acid-amide bonds that`)

**skip4** (model's actual next token: `' are'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` list` ` “` ` other` ` way` `.)` ` full` ` available` ` 
` `.).` `.”).` | `uxe` ` Aviv` `wers` `ividual` `pero` `psis` `/*!` `CRIPTION` `hens` `uclide` |
| 1 | ` other` ` number` ` people` ` store` ` same` ` for` ` pool` ` all` ` event` ` group` | `uxe` `●` `hens` `there` `wers` `anse` `&-` `acter` `bris` `ividual` |
| 2 | ` “` `’` `…` `.”` ` pool` ` they` ` […]` `…”` ` way` ` association` | `there` `urbs` `bris` `acter` ` there` `)\[` `)[` `[\*\*` ` thereon` ` either` |
| 3 | ` association` ` includes` ` included` ` companies` `’` ` they` ` include` ` sales` ` associations` ` will` | `ardless` `acter` `urbs` `anse` `imab` `there` `unnumbered` `ury` `ipsych` `/*!` |
| 4 | ` marketplace` ` specific` ` association` ` range` ` commerce` ` market` ` products` `’` ` display` ` certain` | `PI` `)\[` `rots` `ury` `/*!` `ardless` `iman` ` either` `ansom` `  ` |
| 5 | ` nons` ` typically` ` specific` ` “` ` lead` ` association` ` performance` ` display` ` marketplace` ` individuals` | ` either` `Bits` `PI` ` hitherto` `ames` `unnumbered` `uran` ` heretofore` `either` `rors` |
| 6 | ` lead` ` typically` ` nons` ` specific` ` display` ` history` ` leads` ` reference` ` association` ` control` | ` either` ` hitherto` `PI` `ames` `================================================` ` heretofore` `Ps` ` otherwise` `unnumbered` ` r` |
| 7 | ` “` ` typically` ` ways` ` display` ` portions` ` leads` ` reference` ` way` ` essentially` ` history` | ` either` `PI` `unnumbered` ` they` ` otherwise` ` hitherto` `eng` ` he` `uran` ` or` |
| 8 | ` typically` ` ways` ` include` ` particular` ` specific` ` way` ` contact` ` forms` ` primary` ` generally` | `ets` `uran` ` otherwise` `eng` ` these` ` mas` `$’` ` hitherto` `udes` `mit` |
| 9 | ` are` ` include` ` they` ` (` ` the` ` have` ` form` ` support` ` contained` ` way` | ` otherwise` `unnumbered` `GPU` `ades` ` eu` `$’` `uran` `ets` ` they` ` either` |
| 10 | ` are` ` include` ` support` ` were` ` contained` ` they` ` have` ` themselves` ` form` ` included` | ` otherwise` `unnumbered` ` brought` `innen` ` themselves` `ollen` ` respectively` `oked` `Flags` `essen` |
| 11 | ` themselves` ` are` ` either` ` these` ` form` ` they` ` have` ` occur` ` this` ` include` | `unnumbered` ` otherwise` `lict` `Congress` ` passed` ` hitherto` `{};` `innen` ` occur` ` brought` |
| 12 | ` either` ` themselves` ` otherwise` ` are` ` any` ` such` ` anyone` ` function` ` or` ` anywhere` | ` otherwise` ` existed` ` exist` ` occur` ` occurred` ` exists` `ets` ` had` ` passed` `NAT` |
| 13 | ` otherwise` ` subsequently` ` are` ` form` ` either` ` were` ` themselves` ` contain` ` occurs` ` secondary` | ` existed` ` exist` ` exists` ` otherwise` ` encountered` ` occur` ` passed` `unnumbered` `osp` ` entered` |
| 14 | ` otherwise` ` themselves` ` occurs` ` surrounds` `>`` ` may` ` analogy` ` either` ` are` ` xe` | ` otherwise` ` exist` ` existed` ` encountered` ` appeared` ` exists` ` occur` ` had` ` met` `ch` |
| 15 | ` accompanies` ` occurs` ` accompany` ` occur` ` surrounds` ` exists` ` may` ` physic` ` exist` ` occurred` | ` otherwise` ` exist` `unnumbered` ` existed` ` arise` ` exists` ` occur` ` normally` ` occurred` ` brought` |
| 16 | ` occurs` ` occur` ` they` ` exists` ` are` ` surrounds` ` accompanies` ` occurred` ` arise` ` exist` | ` are` ` may` ` might` ` were` ` have` ` they` ` otherwise` ` the` ` occur` ` could` |
| 17 | ` occurs` ` are` ` occur` ` exists` ` occurred` ` arise` ` arises` ` may` ` they` ` were` | ` they` ` are` ` may` ` occur` ` might` ` their` ` appear` ` occurs` ` have` ` arise` |
| 18 | ` occur` ` are` ` occurs` ` arise` ` exist` ` were` ` occurred` ` contain` ` appear` ` exists` | ` are` ` occur` ` they` ` may` ` appear` ` the` ` have` ` otherwise` ` might` ` exist` |
| 19 | ` occur` ` are` ` arise` ` exist` ` appear` ` contain` ` originate` ` accompany` ` constitute` ` have` | ` are` ` occur` ` they` ` may` ` the` ` exist` ` appear` ` have` ` arise` ` might` |
| 20 | ` are` ` occur` ` arise` ` exist` ` appear` ` have` ` originate` ` contain` ` were` ` constitute` | ` are` ` may` ` occur` ` the` ` have` ` exist` ` they` ` can` ` appear` ` might` |
| 21 | ` are` ` occur` ` arise` ` exist` ` have` ` appear` ` were` ` originate` ` contain` ` remain` | ` are` ` exist` ` may` ` occur` ` the` ` they` ` can` ` might` ` form` ` have` |
| 22 | ` are` ` occur` ` have` ` may` ` arise` ` exist` ` appear` ` were` ` might` ` contain` | ` are` ` may` ` occur` ` can` ` might` ` they` ` exist` ` have` ` form` ` arise` |
