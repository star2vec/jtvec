# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-1.4b (revision fedc38a16eea3bd36a96b906d78d11d2ce18ed79)
- device: mps, dtype: float32, seed: 1
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-20T15:15:38, 5385.1s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [5, 7, 8, 11, 12, 13, 14, 15]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [4, 12, 13, 14]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.3534 vs random +0.0003

## Task baseline gate

Included = in-context top-1 accuracy >= 80%.

| task | protocol | accuracy | items | verdict |
|---|---|---|---|---|
| capital-operand | completion | 91.7% | 36 | **INCLUDED** |
| capital-recall | completion | 91.7% | 36 | **INCLUDED** |
| context-binding | completion | 53.3% | 30 | dropped |
| multihop-scaled | completion | 62.5% | 24 | dropped |
| opposites | completion | 100.0% | 16 | **INCLUDED** |
| swap-capitals | swap | 100.0% | 16 | **INCLUDED** |
| typo-robustness | typo | 76.7% | 30 | dropped |
| word-pairs | completion | 95.8% | 24 | **INCLUDED** |

## Probing eval (rank of the intermediate token in the lens readout)

HMR = harmonic mean rank over items (lower is better); pass@10 = fraction of items with rank <= 10. `random` = mean over Frobenius-matched Gaussian matrices (10 seeds).

### skip4 / capital-operand

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 818.8 | 540.8 | 1716.0 | 0.00 | 0.00 | 0.01 |
| 1 | 207.1 | 215.8 | 1572.2 | 0.03 | 0.00 | 0.00 |
| 2 | 420.7 | 152.7 | 2876.4 | 0.00 | 0.00 | 0.00 |
| 3 | 443.8 | 177.3 | 2972.6 | 0.00 | 0.00 | 0.00 |
| 4 | 180.0 | 424.8 | 1324.1 | 0.03 | 0.00 | 0.00 |
| 5 | 291.6 | 940.2 | 2387.3 | 0.00 | 0.00 | 0.00 |
| 6 | 239.9 | 755.6 | 1970.6 | 0.00 | 0.00 | 0.00 |
| 7 | 373.2 | 2870.0 | 2131.1 | 0.00 | 0.00 | 0.00 |
| 8 | 360.5 | 1233.0 | 2452.6 | 0.00 | 0.00 | 0.00 |
| 9 | 512.1 | 663.7 | 1938.0 | 0.00 | 0.00 | 0.01 |
| 10 | 682.5 | 578.8 | 3043.6 | 0.00 | 0.00 | 0.00 |
| 11 | 16.5 | 66.0 | 3067.3 | 0.17 | 0.03 | 0.00 |
| 12 | 15.8 | 126.7 | 1263.9 | 0.19 | 0.03 | 0.01 |
| 13 | 1.6 | 19.7 | 2232.8 | 0.94 | 0.17 | 0.00 |
| 14 | 2.6 | 67.5 | 1993.5 | 0.83 | 0.00 | 0.00 |
| 15 | 1.6 | 1.3 | 2361.4 | 1.00 | 1.00 | 0.00 |
| 16 | 1.6 | 1.6 | 2000.2 | 1.00 | 1.00 | 0.00 |
| 17 | 1.7 | 1.7 | 1806.7 | 1.00 | 1.00 | 0.01 |
| 18 | 1.8 | 2.7 | 1990.7 | 1.00 | 0.92 | 0.01 |
| 19 | 2.4 | 4.6 | 2553.7 | 1.00 | 0.78 | 0.00 |
| 20 | 3.5 | 23.6 | 4310.6 | 0.97 | 0.14 | 0.01 |
| 21 | 5.2 | 37.1 | 2258.5 | 0.75 | 0.06 | 0.00 |
| 22 | 5.2 | 41.4 | 2144.7 | 0.75 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.3 / pass@10 1.00; logit HMR 1.3 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 381.7 | 236.3 | 1636.0 | 0.00 | 0.03 | 0.00 |
| 1 | 325.6 | 162.1 | 925.1 | 0.00 | 0.03 | 0.01 |
| 2 | 469.3 | 130.9 | 1640.0 | 0.00 | 0.00 | 0.00 |
| 3 | 654.7 | 87.5 | 3315.0 | 0.00 | 0.03 | 0.00 |
| 4 | 505.2 | 237.4 | 2713.5 | 0.00 | 0.00 | 0.00 |
| 5 | 380.7 | 992.2 | 2050.1 | 0.00 | 0.00 | 0.00 |
| 6 | 376.8 | 368.5 | 1377.8 | 0.00 | 0.00 | 0.00 |
| 7 | 372.4 | 1746.2 | 1479.5 | 0.00 | 0.00 | 0.01 |
| 8 | 370.1 | 984.4 | 1832.0 | 0.00 | 0.00 | 0.00 |
| 9 | 671.6 | 347.1 | 3129.8 | 0.00 | 0.00 | 0.00 |
| 10 | 417.1 | 157.8 | 2288.8 | 0.00 | 0.00 | 0.00 |
| 11 | 24.4 | 34.1 | 2444.2 | 0.08 | 0.08 | 0.00 |
| 12 | 35.7 | 110.7 | 739.0 | 0.08 | 0.00 | 0.01 |
| 13 | 3.0 | 39.4 | 2347.2 | 0.69 | 0.08 | 0.00 |
| 14 | 2.0 | 44.3 | 2215.7 | 0.78 | 0.06 | 0.00 |
| 15 | 1.5 | 1.5 | 1818.7 | 0.92 | 0.97 | 0.01 |
| 16 | 1.4 | 1.4 | 1804.2 | 0.89 | 0.97 | 0.00 |
| 17 | 1.3 | 1.2 | 1704.3 | 0.86 | 0.97 | 0.01 |
| 18 | 1.3 | 1.1 | 2687.7 | 0.92 | 1.00 | 0.00 |
| 19 | 1.3 | 1.1 | 2085.0 | 0.89 | 1.00 | 0.00 |
| 20 | 1.1 | 1.3 | 2266.2 | 0.94 | 0.97 | 0.00 |
| 21 | 1.1 | 1.3 | 2945.9 | 0.97 | 0.97 | 0.00 |
| 22 | 1.1 | 1.5 | 1793.3 | 1.00 | 0.94 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 204.3 | 1999.8 | 3661.0 | 0.00 | 0.00 | 0.00 |
| 1 | 360.6 | 909.1 | 4382.1 | 0.00 | 0.00 | 0.00 |
| 2 | 31.3 | 574.9 | 4405.9 | 0.06 | 0.00 | 0.00 |
| 3 | 76.2 | 725.9 | 7653.7 | 0.06 | 0.00 | 0.00 |
| 4 | 239.0 | 654.1 | 5519.1 | 0.00 | 0.00 | 0.00 |
| 5 | 336.1 | 2177.0 | 4024.8 | 0.00 | 0.00 | 0.00 |
| 6 | 296.1 | 332.1 | 4874.3 | 0.00 | 0.00 | 0.00 |
| 7 | 245.9 | 551.5 | 4070.1 | 0.00 | 0.00 | 0.00 |
| 8 | 129.8 | 114.5 | 2367.4 | 0.00 | 0.06 | 0.00 |
| 9 | 115.5 | 85.1 | 4498.2 | 0.06 | 0.06 | 0.00 |
| 10 | 65.0 | 50.3 | 4101.0 | 0.06 | 0.06 | 0.00 |
| 11 | 6.1 | 2.9 | 3743.0 | 0.25 | 0.44 | 0.00 |
| 12 | 3.9 | 3.0 | 3447.1 | 0.44 | 0.38 | 0.00 |
| 13 | 1.1 | 1.0 | 4971.4 | 1.00 | 1.00 | 0.00 |
| 14 | 1.0 | 1.0 | 3523.4 | 1.00 | 1.00 | 0.00 |
| 15 | 1.0 | 1.0 | 4494.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.0 | 1.0 | 3822.6 | 1.00 | 1.00 | 0.00 |
| 17 | 1.0 | 1.0 | 4767.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.0 | 1.0 | 4665.5 | 1.00 | 1.00 | 0.00 |
| 19 | 1.2 | 1.0 | 4868.3 | 1.00 | 1.00 | 0.00 |
| 20 | 1.2 | 1.0 | 4720.0 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 2602.5 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 3637.8 | 1.00 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 177.3 | 139.0 | 1923.1 | 0.04 | 0.04 | 0.02 |
| 1 | 20.1 | 23.5 | 2188.1 | 0.08 | 0.04 | 0.00 |
| 2 | 85.3 | 21.1 | 3351.4 | 0.08 | 0.08 | 0.00 |
| 3 | 20.7 | 22.9 | 3327.9 | 0.08 | 0.04 | 0.00 |
| 4 | 17.1 | 22.4 | 3075.7 | 0.08 | 0.04 | 0.00 |
| 5 | 69.8 | 22.8 | 2714.2 | 0.08 | 0.04 | 0.00 |
| 6 | 31.5 | 22.5 | 2034.3 | 0.04 | 0.04 | 0.00 |
| 7 | 39.1 | 22.4 | 2644.8 | 0.04 | 0.04 | 0.00 |
| 8 | 20.0 | 15.1 | 1849.7 | 0.04 | 0.08 | 0.00 |
| 9 | 62.6 | 17.5 | 3901.3 | 0.04 | 0.08 | 0.00 |
| 10 | 93.2 | 14.4 | 2452.9 | 0.04 | 0.08 | 0.00 |
| 11 | 7.4 | 4.5 | 2339.0 | 0.29 | 0.33 | 0.00 |
| 12 | 5.4 | 6.3 | 1987.5 | 0.29 | 0.29 | 0.00 |
| 13 | 2.1 | 3.0 | 2385.8 | 0.62 | 0.50 | 0.00 |
| 14 | 1.8 | 1.8 | 4208.1 | 0.79 | 0.83 | 0.00 |
| 15 | 1.6 | 1.6 | 2123.7 | 0.75 | 0.88 | 0.00 |
| 16 | 1.7 | 1.5 | 1935.3 | 0.79 | 0.88 | 0.00 |
| 17 | 1.4 | 1.5 | 2237.0 | 0.79 | 0.88 | 0.00 |
| 18 | 1.4 | 1.6 | 2215.5 | 0.83 | 0.88 | 0.00 |
| 19 | 2.3 | 1.7 | 2369.4 | 0.92 | 0.75 | 0.00 |
| 20 | 2.1 | 1.7 | 4596.1 | 0.83 | 0.71 | 0.00 |
| 21 | 1.4 | 1.7 | 2914.9 | 0.92 | 0.79 | 0.00 |
| 22 | 1.2 | 1.5 | 2567.1 | 0.96 | 0.92 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 0.96; logit HMR 1.3 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.3534 | +0.0003 | -0.4295 | 37.5% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `...; the other compounds were decomposed extensively. Butene was the`)

**skip4** (model's actual next token: `' major'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` ` \"` ` get` ` $` ` name` ` <` ` ways` ` new` ` work` `

` | `iet` ` programmers` ` independ` `medsc` `rette` `ingo` ` IB` `thur` `ophyll` `===========================` |
| 1 | `s` ` which` ` new` ` where` ` set` ` the` ` that` ` site` ` done` ` name` | `rette` `same` `ophyll` `medsc` ` GENERATED` `iet` ` programmers` `achy` ` follow` `rett` |
| 2 | ` ‘` ` which` `

` ` value` ` followed` ` defined` ` where` ` new` ` set` `
` | `same` `iet` `ophyll` `inki` ` same` `medsc` ` nature` `achy` `tha` ` GENERATED` |
| 3 | `

` ` the` ` which` `
` ` following` ` where` ` followed` ` only` ` a` ` above` | `same` `ophyll` `inki` `iet` `tha` `buk` `acha` ` same` `14514` `medsc` |
| 4 | ` the` `s` ` which` ` another` ` “` ` that` ` followed` `

` ` a` ` itself` | `ophyll` `same` `ietz` `inki` `Hell` `acha` `tha` ` establ` `ps` `τά` |
| 5 | ` which` ` the` `s` ` only` ` defined` ` not` ` self` ` '` ` that` ` different` | `same` ` prevailing` ` same` `tha` ` likewise` `ietz` `ophyll` `yan` ` strongest` ` establ` |
| 6 | ` only` `s` ` self` ` ‘` ` which` ` $` ` data` ` a` ` the` ` an` | `ietz` `ophyll` `same` ` same` ` liv` ` only` ` prevailing` `ople` `RI` `14514` |
| 7 | `s` ` only` ` which` ` new` ` followed` ` data` ` different` ` ‘` ` following` ` the` | ` predominant` ` following` ` only` `der` `ophyll` ` dominating` ` prevailing` ` follow` ` predomin` ` preferred` |
| 8 | ` followed` ` only` ` '` ` ‘` ` the` ` most` ` a` ` which` ` nature` ` new` | ` only` ` predominant` ` preferred` ` followed` ` strongest` ` dominating` ` most` `ophyll` ` following` ` dominant` |
| 9 | ` followed` ` only` ` $` ` most` ` the` ` preferred` ` which` ` index` ` $(` ` choice` | ` predominant` ` most` ` preferred` ` only` ` dominant` ` strongest` ` prevailing` ` result` ` same` ` dominating` |
| 10 | ` only` ` followed` ` $` ` which` ` most` `  ` ` '` ` ‘` ` preferred` `:` | ` predominant` ` preferred` ` most` ` dominant` ` only` ` prevailing` ` strongest` ` highest` ` best` ` dominating` |
| 11 | ` only` ` followed` ` most` ` '` ` preferred` ` ‘` ` predominant` ` primarily` ` closest` ` “` | ` most` ` preferred` ` predominant` ` only` ` best` ` dominant` ` strongest` ` highest` ` fastest` ` prevailing` |
| 12 | ` only` ` most` ` preferred` ` predominant` ` highest` ` primarily` ` closest` ` dominant` ` predominantly` ` source` | ` only` ` most` ` preferred` ` predominant` ` highest` ` strongest` ` main` ` best` ` dominant` ` lowest` |
| 13 | ` only` ` most` ` highest` ` largest` ` predominant` ` closest` ` biggest` ` strongest` ` lowest` ` smallest` | ` most` ` only` ` preferred` ` highest` ` predominant` ` strongest` ` first` ` main` ` best` ` largest` |
| 14 | ` only` ` predominant` ` largest` ` most` ` key` ` source` ` result` ` major` ` preferred` ` main` | ` only` ` most` ` main` ` predominant` ` preferred` ` major` ` highest` ` first` ` primary` ` largest` |
| 15 | ` most` ` largest` ` only` ` strongest` ` highest` ` closest` ` best` ` fastest` ` predominant` ` biggest` | ` most` ` only` ` main` ` predominant` ` largest` ` highest` ` major` ` greatest` ` biggest` ` principal` |
| 16 | ` most` ` strongest` ` predominant` ` largest` ` highest` ` fastest` ` cheapest` ` lowest` ` biggest` ` closest` | ` most` ` main` ` only` ` predominant` ` preferred` ` principal` ` primary` ` major` ` highest` ` dominant` |
| 17 | ` predominant` ` most` ` dominant` ` shortest` ` strongest` ` preferred` ` closest` ` largest` ` product` ` smallest` | ` most` ` main` ` only` ` predominant` ` major` ` primary` ` preferred` ` first` ` product` ` principal` |
| 18 | ` predominant` ` dominant` ` most` ` major` ` largest` ` strongest` ` biggest` ` smallest` ` closest` ` main` | ` most` ` main` ` major` ` predominant` ` primary` ` only` ` principal` ` dominant` ` product` ` first` |
| 19 | ` *` ` predominant` ` major` ` most` ` dominant` ` largest` ` biggest` ` hottest` ` main` ` greatest` | ` major` ` most` ` main` ` primary` ` predominant` ` only` ` principal` ` product` ` dominant` ` first` |
| 20 | ` predominant` ` dominant` ` most` ` primary` ` major` ` main` `
` `
` ` principal` ` degradation` | ` most` ` major` ` main` ` primary` ` only` ` principal` ` predominant` ` first` ` product` ` dominant` |
| 21 | ` predominant` ` dominant` ` primary` ` volatile` ` major` ` chemical` ` toxin` ` most` ` molecule` ` metabolite` | ` most` ` major` ` main` ` only` ` primary` ` predominant` ` first` ` principal` ` dominant` ` by` |
| 22 | ` predominant` ` dominant` ` major` ` primary` ` most` ` toxic` ` principal` ` chemical` ` volatile` ` vol` | ` most` ` major` ` main` ` only` ` predominant` ` first` ` primary` ` dominant` ` principal` ` key` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... late 1960s, when the latter used to deliver beefburg`)

**skip4** (model's actual next token: `'ers'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` `sburg` `burg` `sar` `ers` `BUR` `shire` ` Burg` `gy` `ging` | `ers` `und` `s` `am` `osity` `,` `undy` `ulence` `arel` `dyn` |
| 1 | `s` `,` `ers` `sm` `sburg` ` (` ` in` `ed` `san` ` and` | `ers` `s` `ite` `ado` `undy` `acious` ` dec` `ous` ` wre` `  ` |
| 2 | `s` `ed` `sburg` `ing` `ers` `,` ` LB` ` bc` `\` ` inter` | `ers` `s` `eh` `ado` ` dec` `ha` `setlength` `ite` `undy` `acious` |
| 3 | `s` `sburg` `ers` `ed` ` --` `burg` `ized` `ulations` ` Burg` `san` | `ers` `s` `  ` `,` ` (` `gers` ` dec` ` ,` `am` ` and` |
| 4 | `s` `ers` `sburg` `ed` ` bul` `san` ` Burg` ` LB` `os` ` ov` | `s` `ers` `,` `-` ` and` `  ` `2` `ite` ` (` `am` |
| 5 | `s` `ers` ` Burg` `burg` `ed` `sburg` `some` ` (` ` az` `sl` | `s` `ers` `  ` `-` `,` ` (` `m` ` and` `ous` `2` |
| 6 | `s` `ers` `ed` `burg` `some` ` Burg` `burgh` `er` `sb` `ging` | `s` `ers` `ous` `ing` `-` `ed` `am` `gery` ` wre` `  ` |
| 7 | `s` `ers` `some` `ed` `er` `ging` ` (` `sp` `ian` `-` | `s` `ers` `-` `am` `m` `os` `ous` `ed` `st` `ie` |
| 8 | `s` `ers` `ging` `some` `sa` ` "` `er` ` Burg` `nu` `ular` | `ers` `s` `m` `-` `ress` `ter` `ed` `id` `gery` `und` |
| 9 | `s` `ers` `ru` `he` `sa` `some` `ah` `ue` `t` `iv` | `ers` `s` `m` `-` `gery` `id` `ger` `anc` `ous` `ue` |
| 10 | `s` `ers` `t` `he` `b` `ue` `ru` `l` `st` ` and` | `ers` `s` `m` `-` `er` `am` `ger` `id` `il` `h` |
| 11 | `s` `ers` ` then` ` ‘` `  ` ` and` ` etc` ` but` ` '` `l` | `ers` `s` `-` `m` `om` `h` `at` `ger` ` and` `il` |
| 12 | ` ‘` `ers` ` GB` ` BG` ` Burg` ` B` `s` ` AB` ` BV` ` BA` | `ers` `s` `-` `m` `om` `at` `er` `h` `le` `ger` |
| 13 | ` ‘` ` which` `ers` `erville` `’` ` ’` `s` ` post` ` “` ` micro` | `ers` `s` `-` `m` `om` ` and` `p` `at` `h` `,` |
| 14 | `ers` ` ‘` ` “` ` ’` `’` `ERS` ` [*` ` present` ` which` `’.` | `ers` `s` `m` `-` `h` `le` `,` ` and` `at` `t` |
| 15 | `ers` `ERS` `!(` `)",` ` represents` ` ‘` `)` `ination` `s` `gery` | `ers` `s` `-` `le` `m` `er` ` and` `erm` `om` `,` |
| 16 | `ers` `worm` ` meat` `undy` ` potato` `ler` `ants` `ERS` ` cattle` ` Bacon` | `ers` `s` `-` `er` ` and` `le` `,` `al` `om` `h` |
| 17 | `ers` `lers` `ERS` `ues` `hs` `ler` `ling` ` vegetarian` ` meat` ` lettuce` | `ers` `s` `-` ` and` `,` `er` `al` `h` `in` `le` |
| 18 | `ers` `rations` `ler` `lers` `led` `ERS` `ling` `chers` `rers` `erville` | `ers` `s` `-` `,` `h` ` and` `in` `er` ` (` ` in` |
| 19 | ` *` ` (*` ` \[*` `.[@` ` *(` ` ([@` `ers` ` [*` `[@` `*.*` | `ers` `s` `-` `,` `h` ` and` ` (` ` in` `in` `ed` |
| 20 | `
` ` 
` ` he` ` {#` `he` `ers` `
` ` she` `she` ` 
` | `ers` `s` `-` `,` `h` ` and` ` (` ` in` `in` `w` |
| 21 | `ers` ` wur` `undy` `ues` ` Parish` ` diesel` `hm` `hers` `rers` `lers` | `ers` `s` `-` `h` `,` ` and` ` in` `in` ` (` `ues` |
| 22 | `ers` `undy` `ues` `hs` `ERS` `ars` ` burgers` `hal` `hers` `uit` | `ers` `s` `h` `-` `he` `ar` `er` `,` `hs` `­` |

### Held-out prompt 3

(128-token window; last 12 tokens: `...ptidases and hydrolyze any acid-amide bonds that`)

**skip4** (model's actual next token: `' are'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` “` ` work` ` <` ` maybe` ` _` ` instead` ` people` ` really` ` works` `s` | `special` `vor` ` they` ` specifically` ` LIMITED` `ben` ` biography` `dd` ` BGCOLOR` `rette` |
| 1 | `’` `s` ` which` ` instead` ` work` `,` ` —` ` means` ` set` ` (` | `rette` `dd` `ld` ` specifically` ` mainly` ` had` `specifically` `iet` `jh` ` they` |
| 2 | `’` ` —` `:` `s` ` reductions` ` defined` ` where` ` add` ` value` ` improvements` | ` have` `jh` ` are` ` will` ` had` `ld` `dj` `Ul` `rette` ` also` |
| 3 | ` follows` ` which` ` follow` ` includes` ` the` ` those` ` changes` ` include` ` following` ` specifically` | `dd` ` possibly` `dj` ` Immigration` ` forthcoming` ` adiab` `                        ` ` traders` ` Maj` `cd` |
| 4 | ` follows` ` the` ` includes` ` follow` `’` ` those` ` include` ` specific` ` previous` ` itself` | ` possibly` `aa` `dd` `aber` ` ICE` ` formed` ` Immigration` ` defined` ` might` `sp` |
| 5 | `…` ` specifically` ` follows` ` subsequent` ` the` ` positive` ` are` ` specific` ` previous` `’` | ` have` ` possibly` ` might` ` resulted` ` subsequently` ` form` ` result` ` follows` `result` `dd` |
| 6 | `…` ` “` `…”` ` separate` `’` ` ratings` ` follows` ` are` ` follow` ` specifically` | `aa` ` seniors` `bish` ` resulted` ` deposited` ` entered` ` developers` ` therein` ` subsequently` ` deposits` |
| 7 | `’` ` —` ` are` `…` ` loss` ` the` `

` `:` ` those` ` follows` | `aa` ` had` ` form` ` have` ` dic` ` might` ` met` ` follows` `cd` ` formed` |
| 8 | ` —` ` _` ` are` ` similar` ` math` ` the` ` chemistry` `:` `

` ` developers` | `aa` ` might` ` follows` ` advert` ` preceded` ` advertisers` ` possible` ` entered` ` met` ` contained` |
| 9 | ` —` ` are` ` _` ` function` ` either` ` whether` ` products` ` product` ` code` ` themselves` | `aa` ` contained` ` either` ` occur` ` met` ` result` ` de` ` normal` ` might` ` are` |
| 10 | ` are` ` —` ` _` ` **` ` may` ` were` `’` ` have` ` –` ` is` | ` may` ` in` ` either` ` any` ` the` ` S` ` contained` ` R` `aa` ` might` |
| 11 | ` —` ` are` ` otherwise` ` such` ` _` ` may` ` either` ` –` ` have` ` (` | ` in` ` might` ` may` ` any` ` possibly` ` ordinary` ` have` ` are` ` produced` ` would` |
| 12 | ` may` ` are` ` normal` ` is` ` any` ` might` ` either` ` such` ` in` ` the` | ` may` ` might` ` are` ` occur` ` any` ` possibly` ` have` ` occurred` ` the` ` occurs` |
| 13 | ` are` ` may` ` might` ` they` ` have` ` is` ` were` ` can` ` either` ` normal` | ` are` ` may` ` the` ` might` ` it` ` any` ` have` ` occur` ` either` ` in` |
| 14 | ` they` ` are` ` may` ` their` ` themselves` ` might` ` occurs` ` have` ` is` ` arises` | ` are` ` may` ` the` ` they` ` might` ` have` ` occur` ` any` ` is` ` their` |
| 15 | ` are` ` occurs` ` they` ` may` ` is` ` exists` ` occur` ` might` ` arises` ` has` | ` are` ` may` ` they` ` the` ` might` ` occur` ` a` ` is` ` have` ` exist` |
| 16 | ` are` ` they` ` occur` ` occurs` ` exists` ` may` ` arise` ` comprise` ` exist` ` might` | ` are` ` may` ` they` ` occur` ` might` ` were` ` exist` ` the` ` have` ` a` |
| 17 | ` are` ` occur` ` occurs` ` exist` ` were` ` comprise` ` they` ` constitute` ` exists` ` bonds` | ` are` ` may` ` they` ` occur` ` might` ` have` ` the` ` were` ` is` ` can` |
| 18 | ` are` ` occur` ` were` ` may` ` comprise` ` might` ` occurs` ` have` ` constitute` ` belong` | ` are` ` may` ` the` ` they` ` occur` ` can` ` a` ` have` ` might` ` is` |
| 19 | ` are` ` occur` ` were` ` *` ` they` ` occurs` ` contain` ` arise` ` bonds` ` may` | ` are` ` the` ` may` ` a` ` they` ` in` ` have` `,` ` un` ` occur` |
| 20 | ` are` ` occur` ` they` ` arise` `
` ` have` ` may` ` were` ` appear` ` lie` | ` are` ` the` ` may` ` in` ` a` ` is` ` have` `,` ` occur` ` can` |
| 21 | ` are` ` occur` ` have` ` arise` ` appear` ` comprise` ` contain` ` were` ` remain` ` exist` | ` are` ` the` ` may` ` a` ` in` ` they` `,` ` can` ` is` ` have` |
| 22 | ` are` ` occur` ` arise` ` appear` ` contain` ` have` ` may` ` exist` ` constitute` ` comprise` | ` are` ` may` ` occur` ` they` ` have` ` can` ` the` ` exist` ` do` ` a` |
