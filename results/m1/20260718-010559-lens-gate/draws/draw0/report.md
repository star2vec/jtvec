# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-410m (revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f)
- device: mps, dtype: float32, seed: 0
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-18T00:22:30, 919.3s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [8, 10, 12]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.6046 vs random +0.0086

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
| 0 | 1409.3 | 3234.6 | 2282.0 | 0.00 | 0.00 | 0.00 |
| 1 | 1988.9 | 2129.3 | 3631.1 | 0.00 | 0.00 | 0.00 |
| 2 | 935.5 | 3305.3 | 3222.7 | 0.00 | 0.00 | 0.01 |
| 3 | 1455.3 | 3182.7 | 3087.2 | 0.00 | 0.00 | 0.00 |
| 4 | 793.1 | 4239.2 | 2965.6 | 0.00 | 0.00 | 0.00 |
| 5 | 375.2 | 3500.7 | 4648.0 | 0.00 | 0.00 | 0.00 |
| 6 | 363.5 | 3626.2 | 4163.4 | 0.00 | 0.00 | 0.00 |
| 7 | 90.9 | 2994.3 | 2686.8 | 0.03 | 0.00 | 0.01 |
| 8 | 337.0 | 1193.3 | 3787.6 | 0.00 | 0.00 | 0.00 |
| 9 | 429.9 | 3013.9 | 2699.9 | 0.00 | 0.00 | 0.00 |
| 10 | 19.2 | 1859.4 | 2765.8 | 0.19 | 0.00 | 0.00 |
| 11 | 324.2 | 2604.2 | 3701.5 | 0.00 | 0.00 | 0.00 |
| 12 | 191.4 | 2670.6 | 3114.4 | 0.00 | 0.00 | 0.00 |
| 13 | 2.8 | 30.5 | 2544.7 | 0.56 | 0.03 | 0.01 |
| 14 | 2.6 | 158.1 | 3952.5 | 0.81 | 0.00 | 0.00 |
| 15 | 9.7 | 91.2 | 1716.6 | 0.22 | 0.03 | 0.00 |
| 16 | 9.3 | 58.8 | 1830.4 | 0.28 | 0.06 | 0.00 |
| 17 | 1.3 | 1.1 | 2573.2 | 1.00 | 1.00 | 0.00 |
| 18 | 1.5 | 1.5 | 3950.4 | 1.00 | 1.00 | 0.00 |
| 19 | 2.2 | 2.5 | 1964.3 | 1.00 | 0.94 | 0.00 |
| 20 | 3.2 | 5.0 | 2024.2 | 0.94 | 0.67 | 0.01 |
| 21 | 4.8 | 30.9 | 1897.1 | 0.83 | 0.11 | 0.00 |
| 22 | 7.4 | 51.6 | 1774.4 | 0.58 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.2 / pass@10 1.00; logit HMR 1.1 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 1690.8 | 3864.1 | 2759.8 | 0.00 | 0.00 | 0.00 |
| 1 | 2729.6 | 1696.3 | 3217.8 | 0.00 | 0.00 | 0.00 |
| 2 | 1976.5 | 3473.4 | 3239.4 | 0.00 | 0.00 | 0.00 |
| 3 | 814.9 | 3572.8 | 2149.7 | 0.00 | 0.00 | 0.00 |
| 4 | 1602.4 | 3320.2 | 2449.2 | 0.00 | 0.00 | 0.00 |
| 5 | 2066.7 | 2106.8 | 3020.3 | 0.00 | 0.00 | 0.00 |
| 6 | 1238.2 | 4384.9 | 4432.9 | 0.00 | 0.00 | 0.00 |
| 7 | 893.6 | 2021.2 | 4182.6 | 0.00 | 0.00 | 0.00 |
| 8 | 607.8 | 2897.9 | 3029.3 | 0.00 | 0.00 | 0.01 |
| 9 | 502.2 | 2274.9 | 2670.8 | 0.00 | 0.00 | 0.01 |
| 10 | 303.1 | 203.5 | 2392.5 | 0.00 | 0.03 | 0.00 |
| 11 | 361.2 | 1377.2 | 3105.4 | 0.00 | 0.00 | 0.01 |
| 12 | 46.2 | 562.0 | 3413.3 | 0.08 | 0.00 | 0.00 |
| 13 | 6.0 | 318.5 | 1982.8 | 0.36 | 0.00 | 0.00 |
| 14 | 6.6 | 537.4 | 2357.8 | 0.36 | 0.00 | 0.01 |
| 15 | 5.3 | 231.2 | 2327.7 | 0.36 | 0.00 | 0.01 |
| 16 | 2.5 | 61.5 | 1740.5 | 0.61 | 0.08 | 0.00 |
| 17 | 1.7 | 1.9 | 2131.6 | 0.89 | 0.94 | 0.00 |
| 18 | 1.4 | 1.4 | 2889.0 | 0.89 | 0.92 | 0.01 |
| 19 | 1.2 | 1.2 | 2122.4 | 0.86 | 0.97 | 0.00 |
| 20 | 1.2 | 1.1 | 1753.2 | 0.89 | 0.97 | 0.00 |
| 21 | 1.2 | 1.1 | 1999.4 | 0.92 | 1.00 | 0.00 |
| 22 | 1.2 | 1.1 | 2134.0 | 0.92 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 0.92; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 1308.1 | 9312.7 | 3486.5 | 0.00 | 0.00 | 0.00 |
| 1 | 532.6 | 6446.2 | 5285.1 | 0.00 | 0.00 | 0.00 |
| 2 | 630.6 | 8306.8 | 5368.1 | 0.00 | 0.00 | 0.00 |
| 3 | 429.4 | 5045.2 | 5986.2 | 0.00 | 0.00 | 0.00 |
| 4 | 465.2 | 7144.9 | 4160.5 | 0.00 | 0.00 | 0.00 |
| 5 | 212.1 | 4939.0 | 3520.7 | 0.00 | 0.00 | 0.00 |
| 6 | 365.4 | 5752.8 | 5641.4 | 0.00 | 0.00 | 0.00 |
| 7 | 422.2 | 4110.1 | 6305.6 | 0.00 | 0.00 | 0.00 |
| 8 | 153.0 | 2062.7 | 6211.6 | 0.00 | 0.00 | 0.00 |
| 9 | 163.6 | 6150.7 | 4471.3 | 0.00 | 0.00 | 0.01 |
| 10 | 7.7 | 216.0 | 5865.6 | 0.19 | 0.00 | 0.00 |
| 11 | 40.1 | 212.8 | 3452.3 | 0.06 | 0.00 | 0.00 |
| 12 | 33.0 | 23.9 | 3492.3 | 0.06 | 0.19 | 0.00 |
| 13 | 1.6 | 1.3 | 5915.1 | 0.81 | 0.94 | 0.00 |
| 14 | 1.5 | 1.4 | 4140.7 | 0.94 | 1.00 | 0.00 |
| 15 | 1.3 | 1.3 | 4687.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.4 | 1.0 | 4950.8 | 1.00 | 1.00 | 0.00 |
| 17 | 1.1 | 1.0 | 5793.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.0 | 1.0 | 4840.9 | 1.00 | 1.00 | 0.00 |
| 19 | 1.0 | 1.0 | 4740.7 | 1.00 | 1.00 | 0.00 |
| 20 | 1.0 | 1.0 | 3693.3 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 4328.3 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 4541.5 | 1.00 | 1.00 | 0.01 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 23.9 | 933.5 | 3186.5 | 0.04 | 0.00 | 0.00 |
| 1 | 579.7 | 1090.8 | 3880.0 | 0.00 | 0.00 | 0.00 |
| 2 | 208.2 | 1566.1 | 2771.8 | 0.04 | 0.00 | 0.01 |
| 3 | 987.5 | 168.7 | 3388.1 | 0.00 | 0.04 | 0.00 |
| 4 | 414.3 | 159.4 | 2487.8 | 0.00 | 0.00 | 0.00 |
| 5 | 151.8 | 23.1 | 3501.5 | 0.00 | 0.04 | 0.00 |
| 6 | 65.4 | 22.6 | 3684.3 | 0.04 | 0.04 | 0.00 |
| 7 | 31.8 | 22.7 | 3858.8 | 0.04 | 0.04 | 0.00 |
| 8 | 19.6 | 20.5 | 2830.0 | 0.04 | 0.08 | 0.00 |
| 9 | 37.8 | 27.3 | 4453.4 | 0.04 | 0.08 | 0.00 |
| 10 | 8.6 | 13.7 | 3113.0 | 0.21 | 0.12 | 0.00 |
| 11 | 12.0 | 9.4 | 2971.3 | 0.17 | 0.17 | 0.00 |
| 12 | 6.9 | 12.9 | 2125.5 | 0.17 | 0.17 | 0.00 |
| 13 | 4.8 | 4.4 | 2645.9 | 0.33 | 0.33 | 0.00 |
| 14 | 3.7 | 3.5 | 3625.3 | 0.42 | 0.42 | 0.00 |
| 15 | 2.9 | 2.4 | 2938.7 | 0.54 | 0.58 | 0.00 |
| 16 | 2.7 | 1.8 | 2650.1 | 0.62 | 0.67 | 0.00 |
| 17 | 2.3 | 1.9 | 1853.7 | 0.62 | 0.79 | 0.00 |
| 18 | 2.3 | 1.7 | 3189.3 | 0.62 | 0.75 | 0.00 |
| 19 | 2.0 | 1.4 | 2861.3 | 0.75 | 0.83 | 0.00 |
| 20 | 1.8 | 1.4 | 1975.8 | 0.83 | 0.83 | 0.00 |
| 21 | 1.5 | 1.4 | 1352.8 | 0.88 | 0.79 | 0.00 |
| 22 | 1.3 | 1.2 | 2831.1 | 0.96 | 0.96 | 0.00 |

min-over-layers: J-lens HMR 1.2 / pass@10 0.96; logit HMR 1.1 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.6046 | +0.0086 | -0.6739 | 87.5% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `... weight. Addition of SO2 to the organic phase before the`)

**skip4** (model's actual next token: `' addition'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` `
` `ses` `Register` `government` ` Government` ` Security` `present` ` Infrastructure` `assembly` | `izabeth` `manuel` `imab` `slant` `PI` `uese` `achus` `ocardi` `)\[` `utum` |
| 1 | `ses` ` movement` ` motor` ` Motor` ` highway` ` p` ` Assembly` ` compiler` `bor` ` constructor` | `achus` `PI` `manuel` `)\[` `imab` `anse` `;\|` `ajor` `PR` ` thereof` |
| 2 | ` written` ` reading` ` book` ` changed` ` information` ` past` ` other` ` block` ` average` ` work` | `PI` `;\|` `)\[` `manuel` `imab` `achus` `PR` `erad` `anse` `uckles` |
| 3 | ` distribution` ` average` ` new` ` price` `ses` ` table` ` hotels` ` general` ` places` ` other` | `)\[` `achus` `amss` `)}}{\` `;\|` `anse` `kees` `ousseau` `rosse` `ritz` |
| 4 | ` states` `GRect` ` dress` ` locations` ` location` ` relation` ` contracts` ` hotels` ` state` ` relations` | `kees` `ajor` `;\|` `rosse` `PR` `PI` `bage` ` thereof` `tub` `ividual` |
| 5 | ` "` ` development` ` reference` ` release` ` transfer` ` location` `:"` ` administration` ` traffic` `."` | `chter` `apine` `�` `bage` ` emergence` ` colle` `oxyl` `ajor` `$).` ` finally` |
| 6 | ` “` ` development` ` table` ` formation` ` "` ` initiation` ` said` ` production` ` approval` ` training` | ` finally` ` emergence` `!--` `]--` `ritz` ` latter` `oxyl` `chter` `ractice` `hens` |
| 7 | ` development` ` “` ` production` `.”` ` initiation` `…”` ` said` ` present` `production` ` ‘` | `cie` `chter` `!--` ` invention` `tub` `apine` `OY` `--**` `ajor` ` emergence` |
| 8 | ` said` ` has` ` “` ` means` ` development` ` execution` ` ‘` ` instruction` ` present` ` initiation` | `cie` `rapeut` `onset` `inical` `REEK` `chter` `!--` `unnumbered` `bsite` ` colle` |
| 9 | ` said` ` initiation` ` execution` ` “` ` "` ` mentioned` ` development` ` means` ` has` ` is` | `cie` `cci` `tub` ` colle` `REEK` `onset` `gem` `)",` ` aforementioned` `rapeut` |
| 10 | ` said` ` also` ` mentioned` ` has` ` execution` ` initiation` ` is` ` will` ` development` ` other` | `inical` ` aforementioned` ` afore` `cci` ` pren` ` disappe` ` colle` `gem` `onal` `cie` |
| 11 | ` said` ` master` ` has` ` production` ` preparation` ` assembly` ` formation` ` development` ` itself` ` mentioned` | `unnumbered` `inical` ` pren` ` disappe` `iki` `lotte` `)\<` ` aforementioned` `gem` `agin` |
| 12 | ` formation` ` preparation` ` conversion` ` bridge` ` transformation` ` crystallization` ` mixing` ` assembly` ` compilation` `?)` | ` disappe` `inical` `sembling` ` pren` `bsite` ` '[` `unnumbered` `REEK` `ikov` `gem` |
| 13 | `?).` ` preparation` ` conversion` `).` ` initiation` ` transformation` ` mixing` ` commencement` ` formation` ` crystallization` | ` disappe` `sembling` ` orche` `ming` `gem` ` initiation` `olis` `inical` ` formation` ` onset` |
| 14 | ` crystallization` ` preparation` ` initiation` ` formation` `ttp` ` conversion` ` commencement` ` completion` ` polymerization` ` composition` | ` disappe` ` orche` `ming` `sembling` ` commencement` ` initiation` ` polymerization` ` afore` ` onset` ` lup` |
| 15 | ` initiation` ` commencement` ` completion` ` preparation` ` invention` ` onset` ` arrival` ` introduction` ` fabrication` ` formation` | ` disappe` ` aforementioned` ` introduction` ` onset` ` latter` ` orche` ` afore` ` incorporation` `inical` ` initiation` |
| 16 | ` polymerization` ` mixer` ` formation` ` incorporation` ` mixing` ` synthesis` ` formulation` ` injection` ` introduction` ` fabrication` | ` onset` ` formation` ` aforementioned` ` beginning` ` latter` ` afore` ` desired` ` commencement` ` introduction` ` initiation` |
| 17 | ` polymerization` ` synthesis` ` formation` ` incorporation` ` initiation` ` nucleation` ` preparation` ` insertion` ` mixer` ` injection` | ` beginning` ` introduction` ` formation` ` preparation` ` commencement` ` initiation` ` onset` ` addition` ` final` ` latter` |
| 18 | ` polymerization` ` polymer` ` nucleation` ` synthesis` ` crystallization` ` preparation` ` emulsion` ` precipitation` ` mixture` ` mixing` | ` final` ` preparation` ` beginning` ` polymerization` ` addition` ` subsequent` ` formation` ` commencement` ` mixing` ` mixture` |
| 19 | ` polymerization` ` polymer` ` precipitation` ` crystallization` ` synthesis` ` nucleation` ` preparation` ` dissolution` ` mixing` ` copolymer` | ` addition` ` beginning` ` preparation` ` first` ` formation` ` subsequent` ` final` ` mixture` ` introduction` ` initial` |
| 20 | ` polymerization` ` polymer` ` mixing` ` crystallization` ` dissolution` ` synthesis` ` mixture` ` copolymer` ` nucleation` ` precipitation` | ` addition` ` subsequent` ` beginning` ` preparation` ` formation` ` mixing` ` first` ` final` ` mixture` ` introduction` |
| 21 | ` polymerization` ` polymer` ` mixing` ` precipitation` ` synthesis` ` crystallization` ` nanoc` ` evaporation` ` dissolution` ` addition` | ` addition` ` mixing` ` subsequent` ` final` ` formation` ` preparation` ` beginning` ` mixture` ` reaction` ` first` |
| 22 | ` polymerization` ` mixing` ` polymer` ` precipitation` ` addition` ` synthesis` ` formation` ` aqueous` ` nucleation` ` mixture` | ` addition` ` mixing` ` formation` ` reaction` ` mixture` ` beginning` ` subsequent` ` preparation` ` polymerization` ` final` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... and many other legends helped bring the club countless trophies from`)

**skip4** (model's actual next token: `' the'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` `rest` `PP` `er` `all` `where` `t` `e` `sam` `lin` | ` scratch` `;\|` `within` `scr` `erl` `among` ` within` ` follower` `ERY` `ividual` |
| 1 | `s` ` where` ` Mang` `where` `y` `t` `rest` ` here` `2013` `yang` | `;\|` ` scratch` `imab` `scr` `ambers` `ager` `within` `hereinafter` `●` `athan` |
| 2 | ` where` ` Austin` ` records` ` below` ` then` `where` ` data` ` speakers` ` locations` `s` | `;\|` ` scratch` `scr` `imab` `uckles` `  ` `manuel` `ager` `inters` `)[` |
| 3 | ` locations` `fred` ` where` ` library` ` libraries` ` studios` `where` `y` `ei` ` loss` | `;\|` `here` ` scratch` `inside` `imab` `within` `agm` `manuel` ` either` `inters` |
| 4 | ` iTunes` ` Penguin` ` apples` `ages` `ager` ` Crick` ` perspective` ` apple` ` 2013` ` apps` | `here` ` scratch` `this` ` these` `;\|` ` either` `these` `scr` `agers` ` mouth` |
| 5 | ` Crick` `ager` ` Penguin` `ages` `stage` `aging` ` apples` `agers` ` Guang` ` scratch` | `among` ` scratch` `here` ` throughout` ` amongst` ` uniformly` ` these` `acrylate` ` theirs` `scr` |
| 6 | ` Crick` ` scratch` ` where` ` records` `ages` `ager` ` across` `where` ` around` ` bearing` | `here` `among` ` scratch` ` these` ` their` ` those` ` amongst` ` different` `scr` ` throughout` |
| 7 | ` where` `ages` ` scratch` ` below` ` here` ` to` `som` ` inside` `ucc` `where` | `scr` `here` ` their` `đ` `among` ` either` ` these` `aria` ` scratch` `akk` |
| 8 | ` where` `ucc` ` to` ` across` ` Crick` ` below` ` inside` ` sources` `ucci` `where` | `scr` `among` ` their` ` throughout` ` which` ` these` ` across` `inside` `yr` ` our` |
| 9 | ` across` ` where` ` into` ` within` ` to` ` locations` ` among` ` which` ` below` ` location` | `among` ` their` ` across` ` amongst` ` around` ` throughout` ` among` ` both` `scr` ` the` |
| 10 | ` across` ` where` ` into` ` below` ` to` ` within` ` which` ` among` ` successful` ` inside` | `among` ` amongst` ` across` ` among` ` throughout` ` their` ` all` ` the` ` around` ` both` |
| 11 | ` across` ` which` ` their` ` into` ` where` ` being` ` among` ` when` ` to` ` the` | ` their` ` amongst` `among` ` all` ` throughout` ` the` ` both` ` among` ` across` ` dere` |
| 12 | ` scratch` ` across` ` respectively` `).` `."` ` various` ` successes` `.”` ` each` ` different` | ` their` ` the` ` all` ` across` ` under` ` both` ` various` ` an` `among` ` dere` |
| 13 | `).` ` being` `:` ` to` ` which` ` among` ` various` ` across` ` rock` ` within` | ` their` ` various` ` every` ` onwards` ` both` ` years` ` numerous` ` the` ` multiple` ` day` |
| 14 | ` competitions` ` onwards` ` award` ` awards` ` within` ` contests` ` various` ` which` ` competitors` ` years` | ` scratch` ` onwards` ` across` ` various` `inside` ` their` `among` ` amongst` `scr` ` birth` |
| 15 | ` competitions` ` award` ` awards` ` across` ` within` ` winning` ` around` ` victories` ` contests` ` competition` | ` across` ` around` ` scratch` ` various` ` throughout` `among` ` their` ` numerous` ` amongst` `around` |
| 16 | ` competitions` ` games` ` tournaments` ` awards` ` FIFA` ` trophy` ` victories` ` contests` ` competitors` ` award` | ` various` ` the` ` around` ` across` ` different` ` all` ` which` ` many` ` throughout` ` within` |
| 17 | ` competitions` ` games` ` tournaments` ` FIFA` ` soccer` ` victories` ` stadium` ` championships` ` Champions` ` clubs` | ` their` ` different` ` the` ` various` ` all` ` previous` ` 1` ` other` ` a` `,` |
| 18 | ` competitions` ` tournaments` ` clubs` ` FIFA` ` championships` ` games` ` Champions` ` championship` ` soccer` ` trophy` | ` the` ` all` ` different` ` various` ` a` ` top` `,` ` high` ` around` ` 1` |
| 19 | ` competitions` ` FIFA` ` clubs` ` tournaments` ` Champions` ` football` ` games` ` championships` ` soccer` ` Soccer` | ` all` ` the` ` top` ` different` ` various` ` their` ` a` ` time` ` high` ` which` |
| 20 | ` clubs` ` Champions` ` FIFA` ` competitions` ` tournaments` ` championships` ` football` ` soccer` ` winning` ` stadium` | ` the` ` all` ` time` ` top` ` various` ` their` ` different` ` both` ` a` ` which` |
| 21 | ` FIFA` ` football` ` tournaments` ` clubs` ` competitions` ` their` ` championships` ` soccer` ` Champions` ` stadium` | ` the` ` all` ` various` ` top` ` both` ` their` ` different` ` time` ` that` ` other` |
| 22 | ` the` ` their` ` football` ` various` ` clubs` ` numerous` ` FIFA` ` its` ` both` ` glory` | ` the` ` various` ` all` ` their` ` both` ` that` ` which` ` many` ` time` ` every` |

### Held-out prompt 3

(128-token window; last 12 tokens: `...192)) + 3*sqrt(192)*-2)**2`)

**skip4** (model's actual next token: `'.'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` `som` `nd` `rit` `2002` `ax` `d` `sg` `bis` `rb` | `ipping` `attention` `xff` `nes` `-*` `;\|` `tract` ` Prem` `iple` `<!--` |
| 1 | `s` `nd` `d` `som` `a` `DNA` `rb` `AST` `sd` `n` | `new` `xff` `;\|` `now` `paths` ` right` `rad` `Lat` `lif` `New` |
| 2 | `s` `nd` `Chan` ` Stevens` `122` `a` `som` `d` ` Bard` `bis` | `new` `Lat` ` new` `living` `flu` `herty` `fast` `trans` ` \-` `;\|` |
| 3 | `a` `s` `nd` ` ss` `*(` `272` ` machines` `&` ` splicing` `shore` | ` new` `new` `;\|` ` derivatives` `herty` `-[` `[/` ` \|=` `fast` ` terms` |
| 4 | `nd` ` "` `*(` `::::` `(&` `.....` `&` `....` `'s` `\|\|` | ` new` `rah` `-[` `[/` ` terms` `new` `herty` ` belongs` `ras` `New` |
| 5 | `nd` `.....` `::::` `.` `
` `(&` `.&` `*(` `.(` `.[` | `*(` ` rect` ` circles` ` ([` `*(-` `*-` ` other` ` new` `circle` `ras` |
| 6 | `.....` `..` `.` `*(` `......` `........` `....` `.(` `!.` `
` | ` other` ` rect` `.` `ras` ` and` `*(` `ra` ` new` `draw` ` a` |
| 7 | `.` `!.` `*(` `.....` `*.` `®` `::::` `.[` `:#` ` panc` | `)*-` `today` ` rect` `니다` ` Today` `loat` `raining` ` cre` `*(` `&+` |
| 8 | `.` `
` `*.` `!.` `.*` `.(` `<\|endoftext\|>` ` cups` ` guests` ` cre` | `)*-` `*(` `*-` `fur` `grav` ` other` `)*` ` cre` `ras` ` +` |
| 9 | `
` `.` `*.` `.*` `.?` `.(` `Ã` `!.` `?.` `.[` | `)*-` ` cre` `*(` `*-` `Medium` `*.` ` +` `)*` `quad` `.` |
| 10 | `*.` `.` `.*` `
` ``.` ` ` `.[` ` "*` `.**` ` "` | ` cre` `*-` `)*-` `central` `)*` `*` `*.` `fur` `*(` ` gest` |
| 11 | `.` `*.` `().` `.[` ` ().` `!.` `?.` `.(` `.*` `
` | `}.` `*.` `)*-` `central` `。` `).` `.$$` `uren` ` cre` `.).` |
| 12 | `*.` `.*` `.` `.(` `.[` `().` `<\|endoftext\|>` `*` `*-` `.**` | `.` `*.` `}.` `*` `<\|endoftext\|>` `.$$` `.*` `.(` `central` ` cre` |
| 13 | `*.` `.*` `.` `*-` `.[` `.**` `*` `%.` `}.` `**.` | `.` `*.` `*` `.*` `*-` `).` `est` `}.` `<\|endoftext\|>` `ere` |
| 14 | `*.` `*-` `}.` `%.` `].` `.*` `.` `+.` `.[` `_.` | `.` `*.` ` +` `*-` `*` `).` ` -` `。` ` Ring` `central` |
| 15 | `*.` `*-` `.*` `}.` `*(` `().` `.**` `.` `__.` `.^[@` | `.` ` +` ` -` `*.` `*` `).` `in` `*-` `。` `}.` |
| 16 | `*.` `*-` `.*` `}.` `.**` `*` `)*-` `**.` ` ==` `*(` | `.` ` -` ` +` `*` ` (` ` is` `
` `<\|endoftext\|>` ` ` ` in` |
| 17 | `*.` `*-` ` ==` ` +` `+.` `*` `.*` `^.` `>.` `#.` | `.` ` +` ` -` `*` ` (` `
` ` ` ` and` `  ` `,` |
| 18 | `*.` `*-` ` +` `.` `+.` ` ==` `.*` `*` `)*-` `}.` | `.` ` -` ` +` `,` ` (` `*` ` in` ` ` ` and` ` *` |
| 19 | `*.` `*-` `.` ` +` `.*` ` ==` `}.` `*` `+.` `).` | `.` ` -` ` +` `*` ` in` `,` ` *` ` (` ` ` `<\|endoftext\|>` |
| 20 | `*.` `.` `*-` `.*` ` +` `).` `}.` `*` `+.` ` ==` | `.` ` +` ` -` `*` `*-` `).` `*.` `,` ` in` ` *` |
| 21 | `.` `*-` ` +` `*.` `.*` `*` `+.` `}.` ` ==` `)*-` | `.` ` +` `*` ` -` `,` `*-` ` in` ` *` `
` ` (` |
| 22 | `.` ` +` `*-` `*.` `*` `.*` `)*-` `+.` `}.` `*(-` | `.` ` +` `*` `*-` ` -` `,` `*.` ` ` ` in` ` a` |
