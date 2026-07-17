# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-410m (revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f)
- device: mps, dtype: float32, seed: 2
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-18T01:02:12, 944.9s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 10, 11]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [12, 13, 14]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.5564 vs random +0.0161

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
| 0 | 513.9 | 3234.6 | 2282.0 | 0.00 | 0.00 | 0.00 |
| 1 | 1019.7 | 2129.3 | 3631.1 | 0.00 | 0.00 | 0.00 |
| 2 | 369.3 | 3305.3 | 3222.6 | 0.00 | 0.00 | 0.01 |
| 3 | 206.6 | 3182.7 | 3087.2 | 0.03 | 0.00 | 0.00 |
| 4 | 200.7 | 4239.2 | 2965.6 | 0.03 | 0.00 | 0.00 |
| 5 | 327.6 | 3500.7 | 4648.0 | 0.00 | 0.00 | 0.00 |
| 6 | 262.9 | 3626.2 | 4163.4 | 0.00 | 0.00 | 0.00 |
| 7 | 163.4 | 2994.3 | 2686.8 | 0.00 | 0.00 | 0.01 |
| 8 | 184.4 | 1193.3 | 3787.6 | 0.03 | 0.00 | 0.00 |
| 9 | 117.6 | 3013.9 | 2699.9 | 0.03 | 0.00 | 0.00 |
| 10 | 58.0 | 1859.4 | 2765.8 | 0.06 | 0.00 | 0.00 |
| 11 | 641.9 | 2604.2 | 3701.5 | 0.00 | 0.00 | 0.00 |
| 12 | 375.7 | 2670.6 | 3114.4 | 0.00 | 0.00 | 0.00 |
| 13 | 6.6 | 30.5 | 2544.7 | 0.28 | 0.03 | 0.01 |
| 14 | 2.8 | 158.1 | 3952.5 | 0.69 | 0.00 | 0.00 |
| 15 | 16.4 | 91.2 | 1716.6 | 0.14 | 0.03 | 0.00 |
| 16 | 10.1 | 58.8 | 1830.4 | 0.22 | 0.06 | 0.00 |
| 17 | 1.4 | 1.1 | 2573.2 | 1.00 | 1.00 | 0.00 |
| 18 | 1.6 | 1.5 | 3950.4 | 1.00 | 1.00 | 0.00 |
| 19 | 2.4 | 2.5 | 1964.3 | 1.00 | 0.94 | 0.00 |
| 20 | 3.2 | 5.0 | 2024.2 | 0.94 | 0.67 | 0.01 |
| 21 | 5.0 | 30.9 | 1897.1 | 0.81 | 0.11 | 0.00 |
| 22 | 7.8 | 51.6 | 1774.4 | 0.56 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.2 / pass@10 1.00; logit HMR 1.1 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 594.5 | 3864.1 | 2759.8 | 0.00 | 0.00 | 0.00 |
| 1 | 4362.4 | 1696.3 | 3217.8 | 0.00 | 0.00 | 0.00 |
| 2 | 1972.0 | 3473.4 | 3239.4 | 0.00 | 0.00 | 0.00 |
| 3 | 1284.6 | 3572.8 | 2149.7 | 0.00 | 0.00 | 0.00 |
| 4 | 1093.5 | 3320.2 | 2449.2 | 0.00 | 0.00 | 0.00 |
| 5 | 974.4 | 2106.8 | 3020.3 | 0.00 | 0.00 | 0.00 |
| 6 | 781.6 | 4384.9 | 4432.9 | 0.00 | 0.00 | 0.00 |
| 7 | 1160.3 | 2021.2 | 4182.6 | 0.00 | 0.00 | 0.00 |
| 8 | 782.4 | 2897.9 | 3029.3 | 0.00 | 0.00 | 0.01 |
| 9 | 685.7 | 2274.9 | 2670.8 | 0.00 | 0.00 | 0.01 |
| 10 | 590.8 | 203.5 | 2392.5 | 0.00 | 0.03 | 0.00 |
| 11 | 825.0 | 1377.2 | 3105.4 | 0.00 | 0.00 | 0.01 |
| 12 | 96.7 | 562.0 | 3413.4 | 0.06 | 0.00 | 0.00 |
| 13 | 7.0 | 318.5 | 1982.8 | 0.19 | 0.00 | 0.00 |
| 14 | 5.8 | 537.4 | 2357.8 | 0.33 | 0.00 | 0.01 |
| 15 | 4.0 | 231.2 | 2327.7 | 0.36 | 0.00 | 0.01 |
| 16 | 2.6 | 61.5 | 1740.5 | 0.53 | 0.08 | 0.00 |
| 17 | 1.6 | 1.9 | 2131.7 | 0.89 | 0.94 | 0.00 |
| 18 | 1.3 | 1.4 | 2889.0 | 0.86 | 0.92 | 0.01 |
| 19 | 1.2 | 1.2 | 2122.4 | 0.86 | 0.97 | 0.00 |
| 20 | 1.2 | 1.1 | 1753.2 | 0.89 | 0.97 | 0.00 |
| 21 | 1.2 | 1.1 | 1999.4 | 0.92 | 1.00 | 0.00 |
| 22 | 1.2 | 1.1 | 2134.0 | 0.92 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.2 / pass@10 0.92; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 2247.6 | 9312.7 | 3486.5 | 0.00 | 0.00 | 0.00 |
| 1 | 559.9 | 6446.2 | 5285.1 | 0.00 | 0.00 | 0.00 |
| 2 | 938.4 | 8306.8 | 5368.1 | 0.00 | 0.00 | 0.00 |
| 3 | 511.2 | 5045.2 | 5986.2 | 0.00 | 0.00 | 0.00 |
| 4 | 661.5 | 7144.9 | 4160.5 | 0.00 | 0.00 | 0.00 |
| 5 | 355.3 | 4939.0 | 3520.7 | 0.00 | 0.00 | 0.00 |
| 6 | 342.5 | 5752.8 | 5641.4 | 0.00 | 0.00 | 0.00 |
| 7 | 255.5 | 4110.1 | 6305.6 | 0.00 | 0.00 | 0.00 |
| 8 | 150.0 | 2062.7 | 6211.6 | 0.00 | 0.00 | 0.00 |
| 9 | 176.3 | 6150.7 | 4471.3 | 0.00 | 0.00 | 0.01 |
| 10 | 11.1 | 216.0 | 5865.6 | 0.19 | 0.00 | 0.00 |
| 11 | 19.8 | 212.8 | 3452.3 | 0.12 | 0.00 | 0.00 |
| 12 | 44.9 | 23.9 | 3492.3 | 0.06 | 0.19 | 0.00 |
| 13 | 1.8 | 1.3 | 5915.1 | 0.94 | 0.94 | 0.00 |
| 14 | 1.4 | 1.4 | 4140.7 | 0.94 | 1.00 | 0.00 |
| 15 | 1.4 | 1.3 | 4687.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.4 | 1.0 | 4950.8 | 1.00 | 1.00 | 0.00 |
| 17 | 1.2 | 1.0 | 5793.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.1 | 1.0 | 4840.9 | 1.00 | 1.00 | 0.00 |
| 19 | 1.0 | 1.0 | 4740.7 | 1.00 | 1.00 | 0.00 |
| 20 | 1.0 | 1.0 | 3693.3 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 4328.3 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 4541.5 | 1.00 | 1.00 | 0.01 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 2075.5 | 933.5 | 3186.5 | 0.00 | 0.00 | 0.00 |
| 1 | 3388.8 | 1090.8 | 3879.9 | 0.00 | 0.00 | 0.00 |
| 2 | 599.4 | 1566.1 | 2771.8 | 0.00 | 0.00 | 0.01 |
| 3 | 566.0 | 168.7 | 3388.1 | 0.00 | 0.04 | 0.00 |
| 4 | 585.7 | 159.4 | 2487.8 | 0.00 | 0.00 | 0.00 |
| 5 | 100.5 | 23.1 | 3501.5 | 0.04 | 0.04 | 0.00 |
| 6 | 209.5 | 22.6 | 3684.3 | 0.00 | 0.04 | 0.00 |
| 7 | 84.6 | 22.7 | 3858.8 | 0.04 | 0.04 | 0.00 |
| 8 | 21.0 | 20.5 | 2830.0 | 0.04 | 0.08 | 0.00 |
| 9 | 74.4 | 27.3 | 4453.4 | 0.04 | 0.08 | 0.00 |
| 10 | 14.7 | 13.7 | 3113.0 | 0.17 | 0.12 | 0.00 |
| 11 | 19.6 | 9.4 | 2971.3 | 0.12 | 0.17 | 0.00 |
| 12 | 8.6 | 12.9 | 2125.5 | 0.12 | 0.17 | 0.00 |
| 13 | 4.3 | 4.4 | 2645.8 | 0.33 | 0.33 | 0.00 |
| 14 | 3.3 | 3.5 | 3625.3 | 0.42 | 0.42 | 0.00 |
| 15 | 3.0 | 2.4 | 2938.7 | 0.54 | 0.58 | 0.00 |
| 16 | 2.7 | 1.8 | 2650.1 | 0.62 | 0.67 | 0.00 |
| 17 | 2.8 | 1.9 | 1853.7 | 0.62 | 0.79 | 0.00 |
| 18 | 2.3 | 1.7 | 3189.3 | 0.62 | 0.75 | 0.00 |
| 19 | 2.0 | 1.4 | 2861.3 | 0.71 | 0.83 | 0.00 |
| 20 | 1.8 | 1.4 | 1975.8 | 0.79 | 0.83 | 0.00 |
| 21 | 1.5 | 1.4 | 1352.8 | 0.88 | 0.79 | 0.00 |
| 22 | 1.3 | 1.2 | 2831.1 | 0.92 | 0.96 | 0.00 |

min-over-layers: J-lens HMR 1.3 / pass@10 0.96; logit HMR 1.1 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.5564 | +0.0161 | -0.6620 | 81.2% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `... the waggle dance. The information contained in the w`)

**skip4** (model's actual next token: `'ag'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` WI` ` w` `nw` `iw` `dw` `w` `wm` `mw` `ow` `ew` | ` either` `within` ` within` `�` `ro` `aster` `TO` `inter` ` ` ` U` |
| 1 | ` w` `ew` ` WI` `dw` `wm` `iw` `wen` ` dw` ` WM` `nw` | `/` ` within` ` either` `within` `�` ` itself` `ro` ` U` `-` `inter` |
| 2 | ` w` `ew` ` dw` ` WI` `nw` `wen` ` ow` ` wit` ` Wit` `dw` | `/` ` either` ` within` `-` `if` `�` ` ` ` or` ` therein` `.` |
| 3 | ` w` `ew` ` dw` ` WI` ` ow` `wm` `ow` ` wig` ` wi` ` wit` | `if` `/` ` either` `/@` `inter` `its` ` Font` `.` ` therein` `intr` |
| 4 | ` w` `ew` ` ow` ` WI` ` dw` `ow` `wn` `wm` `wen` `wd` | `if` `/` ` either` ` if` `int` ` or` `eth` `[` `its` `-` |
| 5 | `ew` ` Winn` ` w` `dw` ` ow` ` dw` `ow` ` WI` `nw` ` pen` | `if` `immer` `int` `/` `itter` `imm` `its` `ret` `ith` `eth` |
| 6 | `ew` `ow` ` w` ` WI` ` pen` ` ow` ` Winn` `dw` ` dw` `iw` | `if` `/` `/@` `int` `intr` ` or` `its` `itter` ` preferably` ` generally` |
| 7 | `ow` ` w` `aw` ` WI` `ew` ` Winn` `dw` ` ow` ` pen` ` wart` | `inter` ` Pand` `imm` `ch` `/` `CH` `intr` ` circumferential` `chi` `inters` |
| 8 | ` w` ` WI` `ow` ` wart` ` wi` ` wit` ` ow` ` pen` ` wig` ` Winn` | `ht` `imm` `anti` `/` `-` `cel` ` Geneva` `inter` `Param` `sl` |
| 9 | `aw` `ows` `ow` ` w` ` pen` `ability` `w` ` words` `'` ` span` | `ht` `imm` `-` `ann` `inn` ` compan` ` Geneva` `ch` `inter` ` family` |
| 10 | `\'` ` w` ` pen` `ows` ` \'` ` \"` `aw` `.."` `'` `‟` | `ht` `sm` `ann` `imm` `sl` ` Influ` `su` ` compan` `Param` `inn` |
| 11 | `aw` `ag` ` w` `agram` `?"` `!"` `ability` ` kw` `ows` ` ward` | ` Influ` `ht` `ch` `ag` `ann` `yr` `sm` `sur` `nu` `ac` |
| 12 | `aw` `?"` ` \"` `ow` ` \'` `ag` `!"` ` Winn` ` w` `)."` | `sm` `agin` `ag` `nu` ` Influ` `sl` `g` `tag` `de` `map` |
| 13 | `?"` `)"` `?”` `)."` `'?"` `!"` `)",` ` \"` `"?` `aw` | `-` `g` `nu` `"` `ug` `ch` `ag` `/` `(` `sm` |
| 14 | `)"` `?"` `'?"` ` corrid` `)."` `)",` `?”` `agram` `ags` `awn` | `inc` `nu` `ag` `sm` `SM` `ch` `g` `ue` `uk` `ug` |
| 15 | ` corrid` `agues` `‟` `abbing` ` _"` `aw` ` "~` ` Wrest` ` lyrics` `oken` | `ag` `inc` `-` `index` `tf` `ug` `aff` `us` `agging` `agin` |
| 16 | `*"` `‟` ` *"` `agg` `agram` `\"` `ags` `walker` `oken` `agging` | `"` `ag` `inc` `-` `
` `us` `g` `*` `'` `ug` |
| 17 | `agram` ` corrid` `agg` ` gestures` ` gesture` `*"` `~*` `ags` `agging` `agt` | `ag` `inc` `index` `ach` `agin` `-` `ug` `ob` `us` `es` |
| 18 | `agg` `ags` `agram` `ag` `agging` ` gestures` `agt` `agger` `agged` `agm` | `ag` `-` `ug` `ob` `ra` `index` `m` `ach` `agging` `ad` |
| 19 | `ag` `ags` `agg` `agram` `agging` `agm` `agers` `rag` `ager` `agger` | `ag` `ach` `-` `ug` `index` `agging` `ags` `ob` ` sw` `agg` |
| 20 | `ag` `ags` `agg` `agging` `agram` `agin` `agger` `agm` `agged` `ager` | `ag` `-` `agg` `ach` `ags` `index` `ad` `ob` `agging` `inc` |
| 21 | `ag` `ags` `agg` `agging` `agm` `avy` `ager` `agger` `aff` `aga` | `ag` `-` `ags` `aff` `agg` `ug` `ad` `av` `ob` `agging` |
| 22 | `ag` `ags` `agg` `agging` `ager` `ug` `igg` `agger` `aga` `agged` | `ag` `agg` `ags` `ug` `ad` `-` `arg` `agging` `ab` `av` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... the G20 protests was being taken out. Andrew Kendle`)

**skip4** (model's actual next token: `','`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `LE` ` Belle` `ole` `car` ` Lem` ` LE` `wl` `cars` `Le` `lei` | `aving` ` none` `bbe` `win` ` exchange` `aves` `210` `xt` `aders` `var` |
| 1 | `LE` ` Belle` ` Lakes` `ma` ` Ole` `lee` ` LE` ` Le` ` Carol` `lease` | `part` `�` `xt` `ipart` `ukin` ` part` `any` `dule` `immer` ` none` |
| 2 | ` Lakes` ` Belle` ` squeeze` ` Marvin` ` Dre` ` Cle` ` “` ` Leon` ` Lem` ` Lance` | ` [_` `ukin` ` therefrom` ` ...,` ` States` `bars` `spread` `wic` `part` `atura` |
| 3 | ` Wheel` ` Durham` ` Lakes` ` Delaware` ` wheel` `lle` ` Driver` ` Lew` ` Mile` ` Dew` | `[[` `try` `gs` `({{\` `free` ` ...,` ` [_` `ukin` `wy` ` Free` |
| 4 | ` colour` `colour` ` coloured` `stone` `mb` ` Bradford` ` Centre` ` Mile` ` Lad` `gl` | ` free` `gs` `free` `any` `ass` ` and` ` #` ` fre` ` freely` ` Free` |
| 5 | `
` ` Mouse` `’` ` Chair` `‐` ` Clean` `ch` `..` `…` `chair` | `worth` `,` ` and` ` charcoal` `oke` `ass` ` free` ` in` ` co` ` the` |
| 6 | `‐` ` Chair` `board` `–` ` Leg` `’` ` UK` `bl` `,` ` ‘` | `,` `ass` `rel` ` and` ` Mode` `oke` `ich` ` charcoal` `uc` ` in` |
| 7 | `’` ` ‘` `…..` `…` `….` `……` `‘` `..` `!` ` Chair` | `,` `rel` ` rat` ` on` `.` `ass` ` instead` ` flag` ` in` ` Jr` |
| 8 | `,` ` (` `--` `-` `'s` ` –` ` Chair` ` and` ` is` `–` | `,` `rel` `.` `ad` ` and` `'s` `ger` `ass` ` Good` ` (` |
| 9 | `,` ` (` ` is` `.` `(` `:` ` and` `)` `-` ` from` | `,` ` mac` `art` `'s` ` Mac` `.` `rel` ` National` `ich` ` [` |
| 10 | `,` ` (` `’` ` is` `:` ` and` `'s` ` has` `-` `.` | `,` `os` ` mac` ` Jr` `rel` `'s` ` under` `ade` `art` `-` |
| 11 | `:` `,` ` (` ` was` `’` ` has` ` and` `.` ` is` `-` | `rel` `,` `os` ` rel` `cu` `-` ` mac` ` was` `'s` ` National` |
| 12 | ` was` `’` `,` ` ‘` ` “` `'s` ` (` `:` ` and` `.` | `,` ` was` ` Jr` `-` ` (` `:` `.` `'s` ` has` `’` |
| 13 | `’` ` ‘` ` “` `:` `,` ` (` `'s` ` was` `.` ` and` | `,` ` was` ` (` `-` `'s` `.` ` and` ` is` `(` `/` |
| 14 | `'s` `’` ` Jr` `‘` `,` ` (` ` (@` `:` ` who` ` ‘` | `,` ` (` `'s` `-` `(` `/` ` was` ` and` ` who` ` is` |
| 15 | `’` `'s` ` Jr` `‘` ` (“` ` “` ` was` `http` `:` `,` | `,` ` (` `'s` `-` ` was` `.` `
` ` is` ` and` `:` |
| 16 | `'s` ` himself` `’` ` wrote` `:` ` his` ` writes` ` (@` ` Jr` ` was` | `,` ` (` `'s` `.` `
` `:` ` is` `-` ` and` ` was` |
| 17 | ` was` ` has` ` is` ` wrote` `'s` ` writes` ` explains` `’` ` describes` ` died` | `,` ` (` `
` `:` ` and` ` is` `-` `.` `'s` ` was` |
| 18 | ` writes` ` wrote` ` died` ` was` ` has` ` explains` ` describes` ` says` ` reports` ` believes` | `,` ` (` `
` ` and` `:` `-` `/` `.` ` in` ` is` |
| 19 | ` writes` ` wrote` ` died` ` was` ` explains` ` says` ` reports` ` has` ` describes` ` tells` | `,` ` (` ` and` `-` `
` `:` ` of` `/` ` in` ` is` |
| 20 | ` writes` ` wrote` ` says` ` died` ` reports` ` was` ` explains` ` said` ` tells` ` tweeted` | `,` ` (` `
` ` and` `:` ` of` ` in` ` is` `/` `-` |
| 21 | ` was` ` died` ` says` ` wrote` ` writes` ` reports` ` has` ` said` ` tells` ` told` | `,` ` (` ` of` ` and` `
` `-` ` in` `'s` ` was` `:` |
| 22 | ` was` ` wrote` ` writes` ` reports` ` says` ` died` ` has` ` told` ` tells` ` said` | `,` ` was` ` reports` ` of` ` wrote` ` has` ` (` ` reported` ` said` ` and` |

### Held-out prompt 3

(128-token window; last 12 tokens: `... lenders were unloading foreclosed houses, and they were selling`)

**skip4** (model's actual next token: `' them'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` sell` ` sold` ` sellers` ` seller` ` selling` `Sales` `sell` ` dealer` ` Sales` ` sells` | `ariate` `open` `group` `iem` `fre` ` avoid` `[]` `�` ` group` `++` |
| 1 | ` sell` ` sellers` ` sales` ` sold` ` selling` ` markets` `Sales` ` Sales` ` sale` ` seller` | `iem` `  ` ` them` ` that` ` it` ` ` `udge` `fre` ` 
` `golang` |
| 2 | ` sell` ` sold` ` sales` ` selling` ` sale` ` sellers` ` prices` ` dealers` `sell` ` sells` | ` them` `golang` `iem` `ariate` ` that` `fre` ` Private` ` ` `\*` ` freely` |
| 3 | ` sell` ` sales` ` sold` ` prices` ` sale` ` sellers` ` selling` ` dealers` ` dealer` ` Sale` | ` them` `iem` `golang` ` their` `ariate` `\*` ` Private` `  ` ` it` `see` |
| 4 | ` sell` ` sold` ` prices` ` sales` ` sellers` ` sale` ` selling` ` price` ` seller` ` Sale` | ` them` ` their` ` it` `  ` ` nothing` `achus` ` those` `iem` ` freely` ` they` |
| 5 | ` sell` ` sales` ` sellers` ` selling` ` sale` ` prices` ` sold` `sell` ` dealers` ` Sale` | ` them` ` nothing` ` it` ` their` ` either` ` just` ` this` ` these` ` property` ` within` |
| 6 | ` sales` ` sell` ` sale` ` selling` ` prices` ` sold` ` dealers` ` sellers` `sell` ` price` | ` them` ` just` ` today` ` nothing` ` again` `double` ` either` `just` ` property` `ousseau` |
| 7 | ` sales` ` sell` ` sale` ` sellers` ` dealers` ` prices` ` selling` ` sold` ` buyer` ` dealer` | ` them` `able` `just` ` just` ` their` ` again` `lee` ` nothing` `pee` ` away` |
| 8 | ` sales` ` sellers` ` sell` ` selling` ` sale` ` prices` ` dealers` ` sold` ` dealer` ` buyer` | ` them` ` away` `able` `pee` ` again` ` it` ` nothing` ` their` ` property` `just` |
| 9 | ` prices` ` sales` ` dealers` ` sell` ` sellers` ` sale` ` selling` ` markets` ` sold` ` market` | ` away` ` them` `able` `pee` ` again` `again` ` nothing` ` property` ` machines` `only` |
| 10 | ` prices` ` dealers` ` sales` ` markets` ` selling` ` sale` ` sell` ` sold` ` market` ` sellers` | ` away` `able` ` them` ` double` `________________________________` `oll` ` outright` `again` ` corps` ` machines` |
| 11 | ` prices` ` dealers` ` selling` ` sale` ` sales` ` markets` ` sell` ` price` ` sellers` ` sold` | ` away` ` them` `oll` `itage` `able` `________________________________` ` their` ` outright` ` these` ` corps` |
| 12 | ` prices` ` markets` ` sales` ` selling` ` sold` ` dealers` ` sell` ` sale` ` price` ` market` | ` them` ` away` ` their` `itage` `able` ` themselves` ` outright` ` again` `again` ` to` |
| 13 | ` prices` ` selling` ` sold` ` markets` ` dealers` ` price` ` auction` ` market` ` sale` ` sales` | ` them` ` away` ` their` ` off` ` themselves` ` all` `able` ` outright` `itage` ` lots` |
| 14 | ` prices` ` price` ` markets` ` dealers` ` sellers` ` selling` ` themselves` ` sale` ` sold` ` buyers` | ` them` ` away` ` all` ` off` ` their` ` outright` ` themselves` ` lots` ` enough` ` these` |
| 15 | ` prices` ` price` ` sellers` ` selling` ` auction` ` buyers` ` dealers` ` their` ` sale` ` themselves` | ` them` ` off` ` all` ` away` ` their` ` lots` ` themselves` ` outright` ` along` ` some` |
| 16 | ` sellers` ` prices` ` auction` ` price` ` selling` ` buyers` ` seller` ` sell` ` sale` ` priced` | ` all` ` to` `,` ` it` ` $` ` the` ` only` ` some` ` as` ` for` |
| 17 | ` them` ` their` ` theirs` ` they` ` themselves` ` sellers` ` prices` ` auction` `their` `them` | ` them` ` their` ` themselves` ` they` ` to` ` off` ` for` `,` ` the` ` as` |
| 18 | ` them` ` prices` ` buyers` ` sellers` ` auction` ` price` ` buyer` ` cheaper` ` sale` ` goods` | ` them` ` for` ` to` `,` ` on` ` the` ` in` ` off` ` at` ` all` |
| 19 | ` them` ` prices` ` auction` ` buyers` ` sellers` ` mortgages` ` foreclosure` ` price` `them` ` securities` | ` them` ` for` ` at` ` the` ` to` ` in` ` on` `,` ` off` ` a` |
| 20 | ` them` ` prices` ` mortgages` ` buyers` ` auction` ` foreclosure` ` sellers` ` items` `them` ` loans` | ` them` ` for` ` at` ` off` ` the` ` in` ` to` ` on` `,` ` as` |
| 21 | ` them` ` their` ` those` ` foreclosure` ` mortgages` ` houses` ` loans` ` homes` ` lots` ` tickets` | ` them` ` for` ` at` ` the` ` off` ` on` ` in` ` all` ` to` ` as` |
| 22 | ` them` ` their` ` the` ` some` ` those` ` off` ` lots` ` homes` ` foreclosure` ` houses` | ` them` ` at` ` for` ` off` ` the` ` some` ` on` ` their` ` to` ` in` |
