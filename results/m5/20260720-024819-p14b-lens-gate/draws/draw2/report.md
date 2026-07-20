# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-1.4b (revision fedc38a16eea3bd36a96b906d78d11d2ce18ed79)
- device: mps, dtype: float32, seed: 2
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-20T17:00:19, 4816.6s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 11, 12, 13, 14, 16]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [5, 6, 7, 8, 12, 13, 14, 15, 16]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7, 10]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [4, 7, 12, 13]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.4831 vs random +0.0004

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
| 0 | 479.5 | 540.8 | 1716.0 | 0.00 | 0.00 | 0.01 |
| 1 | 118.6 | 215.8 | 1572.2 | 0.03 | 0.00 | 0.00 |
| 2 | 131.7 | 152.7 | 2876.4 | 0.03 | 0.00 | 0.00 |
| 3 | 172.4 | 177.3 | 2972.6 | 0.03 | 0.00 | 0.00 |
| 4 | 106.4 | 424.8 | 1324.1 | 0.03 | 0.00 | 0.00 |
| 5 | 123.8 | 940.2 | 2387.3 | 0.03 | 0.00 | 0.00 |
| 6 | 213.0 | 755.6 | 1970.6 | 0.00 | 0.00 | 0.00 |
| 7 | 402.0 | 2870.0 | 2131.1 | 0.00 | 0.00 | 0.00 |
| 8 | 489.3 | 1233.0 | 2452.6 | 0.00 | 0.00 | 0.00 |
| 9 | 699.4 | 663.7 | 1938.0 | 0.00 | 0.00 | 0.01 |
| 10 | 978.8 | 578.8 | 3043.6 | 0.00 | 0.00 | 0.00 |
| 11 | 27.3 | 66.0 | 3067.3 | 0.03 | 0.03 | 0.00 |
| 12 | 28.4 | 126.7 | 1263.9 | 0.11 | 0.03 | 0.01 |
| 13 | 1.4 | 19.7 | 2232.8 | 0.97 | 0.17 | 0.00 |
| 14 | 3.2 | 67.5 | 1993.5 | 0.83 | 0.00 | 0.00 |
| 15 | 1.6 | 1.3 | 2361.4 | 1.00 | 1.00 | 0.00 |
| 16 | 1.6 | 1.6 | 2000.2 | 1.00 | 1.00 | 0.00 |
| 17 | 1.7 | 1.7 | 1806.7 | 1.00 | 1.00 | 0.01 |
| 18 | 1.9 | 2.7 | 1990.7 | 1.00 | 0.92 | 0.01 |
| 19 | 104.4 | 4.6 | 2553.7 | 0.00 | 0.78 | 0.00 |
| 20 | 6.5 | 23.6 | 4310.6 | 0.64 | 0.14 | 0.01 |
| 21 | 4.5 | 37.1 | 2258.6 | 0.81 | 0.06 | 0.00 |
| 22 | 4.7 | 41.4 | 2144.7 | 0.69 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.2 / pass@10 1.00; logit HMR 1.3 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 585.0 | 236.3 | 1636.0 | 0.00 | 0.03 | 0.00 |
| 1 | 374.6 | 162.1 | 925.1 | 0.00 | 0.03 | 0.01 |
| 2 | 371.5 | 130.9 | 1640.0 | 0.00 | 0.00 | 0.00 |
| 3 | 440.0 | 87.5 | 3315.0 | 0.00 | 0.03 | 0.00 |
| 4 | 393.1 | 237.4 | 2713.5 | 0.00 | 0.00 | 0.00 |
| 5 | 233.8 | 992.2 | 2050.1 | 0.00 | 0.00 | 0.00 |
| 6 | 269.2 | 368.5 | 1377.8 | 0.00 | 0.00 | 0.00 |
| 7 | 274.9 | 1746.2 | 1479.5 | 0.00 | 0.00 | 0.01 |
| 8 | 228.3 | 984.4 | 1832.0 | 0.00 | 0.00 | 0.00 |
| 9 | 448.2 | 347.1 | 3129.8 | 0.00 | 0.00 | 0.00 |
| 10 | 345.8 | 157.8 | 2288.8 | 0.00 | 0.00 | 0.00 |
| 11 | 41.6 | 34.1 | 2444.2 | 0.03 | 0.08 | 0.00 |
| 12 | 42.7 | 110.7 | 739.1 | 0.11 | 0.00 | 0.01 |
| 13 | 2.5 | 39.4 | 2347.2 | 0.72 | 0.08 | 0.00 |
| 14 | 2.2 | 44.3 | 2215.7 | 0.69 | 0.06 | 0.00 |
| 15 | 1.4 | 1.5 | 1818.7 | 0.92 | 0.97 | 0.01 |
| 16 | 1.3 | 1.4 | 1804.2 | 0.89 | 0.97 | 0.00 |
| 17 | 1.3 | 1.2 | 1704.3 | 0.89 | 0.97 | 0.01 |
| 18 | 1.2 | 1.1 | 2687.7 | 0.89 | 1.00 | 0.00 |
| 19 | 46.4 | 1.1 | 2085.0 | 0.00 | 1.00 | 0.00 |
| 20 | 1.2 | 1.3 | 2266.2 | 0.94 | 0.97 | 0.00 |
| 21 | 1.1 | 1.3 | 2945.9 | 1.00 | 0.97 | 0.00 |
| 22 | 1.1 | 1.5 | 1793.3 | 1.00 | 0.94 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 364.9 | 1999.8 | 3661.0 | 0.00 | 0.00 | 0.00 |
| 1 | 405.5 | 909.1 | 4382.1 | 0.00 | 0.00 | 0.00 |
| 2 | 226.3 | 574.9 | 4405.9 | 0.00 | 0.00 | 0.00 |
| 3 | 261.7 | 725.9 | 7653.7 | 0.00 | 0.00 | 0.00 |
| 4 | 375.4 | 654.1 | 5519.1 | 0.00 | 0.00 | 0.00 |
| 5 | 289.0 | 2177.0 | 4024.8 | 0.00 | 0.00 | 0.00 |
| 6 | 225.9 | 332.1 | 4874.3 | 0.00 | 0.00 | 0.00 |
| 7 | 124.7 | 551.5 | 4070.1 | 0.00 | 0.00 | 0.00 |
| 8 | 123.9 | 114.5 | 2367.4 | 0.00 | 0.06 | 0.00 |
| 9 | 95.8 | 85.1 | 4498.2 | 0.06 | 0.06 | 0.00 |
| 10 | 44.7 | 50.3 | 4101.0 | 0.06 | 0.06 | 0.00 |
| 11 | 10.6 | 2.9 | 3743.0 | 0.25 | 0.44 | 0.00 |
| 12 | 3.7 | 3.0 | 3447.1 | 0.50 | 0.38 | 0.00 |
| 13 | 1.1 | 1.0 | 4971.4 | 1.00 | 1.00 | 0.00 |
| 14 | 1.0 | 1.0 | 3523.4 | 1.00 | 1.00 | 0.00 |
| 15 | 1.0 | 1.0 | 4494.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.0 | 1.0 | 3822.6 | 1.00 | 1.00 | 0.00 |
| 17 | 1.0 | 1.0 | 4767.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.0 | 1.0 | 4665.5 | 1.00 | 1.00 | 0.00 |
| 19 | 48.9 | 1.0 | 4868.3 | 0.00 | 1.00 | 0.00 |
| 20 | 1.1 | 1.0 | 4720.0 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 2602.5 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 3637.8 | 1.00 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 301.8 | 139.0 | 1923.1 | 0.00 | 0.04 | 0.02 |
| 1 | 87.7 | 23.5 | 2188.1 | 0.04 | 0.04 | 0.00 |
| 2 | 246.2 | 21.1 | 3351.4 | 0.00 | 0.08 | 0.00 |
| 3 | 38.0 | 22.9 | 3327.9 | 0.08 | 0.04 | 0.00 |
| 4 | 11.8 | 22.4 | 3075.7 | 0.08 | 0.04 | 0.00 |
| 5 | 53.5 | 22.8 | 2714.2 | 0.04 | 0.04 | 0.00 |
| 6 | 31.4 | 22.5 | 2034.3 | 0.04 | 0.04 | 0.00 |
| 7 | 22.1 | 22.4 | 2644.8 | 0.04 | 0.04 | 0.00 |
| 8 | 20.3 | 15.1 | 1849.7 | 0.04 | 0.08 | 0.00 |
| 9 | 55.4 | 17.5 | 3901.3 | 0.08 | 0.08 | 0.00 |
| 10 | 75.6 | 14.4 | 2452.9 | 0.04 | 0.08 | 0.00 |
| 11 | 14.5 | 4.5 | 2339.0 | 0.21 | 0.33 | 0.00 |
| 12 | 5.7 | 6.3 | 1987.5 | 0.29 | 0.29 | 0.00 |
| 13 | 2.5 | 3.0 | 2385.8 | 0.54 | 0.50 | 0.00 |
| 14 | 2.6 | 1.8 | 4208.1 | 0.71 | 0.83 | 0.00 |
| 15 | 1.8 | 1.6 | 2123.7 | 0.75 | 0.88 | 0.00 |
| 16 | 1.6 | 1.5 | 1935.3 | 0.79 | 0.88 | 0.00 |
| 17 | 1.6 | 1.5 | 2237.0 | 0.88 | 0.88 | 0.00 |
| 18 | 1.5 | 1.6 | 2215.5 | 0.88 | 0.88 | 0.00 |
| 19 | 76.0 | 1.7 | 2369.4 | 0.00 | 0.75 | 0.00 |
| 20 | 2.5 | 1.7 | 4596.1 | 0.75 | 0.71 | 0.00 |
| 21 | 1.3 | 1.7 | 2914.9 | 0.96 | 0.79 | 0.00 |
| 22 | 1.2 | 1.5 | 2567.1 | 0.96 | 0.92 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 0.96; logit HMR 1.3 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.4831 | +0.0004 | -0.5541 | 56.2% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `... the waggle dance. The information contained in the w`)

**skip4** (model's actual next token: `'ag'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `iw` `uw` ` PW` ` CW` ` WI` `ww` `QW` `WD` `nw` ` WM` | `ry` `inger` `ert` `ilt` ` IB` `ille` `itching` `ah` ` const` `Simplify` |
| 1 | ` PW` `iw` `uw` `ww` ` LW` `zw` `ow` ` W` ` CW` ` AW` | `ry` ` IB` `ks` `inger` `Simplify` `ert` ` conce` `ah` `ani` `lan` |
| 2 | ` PW` `iw` `uw` ` W` ` LW` ` CW` `ww` ` MW` ` WB` ` WM` | `ry` `inger` `lan` `Simplify` `syntax` `l` `ks` `iley` `im` `ager` |
| 3 | `uw` ` PW` `iw` ` W` `ow` `ww` `ies` `ry` ` LW` ` MW` | `ry` `itching` `Simplify` `ert` `syntax` `DD` `ah` `inger` `tf` `ks` |
| 4 | `uw` ` W` `ww` `iw` ` PW` `W` ` DW` `wd` `w` ` WD` | `ry` `ager` `Simplify` `tf` `itching` `inger` `ilt` `ah` `syntax` `l` |
| 5 | ` W` `uw` `iw` `ww` `kw` `w` `wd` ` PW` ` CW` `W` | `ry` `ager` `tf` `ya` `itching` `ct` `aken` `DD` `Simplify` `ana` |
| 6 | `iw` `ww` ` WD` `uw` `WD` `rw` ` W` ` PW` `kw` `wd` | `ry` `DD` `ya` `ye` `tf` `esti` `ager` `boxes` `itching` `$` |
| 7 | `rw` `iw` ` PW` ` CW` `kw` ` WD` ` WM` `ws` ` LW` `WD` | `ry` `ana` `ya` `tf` `ks` `Ab` `ab` `fre` `dd` ` Commit` |
| 8 | `kw` `wl` ` LW` ` W` ` WD` ` PW` ` wire` `rw` ` WM` `iw` | `ry` ` Commit` `Fast` `ye` `anda` `Simplify` `dd` `ab` ` duly` ` theoret` |
| 9 | `iw` ` WM` ` PW` ` LW` ` WD` `WD` ` CW` ` W` ` Wi` ` Wendy` | `ari` `ry` `term` `ah` `anda` `map` `fs` `DD` `fre` `ana` |
| 10 | `WD` `iw` ` WD` ` W` ` LW` `rw` ` PW` `W` ` CW` `DW` | `aff` `-` `im` `bar` `F` `l` `g` `att` `ag` `ym` |
| 11 | ` LW` ` “` ` WA` ` CW` `’` ` W` `aw` `iw` `rw` ` AW` | `ag` `aff` `ari` `att` `ak` `ry` `anda` `ah` `map` `-` |
| 12 | ` LW` ` WA` ` GW` ` W` ` CW` ` AW` `iw` `aw` ` Witt` ` WM` | `ag` `g` `-` `ak` `aff` `att` `G` `ari` `ro` `map` |
| 13 | `GW` ` GW` ` ` ` −` `‐` ` Post` ` post` ` ―` `\/` ` “` | `ag` `-` `g` `aff` `att` `G` `ak` `ro` `f` `es` |
| 14 | ` ` ` “` `‐` `
  ` `…”` ` ‘` `=”` `”.` `’` ` ` | `ag` `-` ` a` ` H` ` A` ` U` ` T` `aff` ` (` `att` |
| 15 | `ag` `aging` `agm` `?",` `agma` `agg` `agner` `agging` `agt` ` signature` | `ag` `-` `g` `/` ` T` ` a` ` H` `es` `AG` ` (` |
| 16 | `ag` `agma` `agm` `agn` `agg` `agogue` `agt` `aging` `agging` `AG` | `ag` `-` `g` `/` ` a` `es` ` and` `,` ` T` `ak` |
| 17 | `ag` `agging` `idget` `agh` `agogue` `AG` `aga` `agon` `ager` `agons` | `ag` `-` `/` `,` ` and` `g` ` a` ` (` ` T` ` H` |
| 18 | `ag` `agging` ` *` `-*` `agg` `agh` `aga` `*` `agogue` `\$` | `ag` `-` `/` `,` ` (` ` a` ` and` ` A` ` T` ` H` |
| 19 | ` *` ` [*` ` \[*` ` **` ` \$` ` ` `[^` ` *"` ` (*` ` [@` | `-` `ag` `/` `,` ` a` ` (` ` and` ` A` `*` ` T` |
| 20 | ` ` ` ([@` ` {#` `
` ` [@` ` 
` `ag` `\'` `.[@` `[@` | `-` `ag` `/` `,` ` A` ` a` `*` ` (` ` T` ` H` |
| 21 | `ag` `agg` `igg` `agh` `agging` `ags` `agin` `addle` `ilt` `uffle` | `ag` `-` `/` `,` `ad` `*` ` A` ` H` ` (` `g` |
| 22 | `ag` `agg` `igg` `agin` `agging` `ags` `addle` `agh` `ak` `agt` | `ag` `-` `agg` `g` `ad` `/` `*` `ob` `b` `ig` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... the G20 protests was being taken out. Andrew Kendle`)

**skip4** (model's actual next token: `','`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `le` `LE` `lew` `ld` `e` `lle` `leg` `th` `a` `tle` | `as` `pre` `in` ` based` `ar` `etz` `an` `ins` ` each` `aders` |
| 1 | `le` `LL` `a` `yle` `ll` ` LE` ` %` `LE` `lle` `e` | `ann` `ins` `pre` `an` `venth` `cs` `a` `th` `anno` `ast` |
| 2 | ` LE` ` MN` `le` ` MB` `LL` ` KD` `mber` ` LL` ` rate` ` KL` | `ann` `ins` `jo` `ev` `a` `Mon` `ast` ` spec` `mon` `anno` |
| 3 | `'s` `ky` ` it` ` (` ` those` ` which` `ń` ` itself` ` these` ` am` | `ev` `a` `c` `.` `f` `ann` `ja` `bit` `j` `aw` |
| 4 | ` @` ` &` ` and` ` which` ` *` ` (` `/` `'s` ` a` `,` | ` Jr` `ev` `ann` `.` `c` `a` `kin` `va` `f` `anna` |
| 5 | ` behaviour` ` '` ` colour` ` (` ` characterised` ` &` ` coloured` ` which` ` recognise` ` Whilst` | `,` `.` `cl` `an` `ev` `c` `yn` ` (` `C` `:` |
| 6 | ` (` ` UK` ` &` ` Leigh` ` Whilst` ` Edmund` ` /` ` Sheffield` `/` `UK` | `,` ` Jr` ` (` `.` `:` ` III` `'s` `c` `(` `yn` |
| 7 | ` (` `,` ` and` ` UK` ` (£` `/` ` of` ` a` ` which` ` is` | ` (` `,` `.` `f` `'s` ` Jr` ` and` `(` `an` `a` |
| 8 | ` (` `,` ` and` ` or` ` of` ` John` ` Lewis` `ford` ` is` `c` | `,` ` (` ` Jr` `'s` `.` ` and` ` III` `en` `on` `’` |
| 9 | ` (` ` and` ` or` ` III` `,` ` John` ` Field` ` is` ` field` ` Jr` | ` (` `,` `'s` ` Jr` ` III` `.` `’` ` and` `an` ` of` |
| 10 | ` (` `
` `:` `,` ` ‘` ` and` ` “` ` is` `field` `’` | ` (` `,` `'s` `.` `an` ` Jr` `en` `o` ` was` ` of` |
| 11 | `’` ` ‘` `
` ` (` ` “` `,` `:` ` and` `'s` ` is` | `,` ` (` `'s` ` and` `.` ` was` `c` `on` `’` ` on` |
| 12 | `’` `.` ` ‘` ` (` `:` `,` `'s` ` was` `
` ` “` | `,` ` (` ` was` `'s` ` and` ` is` `.` `-` ` in` `c` |
| 13 | ` his` `’` `'s` ` was` ` he` ` ‘` ` himself` ` Jr` ` is` `:` | `,` ` (` ` was` ` and` `'s` `-` `.` ` is` `:` `
` |
| 14 | `’` ` ‘` ` “` ` was` `‘` ` ` ` –` ` ’` ` is` `'s` | ` (` `,` `
` ` was` `'s` `-` ` and` ` is` `.` ` [` |
| 15 | ` died` ` was` ` is` `'s` ` had` ` has` ` wrote` ` writes` ` did` ` dies` | `,` ` (` `'s` ` was` `-` ` and` `
` `.` ` is` `:` |
| 16 | ` was` ` died` ` is` ` has` ` had` `'s` ` wrote` ` writes` ` does` ` did` | `,` ` (` ` was` `'s` `-` `
` ` is` `.` ` and` `:` |
| 17 | ` died` ` was` ` is` `'s` ` has` ` dies` ` wrote` ` writes` ` wasn` ` had` | `,` ` (` `-` `'s` ` was` ` and` `.` `
` ` is` `:` |
| 18 | ` died` ` was` `'s` ` is` ` has` ` had` ` wasn` ` dies` ` wrote` ` UK` | `,` ` (` `-` `
` ` and` `'s` ` was` `.` ` in` ` is` |
| 19 | ` *` ` [*` ` **` ` (*` `[^` `\'` ` \[*` ` _` ` ` ` \$` | `,` ` (` `-` `
` ` and` ` was` `'s` `.` ` in` ` is` |
| 20 | `
` ` died` ` was` `'s` `’` ` he` ` is` ` had` ` has` `\'` | `,` ` (` `-` ` and` `
` ` was` `.` `'s` ` is` `/` |
| 21 | ` was` ` died` ` has` ` is` ` wasn` ` did` ` does` ` had` ` didn` `was` | `,` ` (` `-` ` and` `
` ` was` ` of` `.` ` in` `/` |
| 22 | ` was` ` died` ` has` ` had` ` did` ` is` `,` ` Davies` ` wrote` ` Smith` | `,` ` was` `-` `'s` ` (` ` of` ` and` ` died` ` is` `
` |

### Held-out prompt 3

(128-token window; last 12 tokens: `... lenders were unloading foreclosed houses, and they were selling`)

**skip4** (model's actual next token: `' them'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` sell` ` sells` ` price` ` markets` ` selling` ` purchaser` ` sale` ` Sale` ` market` ` “` | `at` ` parallel` ` paralle` `Fa` ` finite` `pt` `ot` `sp` ` only` `pha` |
| 1 | ` product` ` this` ` into` ` the` ` sell` ` price` ` purchaser` ` of` ` products` `
` | `at` ` due` `pha` ` on` ` off` `off` ` full` `ix` `ie` `out` |
| 2 | ` price` ` buyers` ` sale` ` Sale` ` sell` ` prices` `
` ` buyer` ` products` ` selling` | ` full` `pha` ` Full` ` process` ` Brid` ` out` ` due` `out` `o` `ations` |
| 3 | ` price` ` products` ` buyers` ` prices` ` markets` ` Price` ` sale` ` market` ` Sale` ` merchandise` | ` full` ` due` ` out` ` off` ` process` ` Full` ` point` ` Brid` `pha` `off` |
| 4 | ` buyers` ` products` ` price` ` Price` ` prices` ` Sale` ` which` ` the` ` and` ` ‘` | ` full` ` out` ` off` ` due` ` and` ` Brid` ` process` ` below` ` direct` `
` |
| 5 | ` buyers` ` products` ` Price` ` Sale` ` the` ` point` ` which` ` price` ` prices` ` points` | ` off` ` out` ` process` `st` `pha` ` Brid` `d` ` full` `Process` ` similar` |
| 6 | ` buyers` ` Sale` ` products` ` Price` ` buyer` ` markets` ` sale` ` sell` ` goods` ` price` | ` off` ` out` ` point` `off` `st` ` points` ` all` `i` ` process` ` i` |
| 7 | ` buyers` ` prices` ` Sale` ` Price` ` markets` ` price` ` products` ` goods` ` auction` ` lists` | ` off` ` point` `off` ` points` ` out` `out` `st` `
` ` similar` `stuff` |
| 8 | ` buyers` ` products` ` prices` `
` ` goods` ` auction` ` objects` ` Sale` ` items` ` all` | ` off` ` point` ` out` ` points` `off` `out` `st` `err` ` similar` ` at` |
| 9 | ` buyers` ` products` ` prices` ` markets` ` goods` ` commodities` ` sales` ` auction` ` Sale` ` market` | ` off` ` all` ` point` ` them` ` new` ` at` ` out` ` their` ` for` ` more` |
| 10 | ` products` ` buyers` ` prices` ` sales` ` sale` ` “` ` Sale` ` markets` ` sells` ` packages` | ` off` ` them` ` all` ` new` ` more` ` as` ` point` ` out` ` for` `-` |
| 11 | ` “` ` products` `
` ` buyers` ` goods` ` prices` ` price` ` markets` ` market` ` ‘` | ` off` ` them` ` all` ` at` ` to` ` as` ` out` ` more` ` point` ` or` |
| 12 | ` “` ` products` ` buyers` ` goods` ` new` ` off` ` points` ` buyer` ` price` ` other` | ` off` ` all` ` them` ` to` ` point` ` out` `-` ` their` ` at` ` more` |
| 13 | ` “` ` "` ` off` ` products` ` themselves` ` also` ` them` ` **` ` price` ` goods` | `-` ` off` ` at` ` them` ` the` ` all` `,` ` to` ` as` ` more` |
| 14 | ` “` ` prices` ` them` ` price` ` $` ` ‘` ` ` ` their` ` –` `‐` | ` them` ` the` ` at` ` more` ` all` `-` `
` ` "` ` as` ` a` |
| 15 | ` them` ` prices` ` price` ` these` ` ones` ` cheaper` ` products` ` their` ` those` ` faster` | ` them` ` at` ` the` `-` ` more` ` as` ` to` ` a` ` all` ` "` |
| 16 | ` them` ` prices` ` price` ` faster` ` their` ` cheaper` `them` ` these` ` they` ` everything` | ` them` ` at` `-` ` more` ` all` ` as` ` the` ` "` ` for` ` off` |
| 17 | ` prices` ` price` ` auction` ` cheaper` ` buyers` ` sellers` ` pricing` ` mortgages` ` property` ` mortgage` | ` them` ` at` `-` ` "` ` as` ` for` ` to` ` all` ` more` ` the` |
| 18 | ` prices` ` price` ` them` ` cheaper` ` auction` ` mortgage` ` property` ` mortgages` ` loans` ` homes` | ` them` ` at` ` for` ` "` `-` ` as` ` to` ` the` ` a` ` all` |
| 19 | ` *` ` \$` ` [*` ` **` ` *"` ` \[*` `\$` ` ([@` ` ` ` _` | ` at` ` "` ` for` `-` ` the` ` them` ` to` ` as` `,` ` a` |
| 20 | ` apartments` ` ([@` `
` ` properties` ` homes` ` property` ` prices` ` he` ` houses` ` house` | ` at` ` for` ` "` `-` ` the` `,` ` to` ` them` ` a` ` as` |
| 21 | ` them` ` property` ` cheap` ` Ferguson` ` properties` ` quickly` ` mortgages` ` apartments` ` foreclosure` ` price` | ` them` ` at` ` for` ` the` ` "` `-` ` to` `,` ` as` ` in` |
| 22 | ` the` ` them` ` apartments` ` at` ` some` ` cheap` ` their` ` fast` ` luxury` ` for` | ` them` ` for` ` at` ` the` ` "` ` fast` ` to` ` in` ` off` ` as` |
