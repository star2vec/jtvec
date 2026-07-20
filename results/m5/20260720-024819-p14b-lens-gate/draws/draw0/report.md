# Phase 1 report: J-lens on GPT-2-small

- model: EleutherAI/pythia-1.4b (revision fedc38a16eea3bd36a96b906d78d11d2ce18ed79)
- device: mps, dtype: float32, seed: 0
- calibration: n=10 x 128 tokens from NeelNanda/pile-10k
- jlens commit: 581d398613e5602a5af361e1c34d3a92ea82ba8e
- lens variants: skip4 (skip_first=4, fitted 2026-07-20T04:58:30, 6877.2s)

## Milestone gate

Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at some layer in the L4-L16 band; (B) J-lens HMR beats the random-matrix control (mean over seeds) at every band layer (L4-L16; ruling 2026-07-14 — the earliest layers are excluded, matching the paper's own caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the random-direction control.

### skip4: **PASS**

- [x] capital-operand (A): J-lens beats logit at band layers [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16]
- [x] capital-operand (B): J-lens beats mean random control at every band layer
- [x] capital-recall (A): J-lens beats logit at band layers [5, 6, 7, 8, 11, 12, 13, 14, 15]
- [x] capital-recall (B): J-lens beats mean random control at every band layer
- [x] opposites (A): J-lens beats logit at band layers [4, 5, 6, 7, 12]
- [x] opposites (B): J-lens beats mean random control at every band layer
- [x] word-pairs (A): J-lens beats logit at band layers [7, 12, 13]
- [x] word-pairs (B): J-lens beats mean random control at every band layer
- [x] swap-capitals (C): dp +0.4948 vs random +0.0005

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
| 0 | 321.6 | 540.8 | 1716.0 | 0.00 | 0.00 | 0.01 |
| 1 | 204.2 | 215.8 | 1572.2 | 0.00 | 0.00 | 0.00 |
| 2 | 138.0 | 152.7 | 2876.4 | 0.03 | 0.00 | 0.00 |
| 3 | 188.5 | 177.3 | 2972.6 | 0.00 | 0.00 | 0.00 |
| 4 | 150.6 | 424.8 | 1324.1 | 0.03 | 0.00 | 0.00 |
| 5 | 172.1 | 940.2 | 2387.3 | 0.00 | 0.00 | 0.00 |
| 6 | 151.9 | 755.6 | 1970.6 | 0.00 | 0.00 | 0.00 |
| 7 | 361.8 | 2870.0 | 2131.1 | 0.00 | 0.00 | 0.00 |
| 8 | 390.2 | 1233.0 | 2452.6 | 0.00 | 0.00 | 0.00 |
| 9 | 600.1 | 663.7 | 1938.0 | 0.00 | 0.00 | 0.01 |
| 10 | 937.3 | 578.8 | 3043.6 | 0.00 | 0.00 | 0.00 |
| 11 | 16.5 | 66.0 | 3067.3 | 0.22 | 0.03 | 0.00 |
| 12 | 15.6 | 126.7 | 1263.9 | 0.11 | 0.03 | 0.01 |
| 13 | 1.5 | 19.7 | 2232.8 | 0.97 | 0.17 | 0.00 |
| 14 | 2.1 | 67.5 | 1993.5 | 0.86 | 0.00 | 0.00 |
| 15 | 1.7 | 1.3 | 2361.4 | 1.00 | 1.00 | 0.00 |
| 16 | 1.6 | 1.6 | 2000.2 | 1.00 | 1.00 | 0.00 |
| 17 | 1.8 | 1.7 | 1806.7 | 1.00 | 1.00 | 0.01 |
| 18 | 1.8 | 2.7 | 1990.7 | 1.00 | 0.92 | 0.01 |
| 19 | 9.2 | 4.6 | 2553.7 | 0.58 | 0.78 | 0.00 |
| 20 | 4.1 | 23.6 | 4310.6 | 0.97 | 0.14 | 0.01 |
| 21 | 4.0 | 37.1 | 2258.6 | 0.86 | 0.06 | 0.00 |
| 22 | 3.9 | 41.4 | 2144.7 | 0.78 | 0.06 | 0.01 |

min-over-layers: J-lens HMR 1.3 / pass@10 1.00; logit HMR 1.3 / pass@10 1.00

### skip4 / capital-recall

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 352.8 | 236.3 | 1636.0 | 0.00 | 0.03 | 0.00 |
| 1 | 202.3 | 162.1 | 925.1 | 0.00 | 0.03 | 0.01 |
| 2 | 183.6 | 130.9 | 1640.0 | 0.00 | 0.00 | 0.00 |
| 3 | 272.1 | 87.5 | 3315.0 | 0.00 | 0.03 | 0.00 |
| 4 | 272.3 | 237.4 | 2713.5 | 0.00 | 0.00 | 0.00 |
| 5 | 169.5 | 992.2 | 2050.1 | 0.00 | 0.00 | 0.00 |
| 6 | 229.2 | 368.5 | 1377.8 | 0.00 | 0.00 | 0.00 |
| 7 | 273.3 | 1746.2 | 1479.5 | 0.00 | 0.00 | 0.01 |
| 8 | 233.0 | 984.4 | 1832.0 | 0.00 | 0.00 | 0.00 |
| 9 | 415.1 | 347.1 | 3129.8 | 0.00 | 0.00 | 0.00 |
| 10 | 373.3 | 157.8 | 2288.8 | 0.00 | 0.00 | 0.00 |
| 11 | 29.0 | 34.1 | 2444.2 | 0.11 | 0.08 | 0.00 |
| 12 | 22.2 | 110.7 | 739.1 | 0.14 | 0.00 | 0.01 |
| 13 | 1.9 | 39.4 | 2347.2 | 0.83 | 0.08 | 0.00 |
| 14 | 1.7 | 44.3 | 2215.7 | 0.83 | 0.06 | 0.00 |
| 15 | 1.4 | 1.5 | 1818.7 | 0.92 | 0.97 | 0.01 |
| 16 | 1.4 | 1.4 | 1804.2 | 0.89 | 0.97 | 0.00 |
| 17 | 1.2 | 1.2 | 1704.3 | 0.92 | 0.97 | 0.01 |
| 18 | 1.2 | 1.1 | 2687.7 | 0.92 | 1.00 | 0.00 |
| 19 | 3.1 | 1.1 | 2085.0 | 0.86 | 1.00 | 0.00 |
| 20 | 2.0 | 1.3 | 2266.2 | 0.94 | 0.97 | 0.00 |
| 21 | 1.1 | 1.3 | 2945.9 | 0.97 | 0.97 | 0.00 |
| 22 | 1.1 | 1.5 | 1793.3 | 1.00 | 0.94 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / opposites

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 209.7 | 1999.8 | 3661.0 | 0.00 | 0.00 | 0.00 |
| 1 | 589.6 | 909.1 | 4382.2 | 0.00 | 0.00 | 0.00 |
| 2 | 294.0 | 574.9 | 4405.9 | 0.00 | 0.00 | 0.00 |
| 3 | 180.1 | 725.9 | 7653.7 | 0.00 | 0.00 | 0.00 |
| 4 | 301.2 | 654.1 | 5519.1 | 0.00 | 0.00 | 0.00 |
| 5 | 317.3 | 2177.0 | 4024.8 | 0.00 | 0.00 | 0.00 |
| 6 | 202.8 | 332.1 | 4874.3 | 0.00 | 0.00 | 0.00 |
| 7 | 176.5 | 551.5 | 4070.1 | 0.00 | 0.00 | 0.00 |
| 8 | 125.0 | 114.5 | 2367.4 | 0.00 | 0.06 | 0.00 |
| 9 | 93.0 | 85.1 | 4498.2 | 0.06 | 0.06 | 0.00 |
| 10 | 52.3 | 50.3 | 4101.0 | 0.06 | 0.06 | 0.00 |
| 11 | 4.0 | 2.9 | 3743.0 | 0.31 | 0.44 | 0.00 |
| 12 | 1.9 | 3.0 | 3447.1 | 0.62 | 0.38 | 0.00 |
| 13 | 1.0 | 1.0 | 4971.4 | 1.00 | 1.00 | 0.00 |
| 14 | 1.0 | 1.0 | 3523.4 | 1.00 | 1.00 | 0.00 |
| 15 | 1.0 | 1.0 | 4494.3 | 1.00 | 1.00 | 0.00 |
| 16 | 1.0 | 1.0 | 3822.6 | 1.00 | 1.00 | 0.00 |
| 17 | 1.0 | 1.0 | 4767.3 | 1.00 | 1.00 | 0.00 |
| 18 | 1.0 | 1.0 | 4665.5 | 1.00 | 1.00 | 0.00 |
| 19 | 3.1 | 1.0 | 4868.3 | 1.00 | 1.00 | 0.00 |
| 20 | 1.7 | 1.0 | 4720.0 | 1.00 | 1.00 | 0.00 |
| 21 | 1.0 | 1.0 | 2602.5 | 1.00 | 1.00 | 0.00 |
| 22 | 1.0 | 1.0 | 3637.8 | 1.00 | 1.00 | 0.00 |

min-over-layers: J-lens HMR 1.0 / pass@10 1.00; logit HMR 1.0 / pass@10 1.00

### skip4 / word-pairs

| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@10 | logit pass@10 | random pass@10 |
|---|---|---|---|---|---|---|
| 0 | 232.4 | 139.0 | 1923.1 | 0.00 | 0.04 | 0.02 |
| 1 | 43.1 | 23.5 | 2188.1 | 0.04 | 0.04 | 0.00 |
| 2 | 42.0 | 21.1 | 3351.4 | 0.04 | 0.08 | 0.00 |
| 3 | 21.4 | 22.9 | 3327.9 | 0.04 | 0.04 | 0.00 |
| 4 | 65.4 | 22.4 | 3075.8 | 0.04 | 0.04 | 0.00 |
| 5 | 29.2 | 22.8 | 2714.2 | 0.08 | 0.04 | 0.00 |
| 6 | 34.3 | 22.5 | 2034.3 | 0.04 | 0.04 | 0.00 |
| 7 | 17.9 | 22.4 | 2644.8 | 0.08 | 0.04 | 0.00 |
| 8 | 16.4 | 15.1 | 1849.7 | 0.08 | 0.08 | 0.00 |
| 9 | 36.4 | 17.5 | 3901.3 | 0.08 | 0.08 | 0.00 |
| 10 | 50.9 | 14.4 | 2452.9 | 0.08 | 0.08 | 0.00 |
| 11 | 6.6 | 4.5 | 2339.0 | 0.25 | 0.33 | 0.00 |
| 12 | 3.3 | 6.3 | 1987.5 | 0.42 | 0.29 | 0.00 |
| 13 | 2.5 | 3.0 | 2385.8 | 0.58 | 0.50 | 0.00 |
| 14 | 2.0 | 1.8 | 4208.1 | 0.79 | 0.83 | 0.00 |
| 15 | 1.8 | 1.6 | 2123.7 | 0.75 | 0.88 | 0.00 |
| 16 | 1.6 | 1.5 | 1935.3 | 0.75 | 0.88 | 0.00 |
| 17 | 1.4 | 1.5 | 2237.0 | 0.88 | 0.88 | 0.00 |
| 18 | 1.4 | 1.6 | 2215.5 | 0.88 | 0.88 | 0.00 |
| 19 | 6.0 | 1.7 | 2369.4 | 0.62 | 0.75 | 0.00 |
| 20 | 2.4 | 1.7 | 4596.1 | 0.92 | 0.71 | 0.00 |
| 21 | 1.3 | 1.7 | 2914.9 | 0.96 | 0.79 | 0.00 |
| 22 | 1.2 | 1.5 | 2567.1 | 0.96 | 0.92 | 0.00 |

min-over-layers: J-lens HMR 1.1 / pass@10 0.96; logit HMR 1.3 / pass@10 0.96

## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)

| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |
|---|---|---|---|---|---|---|
| skip4 | swap-capitals | +0.4948 | +0.0005 | -0.5530 | 56.2% | 16 |

## Held-out prompt readouts

### Held-out prompt 1

(128-token window; last 12 tokens: `... weight. Addition of SO2 to the organic phase before the`)

**skip4** (model's actual next token: `' addition'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `s` ` the` ` a` ` different` ` an` ` A` ` not` ` "` ` D` ` new` | `medsc` ` programmers` `iet` `ophyll` `rette` ` thous` ` IB` `thur` `ingo` ` GENERATED` |
| 1 | ` the` ` a` ` (` `s` ` different` ` in` ` where` ` A` ` of` ` which` | `ophyll` `medsc` `etz` `rette` ` congression` `-------------------------------------------------` `ogs` ` outset` ` forthcoming` ` vain` |
| 2 | ` which` ` the` `s` ` where` ` is` ` in` ` a` ` end` ` of` ` different` | `ophyll` `medsc` `etz` ` congression` `osal` ` outset` `--------------------------------------------------` ` beginning` `-------------------------------------------------` `===========================` |
| 3 | ` the` ` end` `s` ` a` ` each` ` not` ` different` ` an` ` all` ` start` | `ophyll` `medsc` ` outset` ` forthcoming` ` congression` `emer` `yla` ` beginning` `iol` `rette` |
| 4 | ` the` ` in` ` start` `s` ` first` ` set` ` being` ` rather` ` on` ` which` | `ophyll` ` outset` `etz` ` congression` `medsc` ` thous` `dy` `emer` ` beginning` `rette` |
| 5 | `s` ` in` ` different` ` rather` ` start` ` starting` ` which` ` beginning` ` all` ` that` | `ophyll` ` outset` `medsc` ` beginning` `rette` `emer` ` mentioned` ` congression` ` above` ` thous` |
| 6 | `s` ` is` ` which` ` in` ` rather` ` start` ` ‘` ` development` ` R` ` set` | `ophyll` ` outset` `emer` ` Debor` ` beginning` ` mentioned` ` thous` `===========================` ` congression` `bish` |
| 7 | ` is` ` ‘` ` R` ` which` ` any` `s` ` in` ` start` ` A` `
` | ` outset` `ophyll` ` beginning` `tha` `emer` `thur` ` IB` ` PARTICULAR` ` mentioned` `bish` |
| 8 | ` _` ` “` ` in` ` is` ` R` ` [*` ` set` ` ‘` ` the` `
` | ` IB` ` PARTICULAR` `thur` ` outset` ` GENERATED` `
	 ` `tha` ` beginning` `ophyll` ` start` |
| 9 | ` beginning` ` the` ` distribution` ` start` ` present` ` is` ` “` ` first` ` starting` ` (` | ` beginning` ` outset` ` start` ` IB` `ophyll` ` inevitable` `
	 ` ` latter` ` envis` ` above` |
| 10 | ` beginning` ` “` ` start` ` starting` ` first` ` "` ` (` ` ‘` ` the` ` *` | ` beginning` ` start` ` first` ` latter` ` starting` ` outset` ` final` ` above` ` introduction` `ran` |
| 11 | ` start` ` first` ` “` ` beginning` ` introduction` ` addition` ` merger` ` [*` ` initial` ` formation` | ` beginning` ` start` ` formation` ` final` ` first` ` initial` ` introduction` ` starting` ` latter` ` process` |
| 12 | ` start` ` formation` ` addition` ` beginning` ` starting` ` initiation` ` introduction` ` same` ` application` ` creation` | ` formation` ` final` ` start` ` beginning` ` initial` ` starting` ` introduction` ` first` ` initiation` ` latter` |
| 13 | ` addition` ` start` ` merger` ` introduction` ` mixing` ` formation` ` final` ` generation` ` beginning` ` arrival` | ` final` ` addition` ` initial` ` first` ` formation` ` start` ` latter` ` beginning` ` introduction` ` starting` |
| 14 | ` addition` ` "` ` merger` ` aqueous` ` introduction` ` \"` ` mixing` ` incorporation` ` injection` ` initiation` | ` addition` ` final` ` latter` ` initial` ` start` ` same` ` beginning` ` first` ` formation` ` starting` |
| 15 | ` addition` ` formation` ` aqueous` ` creation` ` introduction` ` start` ` initiation` ` application` ` final` ` merger` | ` addition` ` final` ` latter` ` start` ` initial` ` first` ` desired` ` formation` ` introduction` ` starting` |
| 16 | ` addition` ` formation` ` introduction` ` aqueous` ` mixing` ` polymerization` ` initiation` ` incorporation` ` polymer` ` injection` | ` addition` ` final` ` formation` ` latter` ` first` ` initial` ` introduction` ` mixture` ` desired` ` mixing` |
| 17 | ` polymerization` ` aqueous` ` dissolution` ` mixing` ` formation` ` addition` ` introduction` ` injection` ` formulation` ` monomer` | ` addition` ` first` ` final` ` initial` ` introduction` ` formation` ` latter` ` start` ` mixture` ` starting` |
| 18 | ` polymerization` ` mixing` ` mixture` ` aqueous` ` introduction` ` addition` ` polymer` ` dissolution` ` formulation` ` phase` | ` addition` ` final` ` first` ` mixture` ` mixing` ` initial` ` start` ` introduction` ` formation` ` mixed` |
| 19 | ` *` ` [*` ` **` ` _` ` aqueous` ` addition` ` introduction` ` preparation` ` mixing` ` \[*` | ` addition` ` first` ` final` ` initial` ` preparation` ` mixing` ` formation` ` mixture` ` start` ` introduction` |
| 20 | ` addition` `
` ` introduction` ` formation` ` preparation` ` aqueous` ` mixing` ` initiation` ` dissolution` ` conversion` | ` addition` ` first` ` final` ` formation` ` introduction` ` start` ` mixing` ` initial` ` mixed` ` preparation` |
| 21 | ` aqueous` ` addition` ` mixing` ` organic` ` introduction` ` mixture` ` surfactant` ` injection` ` formation` ` precipitation` | ` addition` ` first` ` final` ` inter` ` formation` ` start` ` initial` ` introduction` ` mixed` ` mixing` |
| 22 | ` addition` ` aqueous` ` polymer` ` mixing` ` organic` ` polymerization` ` preparation` ` formation` ` precipitation` ` monomer` | ` addition` ` first` ` final` ` initial` ` inter` ` formation` ` start` ` mixed` ` introduction` ` mixing` |

### Held-out prompt 2

(128-token window; last 12 tokens: `... and many other legends helped bring the club countless trophies from`)

**skip4** (model's actual next token: `' the'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | ` the` ` D` ` (` ` a` ` A` ` F` `s` ` an` ` in` ` B` | `ula` ` within` ` indeed` ` being` ` kinemat` ` Gmb` `ge` ` behind` ` Biosc` ` vain` |
| 1 | ` the` ` “` ` in` ` within` ` where` `s` ` (` ` and` ` a` ` ‘` | `oddsidemargin` ` within` `gery` `kel` `ula` ` behind` ` SOFTWARE` ` kinemat` ` Within` ` indeed` |
| 2 | ` where` `s` ` the` ` within` ` to` ` in` ` each` ` a` ` of` ` different` | ` within` ` behind` `kel` `bi` ` indeed` `oddsidemargin` ` Within` ` Wy` ` Cl` ` Lit` |
| 3 | ` where` ` within` ` being` ` the` ` each` ` a` ` that` ` all` ` in` ` time` | ` within` ` behind` ` Within` `kel` ` across` `within` `ast` ` abroad` ` above` ` Progress` |
| 4 | ` within` ` where` ` across` ` the` `
` ` in` ` being` ` a` ` to` ` inside` | ` within` ` across` ` behind` ` Within` `kel` ` being` `within` ` inside` ` abroad` ` below` |
| 5 | ` within` ` to` ` in` ` where` ` inside` ` which` `.` ` across` ` the` ` also` | ` within` ` across` ` start` ` being` ` abroad` ` beginning` ` {` ` Within` `ilon` `de` |
| 6 | ` within` ` which` ` across` ` the` ` to` ` in` ` inside` ` being` ` time` ` where` | ` within` ` across` ` being` ` start` ` abroad` ` throughout` ` Within` ` behind` ` {` ` inside` |
| 7 | ` within` ` which` ` across` ` the` ` in` `
` ` being` ` where` ` to` ` inside` | ` within` ` across` ` start` ` diverse` ` being` ` various` ` different` `kel` ` Within` `e` |
| 8 | ` which` ` where` ` within` ` across` ` the` ` being` ` in` ` to` ` all` ` inside` | ` across` ` within` ` start` ` being` ` beginning` ` various` ` which` ` early` ` different` ` leading` |
| 9 | ` which` ` within` ` where` ` the` ` of` ` across` ` to` ` in` ` and` ` all` | ` start` ` across` ` various` ` within` ` high` ` the` ` which` ` early` ` diverse` `,` |
| 10 | ` which` ` within` ` that` ` where` ` across` ` the` ` also` `
` ` being` ` throughout` | ` which` `,` ` the` ` across` ` start` ` within` ` being` ` and` ` high` ` time` |
| 11 | ` which` ` within` ` the` ` across` ` “` `.` ` both` ` all` ` ‘` ` also` | ` start` ` across` ` within` ` the` ` being` ` beginning` ` world` ` various` ` which` `.` |
| 12 | ` within` ` across` ` which` ` the` ` “` ` ‘` ` inside` `.` ` to` ` various` | ` the` ` across` `.` ` being` ` a` ` various` `,` ` which` ` start` ` one` |
| 13 | ` across` ` which` ` the` ` within` ` various` ` “` ` around` ` ‘` ` different` ` being` | ` the` ` various` ` a` ` which` ` and` `,` ` across` `.` ` being` ` both` |
| 14 | ` their` `."` ` \"` ` “` ` 1996` ` 1998` ` which` ` various` ` across` ` 1997` | ` the` ` various` ` their` ` different` ` a` ` its` ` both` ` which` ` one` ` across` |
| 15 | ` games` ` their` `."` ` competitions` ` various` ` players` ` numerous` ` football` ` which` ` 1998` | ` the` ` various` ` a` ` both` ` all` ` their` ` one` ` across` ` and` ` different` |
| 16 | ` games` ` players` ` FIFA` ` football` ` player` ` competitions` ` soccer` ` game` ` tournaments` ` their` | ` the` ` various` ` all` ` across` ` one` ` both` ` different` ` their` ` a` ` top` |
| 17 | ` FIFA` ` players` ` games` ` tournaments` ` football` ` competitions` ` soccer` ` player` ` leagues` ` Football` | ` the` ` all` ` both` ` a` ` various` ` one` ` top` ` across` ` different` ` and` |
| 18 | ` players` ` soccer` ` FIFA` ` tournaments` ` games` ` football` ` clubs` ` Football` ` competitions` ` leagues` | ` the` ` all` ` a` ` various` ` both` ` different` ` one` `-` ` and` ` top` |
| 19 | ` *` ` _` ` **` ` [*` ` tournaments` ` competitions` ` \"` ` *"` ` countries` ` clubs` | ` the` ` all` ` a` ` their` ` various` ` its` ` both` ` one` ` different` ` and` |
| 20 | `
` ` Europe` ` the` ` their` ` European` ` England` ` UE` ` World` ` world` ` winning` | ` the` ` all` ` a` ` both` ` various` ` their` ` one` ` different` ` and` ` its` |
| 21 | ` the` ` Europe` ` European` ` UE` ` Liverpool` ` Spain` ` England` ` World` ` Champions` ` Manchester` | ` the` ` all` ` a` ` both` ` various` ` one` ` different` ` that` ` their` `-` |
| 22 | ` the` ` their` ` Europe` ` continental` ` Manchester` ` Spain` ` Liverpool` ` European` ` Barcelona` ` various` | ` the` ` all` ` their` ` that` ` around` ` various` ` different` ` a` ` both` ` across` |

### Held-out prompt 3

(128-token window; last 12 tokens: `...192)) + 3*sqrt(192)*-2)**2`)

**skip4** (model's actual next token: `'.'`)

| layer | J-lens top-10 | logit-lens top-10 |
|---|---|---|
| 0 | `nd` `s` `B` ` B` ` 2` `gether` ` new` `d` `x` ` changes` | `nd` ` IB` `Isa` `odies` `dy` `jax` ` discussed` `all` ` of` `HCl` |
| 1 | `nd` `s` `B` `ded` `bor` `a` ` 2` ` breaks` ` B` `b` | `nd` `=======` ` galax` `jax` `oire` `==================================` `medsc` `------------------------------------` `ock` `tal` |
| 2 | `nd` ` 2` `s` ` "` ` ``` `ded` `
` ` which` ` where` `http` | `Simplify` `oire` ` +` `jax` ` Legal` `=======` ` past` ` reached` `RE` `性` |
| 3 | `nd` ` return` `s` `ded` ` which` `(` ` 2` ` returns` ` comeback` ` returned` | ` +` ` Legal` ` past` ` reached` `*-` `oire` `LEV` ` currently` `*` `olution` |
| 4 | `(` `
` `nd` `+` `-` `~` ` which` ` (` ` 2` `c` | `*-` ` +` ` reticulum` `pril` ` sop` `rox` `blic` `性` ` depl` ` Legal` |
| 5 | `(` `
` `*` ` (` `.` `-` `\` `nd` `,` `:` | `*-` ` +` `*` `性` ` 
` `
` ` performance` ` Border` ` sop` `<\|endoftext\|>` |
| 6 | `
` `*` `http` `(` ` (` ` +` `*(` `.` ` ` ` who` | `*-` ` +` `*` ` performance` ` prem` `*(-` ` Performance` `<\|endoftext\|>` ` private` ` performers` |
| 7 | `
` `*` ` *` `...` `*(` ` ``` `.` `http` `........` `<\|endoftext\|>` | `*-` ` +` `*` ` performance` ` performed` `)*` `*(-` ` performers` `)*-` ` past` |
| 8 | `.` `
` `<\|endoftext\|>` `*` `http` ` *` `...` `.\[[@` `........` ` in` | `*-` ` +` `*` `*.` ` private` `)*` `private` `red` `c` `.` |
| 9 | `.` `*` ` *` `
` ` \[[@` ` of` ` who` `<\|endoftext\|>` `\[[@` ` in` | `*` `*-` ` +` ` past` `.` `red` ` private` `private` ` pre` `c` |
| 10 | ` *` `
` `.` ` +` `*` ` "` ` of` ` who` ` -` `*-` | `*-` ` +` `*` `red` ` ir` `c` `id` ` past` `.` `ev` |
| 11 | `.` ` *` ` +` `*.` `*` `.*` `
` `*-` ` "` `<\|endoftext\|>` | `.` `*-` ` +` `*` `*.` ` f` `<\|endoftext\|>` `c` ` ` ` pre` |
| 12 | `.` ` +` `*.` `*` `*-` `
` ` "` ` *` `.*` `<\|endoftext\|>` | `.` ` +` `*` `*-` `<\|endoftext\|>` `
` ` -` ` in` `c` ` ` |
| 13 | `.` ` +` `*-` ` "` `).` `*.` `
` ` in` `+.` ` +\` | `.` ` +` `*` `
` `,` ` ` ` and` `<\|endoftext\|>` ` in` ` -` |
| 14 | `.` ` +` `*.` `."` `).` `.$$` `".` `.\"` ``.` `.\` | `.` `
` ` +` `*` `<\|endoftext\|>` ` ` `,` ` and` ` -` ` a` |
| 15 | `.` `*.` `."` `).` `.*` `".` `。` ` +` `*-` `.$` | `.` ` +` `*` `
` `<\|endoftext\|>` ` and` `,` ` -` ` to` ` a` |
| 16 | ` +` `.` `*.` `*-` `".` `.*` `).` `+.` `。` `."` | `.` ` +` `*` `*-` ` -` `<\|endoftext\|>` ` *` `*.` `).` `,` |
| 17 | `*.` ` +` `.` `*-` `.*` `*` `".` `_.` `."` `._` | `.` ` +` `*` ` -` `*-` `<\|endoftext\|>` `
` ` *` `,` `).` |
| 18 | `*-` `*.` `*` `.` ` +` `.*` ` *` `)*` `<\|endoftext\|>` `)*-` | `.` ` +` `*` `*-` `<\|endoftext\|>` `
` ` *` ` -` `,` `).` |
| 19 | ` *` ` *"` `*` ` [*` `*-` `*.` ` **` ` \[*` `.*` ` *[` | `.` ` +` `*` `*-` `<\|endoftext\|>` `
` ` *` ` -` `,` ` ` |
| 20 | `
` `<\|endoftext\|>` `.` `


` ` +` `*-` `*.` ` *` `.*` `*` | `.` ` +` `*` `*-` `<\|endoftext\|>` ` *` `
` ` -` `,` ` ` |
| 21 | `.` ` +` `*-` `/*!` `Simplify` `))*-` `<\|endoftext\|>` `` `JM` `)*-` | `.` ` +` `*` `*-` `<\|endoftext\|>` ` *` `,` ` -` `
` ` a` |
| 22 | `.` ` +` `*-` `Simplify` `)*-` `*((` `))*-` `*` `/*!` ` assuming` | `.` ` +` `*` `*-` `,` ` -` `<\|endoftext\|>` ` *` `
` ` a` |
