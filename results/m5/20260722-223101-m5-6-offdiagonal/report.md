# EXP-M5-6 off-diagonal test (410M) — S5 steering on the (decodability × potency) 2×2

- **verdict: A1-DECOMPOSES (S5 is logit-trivially decodable [A1a] but NOT J-lens-privileged [A1b]; A1 splits into reads-at-all vs J-lens-privileged via output-alignment; NOT off-diagonal, deflation NOT refuted)**
- S5 cell: **A1a=y A1b=n (jlens 2 / logit 2); potent=y**

- A1 (decode_vector, E1): jlens label-rank median 2 / logit 2 -> A1a reads-at-all=**True**, A1b J-lens-privileged=**False** (logit-trivial=True; controls ok)
- A2 (injection+ablation ΔS): injection ΔS +13.074 (transfer True) / ablation +0.250 -> **potent** (pos-ctrl +17.411, ok)

## A1 rank table — jlens / logit label-rank (identical decode_vector statistic)

| direction | jlens median | logit median | pattern | per-draw jlens/logit |
|---|---|---|---|---|
| S5-steering | 2 | 2 | jlens≈logit (logit-trivial) | d1:2/2 d2:2/2 d3:2/2 |
| S1-concept(Paris) | 192 | 999 | dark (neither) | d1:192/999 d2:86/183 d3:213/1037 |
| S2-FV(sing-plur, cited E1) | 436 | 3203 | dark (neither) | cited(E1) |

Contrast: S2 (FV) is dark to the J-lens (both high); S5 (steering) is jlens≈logit (logit-trivial, output-aligned) — the sub-axes coincide for S2 and separate for S5.
wall 58.4 s; peak 2.32 GB. raw under raw_completions/.