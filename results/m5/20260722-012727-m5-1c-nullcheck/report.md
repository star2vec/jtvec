# EXP-M5-1c null-check report (410M)

**overall: PASS** (both instruments must report null on null)

## I1 — amended-Q5 max-contrast (scrambled-label latent probe)
- genuine ratio_jlens median 55.37 (sanity ≥ 5.0: ok)
- scrambled ratio_jlens median 0.0 (null < 5.0: PASS)
- **I1: PASS**

## I2 — D-033 extended-ladder concept readout (scrambled labels)
- max scrambled cosine over rungs 0.0895 (null < 0.95: PASS)
- worst |sham-ctrl Δp| 0.00056 over alphas {1.0: 2e-05, 2.0: 0.00035, 4.0: 0.00045, 8.0: 0.00056} (null ≤ 0.005: PASS)
- sanity: genuine cosine@256 0.9953 vs scrambled -0.0205 (discrimination: ok); unembed-injection Δp 0.80185 (ok)
- **I2: PASS**

Per-instrument withdrawal on failure (CONSTRAINTS LAW). wall 3478.3 s, peak 2.05 GB. raw under raw_completions/.