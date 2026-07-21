# EXP-M5-0c swap-decomposition verdict — **H-CONFOUND**

Instrument diagnostic (D-029). Puts a verdict on record; changes NO Q2/Q6 bar.
Controls passed (positive + prob-space negative): True
- positive (410m sham-ctrl gap-shift median >= 0.3): True (value 9.586)
- negative GATING, prob-space dp(swap_answer) under sham in [-0.03, 0.03]: True ({'ref': 0.0088, 'test': 0.0004})
- negative informational, logit target-push: {'ref': 1.592, 'test': 1.05}; logit gap-shift: {'ref': 3.752, 'test': 1.015}
- FLAG: prereg negative control reads 'sham gap-shift median within [-0.03,0.03] logit units'. A norm-matched sham is an ACTIVE edit whose logit effects are O(1) (it ablates the source component, suppressing the answer), so no logit-space sham quantity meets a 0.03 band even on the certified 410m. The band matches the M-series Q3 sham, which is Δp(swap_answer) under the random direction (~0). Gating on the PROB-space target push; logit readings reported. FLAGGED clarification for Ecaterina (does not affect the sham-controlled decision statistic).

Matched item set (base-correct on both): N=15 — swap-China-to-Japan, swap-Cuba-to-Iran, swap-Egypt-to-Greece, swap-England-to-Spain, swap-France-to-Italy, swap-Germany-to-Russia, swap-Greece-to-Egypt, swap-Iran-to-Cuba, swap-Ireland-to-Poland, swap-Italy-to-France, swap-Japan-to-China, swap-Norway-to-Sweden, swap-Poland-to-Ireland, swap-Spain-to-England, swap-Sweden-to-Norway

| substrate | sham-ctrl gap-shift median [q1,q3] (per-draw) | neg dp(sham) | top1-flip med | margin-norm-flip med |
|---|---|---|---|---|
| ref pythia410m_m5_0c_swap | 9.586 [9.394,9.755] (9.586,9.925,9.202) | 0.0088 | 0.867 | 1.000 |
| test pythia1p4b_m5_0c_swap | 9.935 [9.361,10.103] (9.935,8.788,10.270) | 0.0004 | 0.600 | 1.000 |

Decision numbers: 410m_median=9.586, 1.4b_median=9.935, delta=-0.350 (bar 0.15), IQR non-overlap=False, 1.4b margin-flip=1.0, 1.4b raw-flip=0.6.

H-POTENCY: reduced gap-shift at 1.4B (HYPOTHESIS tier, NOT a gate failure). H-CONFOUND: comparable gap-shift, low raw flip but high margin-normalized flip. margin-flip low/high cutoff is a judgment (prereg leaves it qualitative); raw numbers reported.
