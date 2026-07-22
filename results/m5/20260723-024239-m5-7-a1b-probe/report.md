# EXP-M5-7 confirmatory probe (410M) — transient-intermediate J-lens advantage

- model EleutherAI/pythia-410m@9879c9b; cached lens draws 0/1/2; max-contrast cap 5.0; 3 draws median

- **capital-operand** (transient latent (held operand)): jlens HMR 2.54 vs logit HMR 140.64 → max-contrast ratio **55.370078740157474** (random 0.0)
- **capital-recall** (recalled answer token): jlens HMR 3.97 vs logit HMR 212.01 → max-contrast ratio **45.69181034482759** (random 0.0)

The held LATENT operand is J-lens-privileged (jlens HMR << logit) on 410M under the SAME lens draws as the static S1/S2/S5 rows — the transient 410M row is now in-run. OBJECT-TYPE CEILING (prereg): residual state vs pulled-out direction are different objects; this closes only the cross-experiment gap. Clean output null = M5-0b 1.4B fresh-answer ~0.9x.
wall 427.9s peak 2.79GB