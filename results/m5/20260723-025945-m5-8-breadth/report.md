# EXP-M5-8 within-species breadth (410M, Mac arm: S1 + S5)

- model EleutherAI/pythia-410m@9879c9b; 3 extraction draws; 3 cached lens draws; A1 controls ok

- **S1: PROFILE-REPRODUCES (5/5)** (profile = draw-stable ∧ lens-dark ∧ inert; bar ≥4/5)
- **S5: HETEROGENEOUS (1/4)** (profile = output-aligned ∧ logit-trivial ∧ potent; bar ≥3/4, LOAD-BEARING)

| instance | draw-stab cos | jlens rank | logit rank | potent | profile |
|---|---|---|---|---|---|
| S1:Paris | 0.9737 | 219 | 939 | False | MATCH |
| S1:London | 0.9848 | 1184 | 70 | False | MATCH |
| S1:Rome | 0.9739 | 197 | 3162 | False | MATCH |
| S1:Berlin | 0.9759 | 2433 | 3581 | False | MATCH |
| S1:Madrid | 0.9738 | 145 | 2306 | False | MATCH |
| S5:sentiment | 0.9148 | 2 | 7 | True | MATCH |
| S5:formality | 0.9072 | 30 | 347 | True | off |
| S5:politeness | 0.9446 | 87 | 20 | True | off |
| S5:excitement | 0.9263 | 29 | 24 | True | off |

S2 (3 certified FVs) runs on the RTX (FV tensor cache) — folded in on that result.
wall 4221.6 s; peak 1.79 GB. raw under raw_completions/.