# CLAIMS.md — claims ledger

Machine-validated by `jtvec/validators/claims.py` on every CI run. Rules
(from the CONSTRAINTS.md LAWs):

- Every claim has a status in {hypothesis, preliminary, verified, withdrawn}
  and an evidence commit.
- The paper (DRAFT.md) may only cite claims at `verified`.
- Promotion to `verified` is blocked unless all of the following hold:
  - `evidence-commit` is a real commit hash;
  - `results-dir` passes the results-directory check (config copy, run
    record, raw completions on disk);
  - each DECLARED headline cell (the `headline-cells` field, citing the prereg
    decision rule that defines them) holds >= 20 records — the CONSTRAINTS LAW
    is "per headline cell", not per every cell (D-038). Draw-based diagnostic
    claims declare `headline-cells: draw-based` and are EXEMPT from the
    completions gate: they promote under the stochastic-estimator LAWs (>= 3
    draws, median/IQR, prereg'd sham/control arms, per-draw raw on disk),
    enforced at run time and attested by the verify line;
  - LABNOTES.md contains Ecaterina's verification line for this claim, of the
    exact form:
    `verify: CLM-NNN raw-read: <n> re-derived: yes verified-by: Ecaterina date: YYYY-MM-DD`

## Entry schema (CLM-000 is the template; validators ignore it)

### CLM-000
- status: hypothesis
- statement: <one observation with scope, e.g. "on Pythia-410M, skip4_n10, N=16, dp(swap) median=…, IQR=… (sham: …)">
- scope: <model@revision, config, N>
- evidence-commit: none
- prereg: none
- results-dir: none
- raw-completions: none
- headline-cells: <cell stems backing the headline numbers>; <prereg decision rule that defines them>   (or "draw-based; <rule>" for stochastic-estimator claims)
- verified-by: none

## Claims

### CLM-001
- status: hypothesis
- statement: On the three M2-certified tasks, Todd FVs (fv_todd@task, converged_at=25, n_draws=3) carry a task-label component decodable through the M1-gated J-lens arm and separated from the logit-lens arm, per criteria C1-C4 of EXP-M4-E1. Tests the CONSTRAINTS FV-label HYPOTHESIS. E1 counter-evidence (2026-07-19, run below): NOT-DECODABLE on 3/3 tasks — jlens label-rank medians 278/436/56 vs the C1 bar 20; english-french alone passed the random anchor (96/97/95) and failed the logit floor (logit median 114 < 200); jlens < logit ordering held in 33/33 cells on every task. Status stays hypothesis per the preregistered rule.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e1_pythia410m.yaml, 3 tasks x 3 FV draws x 3 lens draws
- evidence-commit: none
- prereg: harness/preregs/EXP-M4-E1-decodability.md
- results-dir: results/m4/20260719-021823-e1-decodability
- raw-completions: results/m4/20260719-021823-e1-decodability/raw_completions
- headline-cells: decode_capitalize, decode_singular-plural, decode_english-french; EXP-M4-E1 C1 label-rank decision rule
- verified-by: none

### CLM-002
- status: hypothesis
- statement: On singular-plural, fv-direction ablation and jspace ablation dissociate task execution from task verbalization per EXP-M4-E2-dissociation (fv cuts execution not report; jspace cuts report not execution), each vs its matched sham, cross-draw over the 3 M2-certified FV draws and 3 M1 lens draws. Tests the CONSTRAINTS execution-vs-verbalization HYPOTHESIS. E2 result (2026-07-19, run below): verdict ONE-WAY, NOT the double dissociation. Direction 1 holds and transfers (fv-ablation execution effect +0.920 vs sham +0.020 all 3 draws; report NOT hurt — it rose, effect -0.638 vs sham +0.037). Direction 2 fails: jspace hurts execution (effect +0.440 vs sham +0.000) and its report effect (+0.341) does not beat its sham (+0.295). Status stays hypothesis per the preregistered rule (only DOUBLE-DISSOCIATION promotes).
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e2_dissociation_pythia410m.yaml, singular-plural, 3 FV draws x 3 lens draws, N_exec=50 / N_report=80
- evidence-commit: none
- prereg: harness/preregs/EXP-M4-E2-dissociation.md
- results-dir: results/m4/20260719-142007-e2-dissociation
- raw-completions: results/m4/20260719-142007-e2-dissociation/raw_completions
- headline-cells: exec_fv_draw1, exec_fv_draw2, exec_fv_draw3, report_fv_draw1, report_fv_draw2, report_fv_draw3; EXP-M4-E2 D-017 dissociation decision rule
- verified-by: none

### CLM-003
- status: verified
- statement: On Pythia-410M, projecting out the M2-certified singular-plural function vector at the final position of band layers 4-16 removes task execution (accuracy 0.920 -> 0.000) while NOT reducing the P3 report readout (report_score rose, effect -0.638 vs sham +0.037), robust and near-identical across all 3 certified FV draws (execution effect +0.920, IQR 0, vs sham +0.020). One direction of the execution-vs-verbalization dissociation (Direction 1 of EXP-M4-E2); the reverse (jspace report-specific) was not established.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e2_dissociation_pythia410m.yaml, singular-plural, 3 FV draws, N_exec=50 / N_report=80
- evidence-commit: a1c7cb1
- prereg: harness/preregs/EXP-M4-E2-dissociation.md
- results-dir: results/m4/20260719-142007-e2-dissociation
- raw-completions: results/m4/20260719-142007-e2-dissociation/raw_completions
- headline-cells: exec_none, exec_fv_draw1, exec_fv_draw2, exec_fv_draw3, exec_sham_fv_draw1, exec_sham_fv_draw2, exec_sham_fv_draw3, report_none, report_fv_draw1, report_fv_draw2, report_fv_draw3, report_sham_fv_draw1, report_sham_fv_draw2, report_sham_fv_draw3; EXP-M4-E2 decision rule (D-017): fv-ablation execution + report effects vs sham
- verified-by: Ecaterina

### CLM-004
- status: verified
- statement: On capitalize (task A) prompts, swapping the certified FV_A component onto FV_B (singular-plural) at band layers 4-16 redirects the model to produce task-B (plural) outputs above a random-target control, per EXP-M4-E3-swap, cross-draw over the 3 certified FV draws. E3 result (2026-07-19): REDIRECTS-BASIS-AGNOSTIC — task-B rate 0.000 -> lens_swap 0.933 / direct_swap 0.800 (random 0.000), task A suppressed to 0.000, transfers across all 3 FV draws; lens-direct gap 0.133 < 0.15 so the identity is carried by the raw residual direction, not specifically the J-lens basis. Promotion to verified needs Ecaterina's raw-read verify line.
- scope: EleutherAI/pythia-410m@9879c9b, configs/m4_e3_swap_pythia410m.yaml, capitalize->singular-plural, 3 FV draws, N=30 shared queries
- evidence-commit: 0d8b278
- prereg: harness/preregs/EXP-M4-E3-swap.md
- results-dir: results/m4/20260719-151956-e3-swap
- raw-completions: results/m4/20260719-151956-e3-swap/raw_completions
- headline-cells: swap_none, swap_lens_draw1, swap_lens_draw2, swap_lens_draw3, swap_direct_draw1, swap_direct_draw2, swap_direct_draw3, swap_random_draw1, swap_random_draw2, swap_random_draw3; EXP-M4-E3 SwapRedirectionRule (D-018): task-B rate by condition vs random
- verified-by: Ecaterina

### CLM-005
- status: hypothesis
- statement: On Pythia-410M, S1 concept mean-difference directions are DRAW-STABLE / convergent (the 1b sense) yet injection-inert. IMPORTANT SCOPE (Ecaterina 2026-07-22, ruling (a) after EXP-M5-6): "decodable" here means DRAW-STABILITY, NOT lens-readout — S1's concept direction is DARK to decode_vector (jlens label-rank median 192, comparable to S2's 436); it was never lens-readout-decodable. EXP-M5-1b: all 8 roster capitals cross the 0.95 cross-draw cosine bar at the extended ladder to 256 (4/8 with a within-ceiling witness at 128; 0/8 plateau below) — draw-stable/convergent; but 0/8 clear the +0.10 sham-controlled Δp potency bar even at 8x injection — injection-inert. The EXP-M5-1c null-check confirms the injection mechanism is live (unembed('Paris') residual-add Δp +0.80; scrambled-label null ~0), so the injection-null is a real property, not a dead knob. EXP-M5-1d (band-layer project-out, sham-controlled Δlogit) corroborates non-potency where measurable: 0/8 potent (1 inert, 3 weak, 4 inconclusive). The (decodability × potency) 2×2 as the H1 anchor is RETIRED (EXP-M5-6): it compared S1 (convergence sense) and S2 (decode_vector sense) on DIFFERENT senses of "decodable"; under the same instrument (decode_vector) both S1 and S2 are lens-dark. The instrument-consistent dissociation is now POTENCY ALONE (EXP-M5-8 + D-035): under the IDENTICAL min_pairwise_cosine, S1 (0.974-0.985, EXP-M5-8 5/5) AND S2 (0.983-0.997, EXP-M5-8 S2 / M2 same-pipeline stability.json 0.991-0.997) are BOTH draw-STABLE and both lens-dark — draw-stability does NOT separate them; the sole S1/S2 separator is POTENCY: S1 = (draw-stable, lens-dark, INERT; 5/5 type-general), S2 = (draw-stable, lens-dark, POTENT [CLM-003/004 VERIFIED]). The earlier "S2 draw-UNSTABLE [cosine 0.43-0.61]" was the v1 cross-code-path degenerate-endpoint number (D-035: one collapsed re-extraction, +38.8% -> +1.8%), NOT the M2 same-pipeline certified tensors; v1's VERIFIED entry stands scoped to the sweep endpoint and is not demoted, with an honest unexplained residue (why that one draw collapsed was never bounded). Status stays hypothesis (S1 legs diagnostic-tier; S2 potency verified; the S2 draw-stability correction rests on D-035) pending Ecaterina's verify line.
- scope: EleutherAI/pythia-410m@9879c9b, EXP-M5-1b/1c/1d + EXP-M5-8 (S1 5-concept breadth) + D-035 (S2 diff), 8-capital roster / 5-concept breadth, 3 draws
- evidence-commit: none
- prereg: harness/preregs/EXP-M5-1b-concept-diagnostic.md
- results-dir: results/m5/20260722-023137-m5-1b-concept-diagnostic
- raw-completions: results/m5/20260722-023137-m5-1b-concept-diagnostic/raw_completions
- headline-cells: draw-based; EXP-M5-1b decision rule: convergence (min_pairwise_cosine, 3 draws) + potency alpha-sweep sham-controlled Δp (3 draws, median/IQR)
- verified-by: none

### CLM-006
- status: hypothesis
- statement: FRAMING / SPINE CLAIM, stated as observed DISCORDANCES so it is falsifiable (reframed 2026-07-23, Ecaterina; synthesis over CLM-005/008/009). On Pythia-410M, four instruments that a residency taxonomy would all read as "A1 decodability" returned DISCORDANT verdicts on the same vectors — so "decodability" is not one measurement. Concretely observed: (1) S1 concept directions and S2 function vectors AGREE on all three of draw-stability (both stable, min-pairwise-cosine ≥ 0.97), lens-readout (both dark: jlens label-rank 192 / 436), and logit-privilege (neither privileged) — NONE of the three separates S1 from S2; the pair is separated ONLY by POTENCY (S1 inert, S2 potent; CLM-005). (2) The S5 formality vector is DARK under the label-rank readout (jlens 30 / logit 347) yet ALIGNED under the top-token readout (its top unembed tokens are formal register) — two "decodability" readouts return OPPOSITE verdicts on one vector (CLM-009). FALSIFICATION: this claim fails if the instruments are CONCORDANT on these vectors — if any of the three decodability senses had separated S1 from S2 the way potency does, or if label-rank and top-token readout had agreed on the formality vector. On the measured vectors they do not. Consequence: the surviving type-separator is POTENCY, and the J-lens privilege that exists is relocated to transient state (CLM-008). The instrument-INCONSISTENT (decodability × potency) 2×2 that opened the taxonomy stays retired; per Ecaterina no further instrument is hunted (harvesting; budget 0).
- scope: EleutherAI/pythia-410m@9879c9b, synthesis over EXP-M5-6 + E1 (CLM-001) + M2 + EXP-M5-7 + EXP-M5-8/8b; sub-claims CLM-005/008/009
- evidence-commit: none
- prereg: harness/preregs/EXP-M5-6-offdiagonal.md
- results-dir: results/m5/20260722-223101-m5-6-offdiagonal
- raw-completions: results/m5/20260722-223101-m5-6-offdiagonal/raw_completions
- headline-cells: A1_rank_table; EXP-M5-6 A1 decode_vector rank-table decision rule (the discordance headline)
- verified-by: none

### CLM-007
- status: hypothesis
- statement: On Pythia-410M, EXP-M5-8 within-species breadth (n>1 per species, one identical instrument set) finds all three species profiles reproduce. S1 concept directions 5/5 (Paris/London/Rome/Berlin/Madrid: draw-stable min-cos 0.974-0.985 at n=256, lens-dark jlens 145-2433, injection+ablation inert) -> type-general, not an n=1 artifact. S2 function vectors 3/3 (capitalize/singular-plural/english-french: draw-stable 0.983-0.997, lens-dark 56-436, potent) -> homogeneous on a draw-stable + lens-dark + potent profile (the ratified "draw-unstable" premise did not reproduce; D-035). S5 steering vectors 4/4 output-aligned (sentiment/formality/politeness/excitement each point at their attribute's output field per the independent EXP-M5-8b top-token read; the label-rank 1/4 was a readout-vocabulary false negative, D-037) and 4/4 potent (inj ΔS +3.3..+12.1, transfer, pos-ctrl ok). The surviving cross-species dissociation is POTENCY (S1 inert vs S2/S5 potent); S1 and S2 are both draw-stable + lens-dark. Status hypothesis pending Ecaterina's verify line.
- scope: EleutherAI/pythia-410m@9879c9b, EXP-M5-8 (Mac S1/S5 + RTX S2) + EXP-M5-8b, 3 draws, 3 lens draws
- evidence-commit: none
- prereg: harness/preregs/EXP-M5-8-within-species-breadth.md
- results-dir: results/m5/20260723-025945-m5-8-breadth
- raw-completions: results/m5/20260723-025945-m5-8-breadth/raw_completions
- headline-cells: draw-based; EXP-M5-8 decision rule: per-instance draw-stability (min_pairwise_cosine, 3 draws) + potency (3 draws, sham) + lens-readout (3 lens draws); reproduction counts
- verified-by: none

### CLM-008
- status: verified
- statement: On Pythia-410M, the J-lens's logit-privilege (A1b: reading what the logit lens cannot) is NOT a property of extracted STATIC direction vectors but of TRANSIENT computational intermediates (split from CLM-006, 2026-07-23). Static directions are not privileged — under decode_vector: S1 concept jlens/logit label-rank 192/999, S2 FV 436/3203, S5 steering logit-trivial (jlens ≈ logit); none clear the A1b criterion. Transient held states ARE privileged: EXP-M5-7 (410M, cached lens draws 0/1/2) held-operand jlens HMR 2.54 vs logit 140.6 (~55x, random arm 0.0) and recalled-answer 3.97 vs 212 (~46x), reproducing the M1 gate's VERIFIED 2.5/61.5; EXP-M5-0b (1.4B) operand 6.98x / bridge 15.5x with the matched OUTPUT token ~0.9x (no advantage). The privilege lives in mid-computation state, not pulled-out directions. Object-type ceiling (stated, not hidden): a residual state and a static direction are different objects; the compatible statistic is the target token's rank under jlens-transport vs logit unembed. HYPOTHESIS tier except the M1 leg (VERIFIED); pending Ecaterina's verify line.
- scope: EleutherAI/pythia-410m@9879c9b (+ Pythia-1.4B@fedc38a for M5-0b), EXP-M5-7 + EXP-M5-6 + EXP-M5-0b + M1 (VERIFIED), cached lens draws 0/1/2
- evidence-commit: 5e5a38a
- prereg: harness/preregs/EXP-M5-7-a1b-locus.md
- results-dir: results/m5/20260723-024239-m5-7-a1b-probe
- raw-completions: results/m5/20260723-024239-m5-7-a1b-probe/raw_completions
- headline-cells: capital-operand_draw0, capital-operand_draw1, capital-operand_draw2, capital-recall_draw0, capital-recall_draw1, capital-recall_draw2; EXP-M5-7 decision rule: max-contrast jlens vs logit HMR over 3 lens draws
- verified-by: none

### CLM-009
- status: hypothesis
- statement: On Pythia-410M, label-rank decodability (decode_vector jlens/logit rank of chosen label words) is CONTINGENT on readout-vocabulary adequacy and produced a FALSE NEGATIVE on record (split from CLM-006, 2026-07-23). EXP-M5-8 scored the S5 formality steering vector NOT output-aligned (jlens 30 / logit 347) against the authored words (therefore/furthermore/regarding/...), while the independent, non-confounded top-token read (EXP-M5-8b) finds the SAME vector's top unembed tokens are formal register (Pursuant/Memorandum/Administrative/Certificate). The label-rank S5 output-alignment result (1/4) is superseded by the top-token test (4/4 output-aligned; CLM-007). Consequence: a null label-rank readout is evidence only against the chosen label words, not against decodability — an instrument-validity failure mode distinct from a true negative, and a concrete instance of the instrument itself being wrong, checked by an independent test. HYPOTHESIS tier; pending Ecaterina's verify line.
- scope: EleutherAI/pythia-410m@9879c9b, EXP-M5-8 + EXP-M5-8b, 3 draws, 3 lens draws
- evidence-commit: none
- prereg: harness/preregs/EXP-M5-8b-s5-toptokens.md
- results-dir: results/m5/20260723-042856-m5-8b-s5-toptokens
- raw-completions: results/m5/20260723-042856-m5-8b-s5-toptokens/raw_completions
- headline-cells: draw-based; EXP-M5-8b decision rule: per-vector top-k unembed-token read (3-draw mean direction); formality false-negative vs label-rank
- verified-by: none
