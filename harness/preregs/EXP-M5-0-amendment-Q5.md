# EXP-M5-0 amendment — Q5 probing-contrast (evidence-based, D-027 outcome c)

- amends: harness/preregs/EXP-M5-0-qualification.md, rule 5 (1.4B lens gate),
  the Q5 probing-contrast criterion only. All other Q-rules (Q1-Q4, Q6)
  unchanged.
- evidence: EXP-M5-0b diagnostic (harness/preregs/EXP-M5-0b-lens-diagnostic.md;
  run results/m5/20260720-215157-p14b-lens-diagnostic; verdict GAP-RETURNS).
- author + date: Claude (proposal), 2026-07-20. RATIFIED by Ecaterina
  2026-07-21 with two riders (folded in below): (a) the fair-statistic clause —
  all baselines computed under the identical max-contrast statistic — is a
  PERMANENT part of Q5, not a one-run device; (b) latent-intermediate probes
  are, by definition, the A1-relevant anchor class. Committing this file is the
  prereg act.

## Why

The original Q5 used the band-min-jlens-layer metric (from the 410M M1 gate,
where the logit lens was poor even at the J-lens's best layer). On 1.4B the
logit lens catches up faster, so at the band-min-jlens layer it has already
resolved the answer and the contrast vanishes — the gate read a null where the
J-lens signal exists two layers earlier. EXP-M5-0b established this on fresh
matched tasks: the J-lens advantage is present and 3-draw-replicated on
latent-intermediate probes (operand 6.98x, bridge 15.52x) and absent on matched
output probes (~0.9x), under a max-contrast statistic applied identically to
every arm; and the same metric recovers capital-recall to 20.35x (vs the gate's
1.1x). The FAIL was metric + anchor miscalibration, not a J-lens/logit
convergence.

## Amended Q5 (proposed)

Replace the band-min-jlens metric with the EXP-M5-0b max-contrast statistic
(RATIFIED there, 2026-07-20): for arm A over the fixed candidate layer set S
(all source layers), ratio_A = max over L in S with HMR_A(L) <= 5.0 of
logit_HMR(L)/HMR_A(L); applied identically to jlens, the logit denominator, and
each random-matrix arm, per draw.

Rider (a), PERMANENT: the identical-statistic requirement (same candidate layer
set S, same argmax, per draw, for the J-lens numerator, the logit denominator,
and every random-matrix baseline) is a standing part of Q5 on every substrate
going forward, not a one-off for this run. Any Q5 evaluation that does not put
its baselines through the identical statistic is non-conformant.

Rider (b), DEFINITIONAL: latent-intermediate probes ARE the A1-relevant anchor
class, by definition. A1 (decodability) asks whether the workspace holds
content the logit lens cannot read; an output-token probe cannot answer that
(the logit lens reads the emitted token by construction). So the A1 anchor set
is the latent-intermediate probes (held operands / bridges); output-token
probes are recorded controls only and never count toward A1.

Q5 (amended) PASSES on a substrate iff, over the probing-contrast ANCHOR set,
>= 2 LATENT-INTERMEDIATE anchor tasks have median-over-draws ratio_jlens >= 5.0
(jlens HMR <= 5.0) AND every random arm's ratio stays < 5.0 (the amendment
guard). Output-token tasks are recorded but do not count toward the >= 2.
Anchor set = the latent-intermediate probes: capital-operand (operand) plus the
fresh operand/bridge probes; the old output-token anchors (capital-recall
answer, opposites, word-pairs) become recorded controls.

## Admission of 1.4B (formal re-run ORDERED 2026-07-21)

Ecaterina ordered a formal EXP-M5-0-labelled gate re-run under the amended
criterion (~40 min, evals-only on the cached draws); 1.4B is admitted on that
PASS, not on the re-graded diagnostic. Anchor N: the 2hop-bridge probe (N=6) is
included only if it reaches adequate N, else marked descriptive; the well-
powered latent anchors (fresh1hop-operand N=28, capital-operand N=33) carry the
>= 2 requirement.

BLOCKER surfaced (needs a ruling before the re-run can PASS): the original gate
(results/m5/20260720-024819-p14b-lens-gate) failed on Q2 and Q6 as well as Q5.
This amendment fixes Q5 only. Under the amended criterion Q5 now PASSES, but:
- Q2 (positive control): swap dp median 0.483 CLEARS the 0.30 dp bar, but the
  top-1 flip rate 0.5625 MISSES the 0.75 flip bar.
- Q6 (draw stability): dp IQR 0.0707 > 0.05 (driven by draw-1; band-min HMR IQR
  is fine).
Both are swap-intervention rules, untouched by the probing-contrast amendment,
so a re-run today returns FAIL and does not admit 1.4B. See the D-029 proposal
in LABNOTES (options: recalibrate the Q2 flip / Q6 IQR bars per substrate on the
same evidence-based footing as Q5 — the swap moves probability strongly (dp
0.48) but a 410M-derived argmax-flip bar may be too brittle; investigate the
swap's per-draw weakness; or scope this admission to A1 (Q5) and hold the
swap-dependent A2/A3 admission pending Q2/Q6). Not adopted — flagged.

## Scope / non-goals

Methodological only — this admits 1.4B as a substrate; it asserts nothing about
workspace residency (that is the M6 axis battery on certified species). 410M
was already admitted (M1). 2.8B escalation stays un-triggered.
