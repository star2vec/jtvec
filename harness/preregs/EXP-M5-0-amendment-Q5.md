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

D-029 RULED (Ecaterina, 2026-07-21) — option (iii), per-axis-class scoped
admission: gate admission is scoped by axis class.
- **Amended-Q5 PASS admits Pythia-1.4b@fedc38a for A1 and A4 work** (the
  lens-readout / decodability + report-coupling axes). The formal amended-Q5
  PASS is on record at results/m5/20260721-001417-p14b-lens-gate-amended.
- **Q2 and Q6 remain BLOCKING for A2/A3 admission** (the swap-potency /
  basis-mediation axes). No Q2/Q6 bar change without an EXP-M5-0c verdict on
  record first (harness/preregs/EXP-M5-0c-swap-decomposition.md, drafted).
- If EXP-M5-0c shows a genuinely reduced swap gap-shift at 1.4B (vs a
  flip-rate/base-margin confound), that is registered as a potency-scaling
  observation, HYPOTHESIS tier — NOT treated as a gate failure.

## Scope / non-goals

Methodological only — this admits 1.4B as a substrate; it asserts nothing about
workspace residency (that is the M6 axis battery on certified species). 410M
was already admitted (M1). 2.8B escalation stays un-triggered.
