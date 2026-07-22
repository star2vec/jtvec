# Preregistration — EXP-M5-7: locus of the J-lens advantage (A1b rescope)

- experiment-id: EXP-M5-7
- status: **RATIFIED** by Ecaterina 2026-07-23. Rescope confirmed; the 410M
  confirmatory probe is RUN (not skipped — see below). Committing this file is
  the prereg act; the probe runs on the Mac (~15 min); the rescoped claim + the
  unified table (transient 410M row measured in-run) are finalized on that
  result. Bars = the E1/M5-0b/M1 statistics already on record.
- claim: rescopes the CLM-006 sub-statement "logit-privilege (A1b) is empty".
- author + date: Claude (proposal), 2026-07-23, per Ecaterina.

## Why

CLM-006 currently records "no measured representation is logit-privileged (A1b)".
As written that is self-undermining — it implies the J-lens never beats the logit
lens, so why use it. But the J-lens advantage IS real; it just lives in a
different object than the extracted static direction vectors we tested for A1b.
EXP-M5-0b (and the VERIFIED M1 gate) already established it on TRANSIENT
computational intermediates. The claim must be rescoped to locate the advantage,
not deny it.

## The distinction (the rescoped claim)

- **Extracted STATIC direction vectors are NOT J-lens-privileged.** Under
  decode_vector (unembed the direction), on 410M: S1 concept jlens/logit
  label-rank 192/999, S2 FV 436/3203 (both lens-dark — neither lens reads them
  well), S5 steering 2/2 (logit-trivially readable — J-lens = logit, no privilege).
- **TRANSIENT computational intermediates ARE J-lens-privileged.** Probing the
  residual STATE at the held-intermediate position: M1 410M capital-recall
  jlens HMR 2.5 vs logit 61.5 (~25x; VERIFIED, results/phase1_report + M1 gate);
  EXP-M5-0b 1.4B held operand 6.98x, bridge 15.52x (max-contrast, 3 draws), while
  the matched OUTPUT token is ~0.9x (NO advantage — the logit lens reads the
  output by construction).
- **Rescoped statement:** the J-lens's advantage lives in MID-COMPUTATION STATE
  (transient held intermediates), NOT in pulled-out static direction vectors; and
  not on the output token either. Extracted static directions are not
  logit-privileged; transient computational intermediates are.

## The unified rank table (compatible statistic: jlens rank vs logit rank of the target)

Static rows use decode_vector label-rank; transient rows use the probe HMR of the
held token. The OBJECT differs (a pulled-out direction vs a mid-stream residual
state) — that difference IS the finding — but the read statistic (rank of the
target under jlens-transport vs logit unembed) is compatible.

| object | type | substrate | jlens rank | logit rank | J-lens vs logit |
|---|---|---|---|---|---|
| capital-recall intermediate (M1 gate) | transient state | 410M | 2.5 | 61.5 | ~25x (VERIFIED) |
| fresh 1-hop operand (M5-0b) | transient state | 1.4B | ≤5.0 | — | 6.98x |
| fresh 2-hop bridge (M5-0b) | transient state | 1.4B | ≤5.0 | — | 15.52x |
| fresh output token (M5-0b) | output token | 1.4B | — | — | 0.89x (none) |
| S1 concept dir (M5-6) | static direction | 410M | 192 | 999 | dark (both) |
| S2 FV (E1, cited) | static direction | 410M | 436 | 3203 | dark (both) |
| S5 steering dir (M5-6) | static direction | 410M | 2 | 2 | 1.0x (logit-trivial) |

Contrast, shown not asserted: transient states → J-lens ranks the held content in
the top few (jlens ≤ 5) while the logit lens is far off; static directions →
J-lens dark (192/436) or only logit-trivial (S5 2=2). The J-lens wins on
transient state, loses/ties on pulled-out directions.

## Confirmatory 410M probe — RUN (Ecaterina ruled 2026-07-23)

Ecaterina ruled the probe RUNS (do not skip): a reviewer attacks the softer
"different experiment, different day" gap first, so the ~15-min probe is cheap
insurance on the paper's softest target.

Probe: on 410M, cached lens draws 0/1/2 (the SAME draws as the static S1/S2/S5
rows), probe the held capital-operand LATENT intermediate (the certified
capital-operand task's held operand, the M1/M5-0b anchor) via the jlens vs logit
readout, and record the jlens-vs-logit rank of the held token — same substrate,
same lens draws, same run as the static rows. Also record the OUTPUT-token
readout as the internal null (expected no J-lens advantage). ≥3 lens draws,
median/IQR.

OPENLY STATED CEILING (not hidden): this removes the CROSS-EXPERIMENT /
cross-substrate objection (transient 410M row now in-run, identical lens draws),
but it does NOT unify OBJECT TYPES — a transient residual state and a pulled-out
static direction are intrinsically different objects, and no run can make them
the same object. The compatible "rank of the target under jlens-transport vs
logit" framing is the honest ceiling; the object difference IS the finding.

Machine: Mac (410M, cached lenses). The transient 410M row of the unified table
is then the IN-RUN measurement (with the M1 gate's VERIFIED 2.5/61.5 retained as
the cross-check).

## What counts as failure / honesty

- The transient vs static contrast is the claim; it is HYPOTHESIS-tier except the
  M1 410M leg (VERIFIED). The output-token null (~0.9x) is retained as the
  internal control that the advantage is specific to held latents, not outputs.

## Deliverable

Rescope CLM-006's A1b sub-statement to the transient-vs-static locus + add the
unified rank table to the record, on Ecaterina's ratification. No claim edited
until she rules.

## Ratification

ratified: EXP-M5-7 2026-07-23 — Ecaterina. Rescope confirmed; the 410M
confirmatory probe RUNS (not skipped) to close the cross-experiment gap, with the
object-type ceiling stated openly. Runs on the Mac; the unified table's transient
410M row is finalized from the in-run measurement, M1 gate (VERIFIED) retained as
cross-check. Then CLM-006's A1b sub-statement is rescoped to the transient-vs-
static locus on Ecaterina's verify.
