# Preregistration — EXP-<id>: <title>

- experiment-id: EXP-<id>
- claim: CLM-<id> (ledger entry this experiment can move)
- model: <name>@<revision, pinned>
- config: <path to committed config>
- author + date: <name>, <ISO date>

LAW: this file must be committed before the first run. `start_run` refuses to
launch otherwise. Every section below is mechanically required
(`jtvec.core.runctx.REQUIRED_PREREG_SECTIONS`). Deviations after the first
run require a flagged decision by Ecaterina, recorded in LABNOTES and in the
Deviations section here.

## Hypothesis

<CONSTRAINTS.md tier + the statement being tested. HYPOTHESIS-tier statements
carry the HYPOTHESIS tag.>

## Decision rule

<The exact numbers to be computed and the thresholds that decide the outcome,
fixed now. Include the sham comparison for interventions.>

## What counts as failure

<The concrete result that would count against the hypothesis, and what
happens on non-convergence or a gate failure.>

## Estimator plan

<Number of independent draws (>= 3), seeds, and the gate certificate
(estimator, converged sample size, evidence run) for every FV-dependent
quantity. Median/IQR is the only summary reported.>

## Instruments

<Each instrument used, with its positive and negative control runs.>

## Interventions and shams

<Spec (kind, layers, positions, direction count, norm). Sham twins are
auto-generated and matched; results include them by construction.>

## Sample plan

<N per cell, tasks, and for report probes: the >= 3 pre-specified phrasings,
all of which are reported.>

## Resource estimate

<Wall-clock + peak memory, estimated before any run > 10 min.
Anything projected > 12 h moves to the A100, not the M1 laptop.>

## Deviations

<Empty at commit time. Any later entry must reference the LABNOTES decision
that authorized it.>
