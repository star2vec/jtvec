# Preregistration — EXP-M5-0b: 1.4B lens diagnostic (latent-vs-output probe)

- experiment-id: EXP-M5-0b
- claim: none (diagnostic; decides the D-027 fork — anchor/metric
  miscalibration vs a real J-lens/logit convergence at 1.4B — before any
  2.8B escalation)
- model: EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79
- lenses: the 3 already-fitted M5.0 draws
  (cache/m5/p14b_draw{0,1,2}/.../skip4_n10) — NO REFIT (D-027 ruling)
- config: configs/m5_0b_diagnostic_pythia1p4b.yaml (draft alongside)
- author + date: Claude (proposal), 2026-07-20, per Ecaterina's D-027
  ruling (diagnostic before escalation). RATIFIED by Ecaterina 2026-07-20
  (second batch) with one amendment: all comparison quantities (logit
  denominator, any random baseline) are computed under the identical
  max-contrast-layer statistic as the J-lens numerator (same layers, same
  max-taking, per draw) — folded into the Decision rule and Instruments
  below. Thresholds (jlens-HMR cap 5.0, advantage ratio 5.0, >=2-latent-task
  rule, 0.6 behavioural bar) ratified as drafted. Committing this file is
  the prereg act.

## Hypothesis

The M5.0 lens gate FAILED on 1.4B (Q5: J-lens beat logit >=5x on only 1 of 4
anchors). This diagnostic tests WHY, to decide D-027. The mechanism under
test: the J-lens's advantage over the logit lens is specific to reading
LATENT INTERMEDIATE content the model is holding but not about to emit; the
logit lens surfaces the OUTPUT token by construction. Three of the four gate
anchors (capital-recall, opposites, word-pairs) probed the output/answer
token, which the logit lens reads directly; only capital-operand probed a
latent (the country operand) — and it alone passed Q5. If the J-lens
advantage returns on latent-intermediate probes and is absent on matched
output probes, the gate's anchor set was miscalibrated for the J-lens's
function (D-027 branch: amend anchors/metric, admit 1.4B). If the advantage
is absent even on latent probes, the J-lens and logit lens have genuinely
converged at this scale (D-027 branch: register the deflation question,
justify 2.8B as its test).

Post-hoc motivation (LABELLED POST-HOC — analysis of the failed run's stored
tensors, EXP-M5-0 run 20260720-024819; not decision-bearing here): under a
max-contrast-layer metric rather than the gate's band-min-jlens-layer metric,
capital-recall shows a 25.8x J-lens/logit ratio at L14 (the J-lens leads by
~2 layers, then the logit lens catches up by L15 where the band-min metric
read it). capital-operand 31.7x, opposites 1.6x, word-pairs 1.9x. This
motivates (a) the max-contrast metric and (b) the latent-vs-output split
below, but the DECISION rests on the FRESH matched tasks, not this re-read.

## Decision rule (bars [proposed] until ratified)

Primary evidence = FRESH matched task pairs the lenses were fit blind to
(pile-10k calibration; these countries/facts are not in any gate task).

Metric per (task, draw) — the max-contrast statistic, applied IDENTICALLY to
every arm (RATIFIED amendment, Ecaterina 2026-07-20): over a fixed candidate
layer set S (all source layers 0..n-2, the same S for every arm and draw),
for a readout arm A (A = jlens, or the random-matrix arm) define
  ratio_A = max over L in S with HMR_A(L) <= 5.0 of  logit_HMR(L) / HMR_A(L).
The max-taking (argmax over the same S, the same jlens-HMR<=5 admissibility
recast per arm as HMR_A(L)<=5, per draw) is identical for the J-lens
numerator and for the random baseline; the logit_HMR(L) denominator terms
are read at whatever layer that arm's statistic selects, per draw. Report the
selected layer and both HMRs for each arm. Marginalize by reporting per-draw
and the median over the 3 draws (E1 lesson). (Rationale: picking the most
favourable layer can inflate a ratio; subjecting the random arm to the
identical selection makes that inflation show up in the control, so the
J-lens verdict only counts if it beats what the same procedure gives noise.)

A task "shows the J-lens advantage" iff its median (over 3 draws) ratio_jlens
>= 5.0 [ratified] with the corresponding jlens HMR <= 5.0, AND the random
arm's ratio_random under the identical statistic stays < 5.0 on that task
(negative control, below).

- GAP RETURNS iff >= 2 latent-intermediate FRESH tasks show the advantage
  AND their matched output-probe FRESH tasks do NOT (the dissociation).
  -> D-027 outcome (c): the gate's anchors/metric were miscalibrated;
  propose an amended EXP-M5-0 Q5 (latent-intermediate anchors + max-contrast
  metric) for ratification, then re-run the (cheap, evals-only) gate on the
  cached draws; 1.4B admitted if it then passes.
- GAP DOES NOT RETURN (latent-intermediate tasks also < 5x) -> D-027
  outcome (b): the J-lens/logit convergence at 1.4B is real; open it as a
  first-class registered question (the TAXONOMY_DESIGN deflation branch),
  and the 2.8B escalation is then justified as that question's test.

Mixed/partial outcomes are reported per-task; no outcome is unpublishable
(every task is pre-registered).

## What counts as failure

- This is a diagnostic, so both outcomes above are informative, not
  "failures". The failure modes are procedural: a fresh latent task the
  model cannot do behaviourally (drop it, report; probe only items the
  model gets right, N recorded), or an instrument-control failure (below).
- Post-hoc re-reads of the existing gate anchors stay labelled post-hoc.

## Estimator plan

The estimator is the already-FITTED (not gate-passing) J-lens per draw; this
diagnostic does not re-fit. Per (task, draw) the probe computes per-layer
jlens and logit HMR over the task's correct items (the vendored scripts/03
probe path). 3 draws = the 3 cached lenses; the max-contrast ratio is
summarised by its median and per-draw values. Deterministic given the lenses
(greedy readout); no new RNG.

## Instruments

- J-lens readout (per-layer HMR), positive control = the fresh
  latent-intermediate tasks are predicted to separate from their matched
  output tasks; negative control = the 10-seed random-matrix probe arm,
  put through the IDENTICAL max-contrast statistic (same S, same max-taking,
  per draw), must yield ratio_random < 5.0 on every task. If any random arm
  reaches >= 5.0 the max-contrast statistic is itself withdrawn (the
  selection inflates noise) and the verdict does not count (instruments LAW)
  — this is the amendment's guard.
- Behavioural gate per fresh task: the model's greedy top-1 must match the
  target on the probed items (>= 0.6 accuracy [proposed] to include a task;
  probe restricted to correct items).

## Sample plan

Fresh matched battery (generated via scripts/make_tasks.py designs with
countries/facts held out of every gate task; deterministic seed):
- FRESH-1hop: capital prompts over >= 12 fresh countries. Two probes on the
  SAME prompts: latent = the country operand; output = the answer capital.
- FRESH-2hop: "capital of the country famous for X" over >= 12 fresh clues.
  Two probes: latent = the bridge country; output = the answer capital.
Existing tasks re-probed for continuity (POST-HOC labelled, not
decision-bearing): capital-operand, capital-recall, multihop-scaled,
opposites, word-pairs.
Raw per-item records retained per (task, draw): per-layer jlens/logit ranks.
N recorded per task (correct-item subset). All cells >= 20 items where the
task admits it; fresh tasks sized to clear 20 correct items.

## Resource estimate (Mac tier, MPS fp32)

Probe-only on the 3 cached lenses, no refit. Measured M5.0 eval cost ~16
min/draw for 4 tasks; this battery is ~8-10 tasks over 3 draws ->
projected ~1.5-2.5 h wall, peak RSS ~7.5 GB (model + one lens at a time).
Under the 12 h LAW; Mac-eligible. Launch detached + Monitor per the sleep
lesson; caffeinate bound to the run PID.

## Deviations

- Uses the max-contrast-layer metric, not the gate's band-min-jlens-layer
  metric. Motivated post-hoc (above) but pre-specified here as the
  diagnostic's registered metric; if the diagnostic supports it, the Q5
  amendment carries it into EXP-M5-0 by a separate ratified amendment.
