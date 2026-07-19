# Preregistration — EXP-M4-E2-reportgate: report instrument for singular-plural

- experiment-id: EXP-M4-E2-reportgate
- claim: none (instrument gate; produces the ControlRecord pair E2's prereg
  cites for its verbalization measure — no CLAIMS.md entry moves)
- model: EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
- config: configs/m4_e2_reportgate_pythia410m.yaml
- author + date: Claude (proposal), 2026-07-19; scope ruled by Ecaterina
  2026-07-19 (D-016, Path A: unblock singular-plural's report measure before
  E2); thresholds ratified by Ecaterina 2026-07-19 (session Q&A). Committing
  this file is the prereg act.

## Hypothesis

No scientific hypothesis is tested. This run applies the instruments LAW to
rebuild the report measure withdrawn at D-013
(report-probe-forced-choice@singular-plural: its forced-choice null read the
"plural" label 26/36 because the label is legible from the singular-noun
inputs, independent of the task mapping). The rebuild is v1's prior-corrected
report SCORE (vendored scripts/11 protocol), re-controlled under a new name,
with the D-013 failure mode turned into a pass/fail negative control. E2
(the dissociation) stays separate and does not start unless this gate passes.

## Decision rule

For a context ctx and probe phrasing p:

    report_score(ctx, p) = log p(" plural" | ctx + p)
                          - baseline(p)
    baseline(p) = mean over N_neutral no-coherent-rule contexts of
                  log p(" plural" | neutral + p)

" plural" = REPORT_LABELS["singular-plural"], scored by the first token of
" plural" (vendored label_token_ids convention). Three cells per phrasing,
N=40 trials each: coherent (real singular-plural 10-shot context), shuffled
(the same singular-noun inputs, outputs scrambled — mapping destroyed, inputs
kept), other (a different certified task's 10-shot context, scored for
" plural"). Phrasings P1, P2, P3 (vendored REPORT_PROBES), all reported.
Bootstrap 95% CIs: 10,000 resamples, seed 0.

Per phrasing:
- positive control (detection): coherent bootstrap CI-low > 0 — the coherent
  context elevates " plural" above the neutral prior.
- negative control (specificity, the D-013 discriminator): coherent CI-low >
  shuffled CI-high AND coherent CI-low > other CI-high — the elevation
  requires the coherent task MAPPING, not the singular-noun inputs (shuffled
  holds them fixed) and not a bare label/grammar prior (other).

The instrument report-score-prior-corrected@singular-plural is GATED iff SOME
single phrasing passes BOTH arms (a positive under one phrasing and a negative
under another is not one working instrument). All three phrasings are
reported regardless. The ControlRecord pair (run, passed, date) is the
deliverable, in controls.json.

[Thresholds await Ecaterina's ratification; the constants in
jtvec/report_instruments.py and scripts/m4_e2_reportgate.py match this file.]

## What counts as failure

- Not gated (no phrasing shows coherent elevated AND specific): singular-plural
  has no valid report measure on this model/config. Under Path A this BLOCKS
  E2's double dissociation on singular-plural (the verbalization half is
  unmeasurable); the disposition — respecify E2's task, accept only a
  one-directional test, or a further redesign — is Ecaterina's, flagged. This
  is a live outcome: the D-013 evidence makes coherent≈shuffled plausible, and
  the gate is built to refuse certification if so rather than paper over it.
- A gate that passes here is still only the CLEAN report instrument; E2's use
  of it under ablation inherits nothing beyond this control pair.
- Post-hoc analyses of this run's records are labeled post-hoc forever.

## Estimator plan

No stochastic model estimator is drawn: no FV (no extraction RNG), no lens.
Report scores are deterministic given the preregistered context seed
(CTX_RNG_SEED=7070); the bootstrap CI is a data-resampling interval over the
fixed N=40 per-cell scores (seed 0), not a re-draw of a model estimator, so
the 3-draw LAW — which governs stochastic estimators with an RNG over model
draws — does not apply (identical treatment to the M3 control run). Per-item
raw records are retained for every reported number. Any later use of this
instrument on a re-drawn estimator (E2 ablations across the 3 certified FV
draws) inherits the M2 draw discipline in E2's own prereg.

## Instruments

This run IS the instrument-control run for
report-score-prior-corrected@singular-plural. Its positive and negative arms
are specified in the Decision rule; the ControlRecord pair points at this
run's results directory. No other instrument is read (no lens, no FV, no
forced-choice probe — the D-013-banned name is untouched).

## Interventions and shams

None. The gate validates the CLEAN report instrument; no model computation is
modified, so the sham-twin LAW is not triggered. The shuffled and other cells
are the reading-level nulls and are reported next to coherent in the same
table. (E2 later applies the M3-gated fv-direction and jspace ablations to
this instrument; those interventions carry their shams in E2's prereg.)

## Sample plan

- Target task: singular-plural. Neutral pool and other-context cells draw
  from the non-target certified tasks {capitalize, english-french}.
- N=40 trials per (cell x phrasing); neutral pool N=40 per phrasing; 3
  phrasings P1/P2/P3, all reported.
- Raw cells (>= 20 records each by construction): report_coherent_P{1,2,3},
  report_shuffled_P{1,2,3}, report_other_P{1,2,3}, neutral_P{1,2,3}.

## Resource estimate

480 forward passes (neutral 120 + coherent/shuffled/other 360), each one ICL
context (10-shot + probe) — single forward, no backward, no lens transport.
At ~0.1-0.3 s/forward on this GPU (D-009 stack) plus ~1 min model load: ~3-8
min wall-clock, bounded 15 min. Peak RSS <= ~4 GB; VRAM ~2/8.2 GB. Under the
10-min flag and far under the 12 h LAW; the run still launches only after this
file's commit (prereg-before-run LAW) and Ecaterina's threshold ratification.

## Deviations

(none at commit time)
