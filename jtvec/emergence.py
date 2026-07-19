"""M4 emergence sweep: model-free logic for the developmental dissociation.

Tests a CONSTRAINTS.md HYPOTHESIS: that ICL task EXECUTION matures early in
training while portable, stability-gated FUNCTION VECTORS (and their
decodability) emerge late (HYPOTHESIS) — a developmental dissociation. The
sweep runs a stability-gated FV extraction (the M2 gate) at each Pythia
training checkpoint, per scale, fixing v1's single-draw + fixed-head
confounds (per-checkpoint AIE re-extraction; per-checkpoint head re-selection
is automatic in compute_function_vector).

This module holds only the model-free summary + onset/classification logic;
scripts/m4_emergence_sweep.py drives the model. Predictions (preregistered,
constants D-019, awaiting ratification):

- P-E1 (dissociation): execution onset << FV-stability onset (a log-step gap).
- P-E2 (co-emergence null): the two onsets coincide.
- P-E3 (scale interaction): the gap grows / appears only at larger scale
  (assessed across per-scale verdicts, not within one).

Nothing here is asserted as a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointRecord:
    """One (scale, checkpoint) summary produced by the sweep driver.

    step: training step (int; >0 for onset log-spacing).
    exec_acc: max over tasks' 10-shot execution accuracy at this checkpoint
        (the "can the model do the task" signal; per-task detail lives in the
        raw sweep.jsonl).
    gate_passed: the per-checkpoint stability gate PASSED (converged_at is not
        None) on >=1 task. None-status checkpoints (too-weak / unstable) set
        this False.
    converged_tasks: how many tasks certified at this checkpoint (0..3).
    outvocab_rank: best output-vocabulary mean rank through this checkpoint's
        lens (decodability; lower = more decodable). None if no FV extracted.
    """

    step: int
    exec_acc: float
    gate_passed: bool
    converged_tasks: int
    outvocab_rank: float | None = None


def onset_step_frac_plateau(records, value_fn, *, frac: float) -> int | None:
    """First step at which value_fn reaches frac x the max observed value.

    Early-onset-and-plateau heuristic: onset = the earliest checkpoint whose
    value is >= frac of the scale's own maximum (robust to absolute scale).
    Returns None if no checkpoint has a positive value.
    """
    vals = [(r.step, value_fn(r)) for r in sorted(records, key=lambda r: r.step)]
    vals = [(s, v) for s, v in vals if v is not None]
    if not vals:
        return None
    vmax = max(v for _, v in vals)
    if vmax <= 0:
        return None
    target = frac * vmax
    for s, v in vals:
        if v >= target:
            return s
    return None


def onset_step_gate(records) -> int | None:
    """First step at which the stability gate PASSED (portable FV emerges)."""
    for r in sorted(records, key=lambda r: r.step):
        if r.gate_passed:
            return r.step
    return None


@dataclass(frozen=True)
class EmergenceRule:
    """Per-scale developmental classification (constants D-019, to ratify).

    exec_plateau_frac: execution onset = first checkpoint reaching this
        fraction of the scale's max 10-shot accuracy.
    min_log10_gap: a dissociation requires FV-stability onset to be at least
        this many log10-steps LATER than execution onset (0.5 ~ 3.2x later).
    min_gate_passes: the sweep's FV-emergence claim for a scale counts only if
        the stability gate passes at >= this many checkpoints (CONSTRAINTS:
        "M2 gate passing at >= 2 checkpoints"); else the scale is INCONCLUSIVE.
    """

    exec_plateau_frac: float = 0.8
    min_log10_gap: float = 0.5
    min_gate_passes: int = 2

    def classify_scale(self, records) -> dict:
        recs = [r for r in records]
        n_gate_passes = sum(r.gate_passed for r in recs)
        exec_onset = onset_step_frac_plateau(recs, lambda r: r.exec_acc,
                                             frac=self.exec_plateau_frac)
        fv_onset = onset_step_gate(recs)

        if exec_onset is None:
            verdict = "NO-EXECUTION"          # task never learned at this scale
        elif n_gate_passes < self.min_gate_passes:
            verdict = "INCONCLUSIVE-FV"       # can't assess FV trajectory
        elif fv_onset is None:
            verdict = "DISSOCIATION-FV-NEVER" # executes but FV never stabilizes
        else:
            gap = math.log10(fv_onset) - math.log10(exec_onset)
            if gap >= self.min_log10_gap:
                verdict = "DISSOCIATION"      # P-E1: execution early, FV late
            elif gap <= -self.min_log10_gap:
                verdict = "REVERSE"           # FV before execution (unexpected)
            else:
                verdict = "CO-EMERGENCE"      # P-E2: onsets coincide

        return {
            "verdict": verdict,
            "exec_onset_step": exec_onset,
            "fv_stability_onset_step": fv_onset,
            "n_gate_passes": n_gate_passes,
            "log10_gap": (None if exec_onset is None or fv_onset is None
                          else math.log10(fv_onset) - math.log10(exec_onset)),
        }


def scale_interaction(per_scale: dict[str, dict]) -> dict:
    """P-E3 across scales: does the dissociation gap grow with scale?

    per_scale maps a scale label (ordered small->large by the caller) to a
    classify_scale result. Reports the gap per scale and whether it is
    monotonically non-decreasing across the scales that have a numeric gap.
    Descriptive; not a single pass/fail (the paper reads the trend).
    """
    gaps = [(scale, res.get("log10_gap")) for scale, res in per_scale.items()]
    numeric = [(s, g) for s, g in gaps if g is not None]
    monotonic = all(numeric[i][1] <= numeric[i + 1][1] + 1e-9
                    for i in range(len(numeric) - 1))
    return {
        "gaps_by_scale": gaps,
        "n_scales_with_gap": len(numeric),
        "gap_monotonic_nondecreasing": monotonic if len(numeric) >= 2 else None,
    }
