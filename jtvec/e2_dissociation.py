"""M4-E2: execution/verbalization dissociation on singular-plural.

Tests the CONSTRAINTS hypothesis that task execution and task verbalization
are causally separable (HYPOTHESIS), via v1's Exp-3 double-dissociation design
rebuilt on v2's gated apparatus (D-016 Path A, singular-plural only;
landmark-country deferred):

- two ablations, both M3-gated, at the final position of the band layers:
  fv-direction-ablation (project out the certified Todd FV) and jspace-ablation
  (project out the top-m J-lens-readout atoms), each vs its matched sham;
- two measures: execution (greedy top-1 accuracy on held-out singular->plural
  queries, exact-match) and report (report_score under P3, the report
  instrument gated this session);
- cross-draw ablation transfer (CONSTRAINTS requirement): the fv ablation is
  re-derived from each of the 3 M2-certified FV draws; the jspace ablation
  from each of the 3 M1 lens draws (jspace reads the lens, so its nuisance
  axis is the lens draw — the E1 lens-draw-marginalization lesson). Every
  ablation effect is therefore a >=3-draw DrawSet (median/IQR), per the draws
  LAW.

Double dissociation (HYPOTHESIS) = fv ablation cuts execution beyond its sham
while NOT cutting report, AND jspace ablation cuts report beyond its sham
while NOT cutting execution. This module holds the model-free decision rule
and the cross-draw effect structure; scripts/m4_e2_dissociation.py drives the
model. Nothing here states the hypothesis as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

from jtvec.core.draws import DrawSet


def effect_drawset(clean_mean: float, ablated_by_draw: dict[int, float]) -> DrawSet:
    """Per-draw ablation effect (clean - ablated) as a DrawSet.

    A positive value means the ablation HURT the measure. The draws are the
    FV draws (fv arm) or lens draws (jspace arm); the clean run is shared
    across draws (it does not depend on which ablation direction is used), so
    the cross-draw spread is the ablation's own draw variability. Raises via
    DrawSet if fewer than 3 draws or non-distinct seeds.
    """
    seeds = tuple(sorted(ablated_by_draw))
    values = tuple(clean_mean - ablated_by_draw[s] for s in seeds)
    return DrawSet(values=values, seeds=seeds)


@dataclass(frozen=True)
class DissociationRule:
    """Pre-registered decision rule (EXP-M4-E2-dissociation; constants D-017).

    An ablation "hurts" a measure iff its effect DrawSet median exceeds its
    sham DrawSet median by at least the measure's minimum drop delta. The
    deltas are bright lines fixed before the run; per-draw values and IQRs are
    reported alongside so an effect whose draws straddle a delta is visible as
    non-transferring rather than silently classified.
    """

    min_exec_drop: float = 0.15    # delta_exec (matches the M3 fv-ablation bar)
    min_report_drop: float = 0.10  # delta_report (log-prob; ~half the ~0.22 P3 margin)

    def hurts(self, effect: DrawSet, sham: DrawSet, delta: float) -> bool:
        return (effect.median - sham.median) >= delta

    def transfers(self, effect: DrawSet, sham: DrawSet, delta: float) -> bool:
        """Cross-draw transfer: EVERY draw's effect clears sham-median + delta.

        A median that fires while individual draws disagree is flagged as a
        non-transferring effect (the CONSTRAINTS cross-draw-transfer check),
        not counted as a clean 'hurts'.
        """
        return all((v - sham.median) >= delta for v in effect.values)

    def verdict(
        self,
        *,
        fv_exec: DrawSet, fv_exec_sham: DrawSet,
        fv_report: DrawSet, fv_report_sham: DrawSet,
        jspace_exec: DrawSet, jspace_exec_sham: DrawSet,
        jspace_report: DrawSet, jspace_report_sham: DrawSet,
    ) -> dict:
        de, dr = self.min_exec_drop, self.min_report_drop
        fv_hurts_exec = self.hurts(fv_exec, fv_exec_sham, de)
        fv_hurts_report = self.hurts(fv_report, fv_report_sham, dr)
        js_hurts_report = self.hurts(jspace_report, jspace_report_sham, dr)
        js_hurts_exec = self.hurts(jspace_exec, jspace_exec_sham, de)

        # Direction 1: fv ablation dissociates execution FROM report.
        dir1 = fv_hurts_exec and not fv_hurts_report
        # Direction 2: jspace ablation dissociates report FROM execution.
        dir2 = js_hurts_report and not js_hurts_exec

        if dir1 and dir2:
            verdict = "DOUBLE-DISSOCIATION"
        elif dir1 or dir2:
            verdict = "ONE-WAY"
        else:
            verdict = "NO-DISSOCIATION"

        return {
            "fv_hurts_exec": fv_hurts_exec,
            "fv_hurts_report": fv_hurts_report,
            "jspace_hurts_report": js_hurts_report,
            "jspace_hurts_exec": js_hurts_exec,
            "direction1_fv_exec_not_report": dir1,
            "direction2_jspace_report_not_exec": dir2,
            "verdict": verdict,
            "cross_draw_transfer": {
                "fv_exec": self.transfers(fv_exec, fv_exec_sham, de),
                "jspace_report": self.transfers(jspace_report, jspace_report_sham, dr),
            },
            "effects": {
                "fv_exec": _summ(fv_exec, fv_exec_sham),
                "fv_report": _summ(fv_report, fv_report_sham),
                "jspace_exec": _summ(jspace_exec, jspace_exec_sham),
                "jspace_report": _summ(jspace_report, jspace_report_sham),
            },
        }


def _summ(effect: DrawSet, sham: DrawSet) -> dict:
    return {
        "effect_median": effect.median,
        "effect_iqr": effect.iqr,
        "effect_values": list(effect.values),
        "sham_median": sham.median,
        "sham_iqr": sham.iqr,
        "sham_values": list(sham.values),
        "effect_minus_sham": effect.median - sham.median,
    }
