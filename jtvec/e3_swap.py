"""M4-E3: cross-basis FV swap — does the FV causally carry task identity?

On task-A (capitalize) 10-shot prompts, replace the FV_A component of the
residual with FV_B (singular-plural) at the final position of the band
layers, and measure whether the model's answer follows task B for a query
valid under both tasks. Conditions (vendored jvec.evals.fvswap.make_swap_hooks):
none / lens_swap (move in J-lens coordinates) / direct_swap (move in raw
residual space) / random_target (lens_swap toward a norm-matched random
vector). Scope D-016-analog (Ecaterina 2026-07-19): the M3-gated
capitalize->singular-plural pair only; the translation pair (needs
english-spanish certified) is deferred.

Cross-draw (CONSTRAINTS): the swap uses draw k of BOTH certified FVs
(k=1,2,3); the task-B rate under each condition is a 3-draw DrawSet, so
"the swap redirects" is a claim about the certified estimator, not a single
extraction. This module holds the model-free decision rule; the orchestrator
scripts/m4_e3_swap.py drives the model.

Redirection (vs the random control) tests whether the FV carries transferable
task identity; lens_swap vs direct_swap tests whether that identity is
specific to the J-lens basis or lives in the raw residual direction. Nothing
here is asserted as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

from jtvec.core.draws import DrawSet


def rate_drawset(by_draw: dict[int, float]) -> DrawSet:
    """A per-draw B-rate (or gain) map as a DrawSet (>=3 distinct draws)."""
    seeds = tuple(sorted(by_draw))
    return DrawSet(values=tuple(by_draw[s] for s in seeds), seeds=seeds)


@dataclass(frozen=True)
class SwapRedirectionRule:
    """Pre-registered decision rule (EXP-M4-E3-swap; constants D-018).

    - Redirection: the better of {lens_swap, direct_swap} lifts the task-B
      rate over clean by >= min_b_gain (median over draws) while the
      random-target control does NOT (one-sided, D-012 lesson: a random swap
      that destroys computation is not evidence of redirection).
    - Cross-draw transfer: every draw's best-swap gain clears min_b_gain.
    - J-specificity (descriptive verdict): lens_swap median − direct_swap
      median >= min_j_specificity => J-specific; otherwise basis-agnostic
      (the raw residual direction carries identity as well as the lens basis).
    """

    min_b_gain: float = 0.20
    max_random_elevation: float = 0.05
    min_j_specificity: float = 0.15

    def verdict(
        self, *, none_b: float, lens_b: dict[int, float],
        direct_b: dict[int, float], random_b: dict[int, float],
    ) -> dict:
        seeds = sorted(lens_b)
        if not (sorted(direct_b) == sorted(random_b) == seeds):
            raise ValueError("lens/direct/random draws must match")
        best_gain = {s: max(lens_b[s], direct_b[s]) - none_b for s in seeds}
        rand_gain = {s: random_b[s] - none_b for s in seeds}
        lens_ds, direct_ds, random_ds = (rate_drawset(lens_b),
                                         rate_drawset(direct_b), rate_drawset(random_b))
        best_ds, rand_gain_ds = rate_drawset(best_gain), rate_drawset(rand_gain)

        redirects = (best_ds.median >= self.min_b_gain
                     and rand_gain_ds.median <= self.max_random_elevation)
        transfers = all(g >= self.min_b_gain for g in best_gain.values())
        j_specific = (lens_ds.median - direct_ds.median) >= self.min_j_specificity

        if redirects and j_specific:
            verdict = "REDIRECTS-J-SPECIFIC"
        elif redirects:
            verdict = "REDIRECTS-BASIS-AGNOSTIC"
        else:
            verdict = "NO-REDIRECTION"

        return {
            "verdict": verdict,
            "redirects": redirects,
            "cross_draw_transfer": transfers,
            "j_specific": j_specific,
            "none_b": none_b,
            "best_swap_gain_median": best_ds.median,
            "best_swap_gain_iqr": best_ds.iqr,
            "best_swap_gain_values": list(best_ds.values),
            "random_gain_median": rand_gain_ds.median,
            "lens_b_median": lens_ds.median,
            "lens_b_values": list(lens_ds.values),
            "direct_b_median": direct_ds.median,
            "direct_b_values": list(direct_ds.values),
            "random_b_median": random_ds.median,
            "random_b_values": list(random_ds.values),
            "j_specificity_gap": lens_ds.median - direct_ds.median,
        }
