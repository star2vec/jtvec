"""M4-E2 prerequisite: a re-controlled report instrument for singular-plural.

D-013 withdrew report-probe-forced-choice@singular-plural. Its forced-choice
argmax null still read the "plural" label 26/36 (M3 run 3) because the label
is legible from the singular-noun INPUTS, independent of the task mapping.
This module rebuilds the report measure as v1's prior-corrected report SCORE
(vendored scripts/11 protocol) and re-controls it with the D-013 failure mode
turned into a pass/fail negative control:

    report_score(ctx) = log p(label | ctx+probe)
                        - mean_neutral log p(label | neutral+probe)

- positive control: the coherent-context report_score is elevated above the
  neutral prior (bootstrap CI-low > 0) — the instrument reads the task when
  it is present.
- negative control: the elevation is SPECIFIC to the coherent task mapping —
  coherent CI-low > shuffled CI-high (shuffled keeps the singular-noun inputs
  but scrambles the mapping: the direct D-013 discriminator) AND coherent
  CI-low > other CI-high (a different task's context scored for "plural":
  rules out a bare label/grammar prior).

An instrument that passes both is named report-score-prior-corrected@<task>
and becomes the verbalization measure E2 consumes. If it fails, the task has
no valid report measure on this model and E2's dissociation there is blocked
— a scope result, reported not worked around (instruments LAW).

Model-free logic only (bootstrap CIs, the control rule, the verdict); the
orchestrator scripts/m4_e2_reportgate.py supplies the per-trial scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The label word scored per task (vendored jvec.evals.exp3.REPORT_LABELS).
#: E2's prerequisite gates singular-plural; the map is kept task-general so the
#: same instrument can be re-controlled on another task under its own prereg.
from jvec.evals.exp3 import REPORT_LABELS  # noqa: E402


def instrument_name(task: str) -> str:
    """The @task-scoped instrument name (M2 fv_todd@<task> / D-013 convention)."""
    return f"report-score-prior-corrected@{task}"


def bootstrap_ci(
    values, *, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(mean, lo, hi) with a percentile bootstrap CI at level 1-alpha.

    Deterministic given seed (np.random.default_rng(seed)); no wall-clock or
    global RNG state enters, so a re-run reproduces the interval exactly.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values for bootstrap CI")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


@dataclass(frozen=True)
class ReportScoreControlRule:
    """Positive+negative control for a prior-corrected report score.

    Both arms are CI-level (no point estimate crosses the other's interval),
    so the rule is quantization-free by construction — it compares continuous
    bootstrap intervals, not discrete rates (contrast the D-010 lesson, which
    applied to rate readouts). All three phrasings are evaluated; the
    instrument is gated iff SOME single phrasing passes both arms together
    (a positive under one phrasing and a negative under another is not one
    working instrument).
    """

    n_boot: int = 10_000
    boot_seed: int = 0

    def evaluate_phrasing(self, *, coherent, shuffled, other) -> dict:
        coh = bootstrap_ci(coherent, n_boot=self.n_boot, seed=self.boot_seed)
        shuf = bootstrap_ci(shuffled, n_boot=self.n_boot, seed=self.boot_seed)
        oth = bootstrap_ci(other, n_boot=self.n_boot, seed=self.boot_seed)
        positive = coh[1] > 0.0
        negative = coh[1] > shuf[2] and coh[1] > oth[2]
        return {
            "coherent_ci": coh,
            "shuffled_ci": shuf,
            "other_ci": oth,
            "n": {"coherent": len(coherent), "shuffled": len(shuffled),
                  "other": len(other)},
            "positive_pass": positive,
            "negative_pass": negative,
            "gated": positive and negative,
        }

    def verdict(self, per_phrasing: dict[str, dict]) -> dict:
        if not per_phrasing:
            raise ValueError("no phrasings evaluated")
        best = next((name for name, v in per_phrasing.items() if v["gated"]), None)
        return {
            "per_phrasing": per_phrasing,
            # ControlRecord booleans: a positive somewhere; a negative that
            # co-occurs with detection (i.e. the gate is witnessed together).
            "positive": any(v["positive_pass"] for v in per_phrasing.values()),
            "negative": best is not None,
            "gated": best is not None,
            "best_phrasing": best,
        }
