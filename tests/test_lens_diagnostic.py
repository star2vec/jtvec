"""Model-free landing test for the EXP-M5-0b verdict logic (jtvec.lens_diagnostic).

Feeds synthetic probe-metric dicts and asserts the max-contrast statistic, the
identical-arms random control, and the >=2-dissociating-pairs decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jtvec.lens_diagnostic import (
    amended_q5_verdict,
    arm_max_contrast,
    diagnostic_verdict,
    task_arm_ratios,
    task_shows_advantage,
)

LAYERS = list(range(23))
CAP = 5.0
RATIO = 5.0


def _per_layer(hmr_by_layer: dict[int, float]) -> dict:
    return {str(l): {"hmr": hmr_by_layer.get(l, 40.0), "pass@10": 0.0} for l in LAYERS}


def _metrics(jlens_hmr, logit_hmr, random_hmr=40.0, jlens_layer=14):
    """A task's probe metrics: jlens dips to jlens_hmr at jlens_layer where the
    logit lens is still logit_hmr; 10 random arms flat at random_hmr."""
    m = {
        "jlens": {"per_layer": _per_layer({jlens_layer: jlens_hmr})},
        "logit": {"per_layer": _per_layer({jlens_layer: logit_hmr})},
    }
    for s in range(10):
        m[f"random-{s}"] = {"per_layer": _per_layer({jlens_layer: random_hmr})}
    return m


def test_arm_max_contrast_picks_best_layer():
    jl = _per_layer({10: 2.0, 14: 1.0})
    lo = _per_layer({10: 20.0, 14: 8.0})
    ratio, layer = arm_max_contrast(jl, lo, LAYERS, CAP)
    assert layer == 10 and ratio == 10.0  # 20/2 beats 8/1


def test_arm_ignores_layers_above_cap():
    jl = _per_layer({14: 6.0})  # 6 > cap 5 -> excluded
    lo = _per_layer({14: 60.0})
    ratio, layer = arm_max_contrast(jl, lo, LAYERS, CAP)
    assert layer is None and ratio == 0.0


def test_task_arm_ratios_reports_random_worstcase():
    # one random arm spikes to a low HMR (would inflate its ratio)
    m = _metrics(jlens_hmr=1.0, logit_hmr=30.0, random_hmr=40.0)
    m["random-3"]["per_layer"]["14"]["hmr"] = 1.5  # 30/1.5 = 20 for this arm
    r = task_arm_ratios(m, LAYERS, CAP)
    assert r["jlens_ratio"] == 30.0
    assert r["random_max_ratio"] == 20.0  # worst-case random captured


def test_task_shows_advantage_true():
    draws = [{"jlens_ratio": 25.0, "random_max_ratio": 1.2}] * 3
    shows, diag = task_shows_advantage(draws, RATIO)
    assert shows is True and diag["jlens_ratio_median"] == 25.0


def test_task_advantage_blocked_when_random_also_high():
    # jlens clears but the random control also reaches the bar -> withdrawn
    draws = [{"jlens_ratio": 25.0, "random_max_ratio": 6.0}] * 3
    shows, _ = task_shows_advantage(draws, RATIO)
    assert shows is False


def test_verdict_gap_returns_on_two_dissociating_pairs():
    strong = [{"jlens_ratio": 20.0, "random_max_ratio": 1.0}] * 3   # latent shows
    flat = [{"jlens_ratio": 1.1, "random_max_ratio": 1.0}] * 3       # output doesn't
    per_task = {
        "fresh1hop-operand": strong, "fresh1hop-answer": flat,
        "fresh2hop-bridge": strong, "fresh2hop-answer": flat,
    }
    v = diagnostic_verdict(per_task, RATIO)
    assert v["verdict"] == "GAP-RETURNS"
    assert v["n_dissociating_pairs"] == 2


def test_verdict_no_gap_when_output_also_shows():
    # latent shows but so does its matched output -> not a dissociation
    strong = [{"jlens_ratio": 20.0, "random_max_ratio": 1.0}] * 3
    per_task = {
        "fresh1hop-operand": strong, "fresh1hop-answer": strong,
        "fresh2hop-bridge": strong, "fresh2hop-answer": strong,
    }
    v = diagnostic_verdict(per_task, RATIO)
    assert v["verdict"] == "NO-GAP"
    assert v["n_dissociating_pairs"] == 0


def test_amended_q5_passes_on_two_adequate_anchors():
    # two adequate-N latent anchors clear; one low-N clears but is descriptive
    v = amended_q5_verdict(
        {"capital-operand": (True, 33), "fresh1hop-operand": (True, 28),
         "fresh2hop-bridge": (True, 6)}, adequate_n=20)
    assert v["passed"] is True
    assert v["clearing_adequate_n"] == ["capital-operand", "fresh1hop-operand"]
    assert v["descriptive_low_n"] == {"fresh2hop-bridge": 6}


def test_amended_q5_fails_with_only_one_adequate_anchor():
    v = amended_q5_verdict(
        {"capital-operand": (True, 33), "fresh1hop-operand": (False, 28),
         "fresh2hop-bridge": (True, 6)}, adequate_n=20)
    assert v["passed"] is False  # only 1 adequate-N anchor clears; low-N doesn't count
    assert v["n_clearing"] == 1


def test_verdict_no_gap_when_only_one_pair_dissociates():
    strong = [{"jlens_ratio": 20.0, "random_max_ratio": 1.0}] * 3
    flat = [{"jlens_ratio": 1.1, "random_max_ratio": 1.0}] * 3
    per_task = {
        "fresh1hop-operand": strong, "fresh1hop-answer": flat,   # dissociates
        "fresh2hop-bridge": flat, "fresh2hop-answer": flat,       # latent doesn't show
    }
    v = diagnostic_verdict(per_task, RATIO)
    assert v["verdict"] == "NO-GAP"  # needs >= 2
    assert v["n_dissociating_pairs"] == 1
