"""Landing test for the EXP-M5-0 1.4B lens-gate verdict logic (model-free).

evaluate_gate is a pure function over per-draw eval payloads; these tests feed
it synthetic probe.json/swap.json-shaped dicts and assert the Q1-Q6 verdict and
each failure mode, so the gate decision is validated without loading a model
(mirrors tests/test_m2_stability.py's model-free discipline).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# scripts/ is not a package; load the module by path.
_spec = importlib.util.spec_from_file_location(
    "m5_lens_gate", REPO_ROOT / "scripts" / "m5_lens_gate.py"
)
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)

BAND = range(4, 17)


def _per_layer(hmr: float) -> dict:
    # a flat HMR profile across L0..22 with a dip to `hmr` inside the band
    layers = {str(l): {"hmr": 40.0, "pass@10": 0.0} for l in range(23)}
    layers["10"] = {"hmr": hmr, "pass@10": 1.0}
    return layers


def _probe_task(jlens_hmr: float, logit_hmr: float, random_hmr: float = 40.0) -> dict:
    metrics = {
        "jlens": {"per_layer": _per_layer(jlens_hmr)},
        "logit": {"per_layer": _per_layer(logit_hmr)},
    }
    for s in range(10):
        metrics[f"random-{s}"] = {"per_layer": _per_layer(random_hmr)}
    return {"metrics": metrics, "per_item": []}


def _swap(dp: float, flip: float, sham: float, n: int = 16) -> dict:
    return {
        "metrics": {
            "n_scored": n,
            "mean_dp_swap_answer": dp,
            "mean_dp_answer": dp,
            "mean_dp_swap_answer_random_ctrl": sham,
            "swap_top1_rate": flip,
        },
        "per_item": [],
    }


def _draw(verdict_pass=True, dp=0.60, flip=0.88, sham=0.01,
          jlens_hmr=2.5, logit_hmr=60.0, random_hmr=40.0) -> dict:
    tasks = ["capital-recall", "capital-operand", "opposites", "word-pairs"]
    probe = {t: _probe_task(jlens_hmr, logit_hmr, random_hmr) for t in tasks}
    return {"verdict_pass": verdict_pass, "swap": _swap(dp, flip, sham), "probe": probe}


def _draws(**kw) -> dict:
    return {k: _draw(**kw) for k in (0, 1, 2)}


def test_all_pass():
    verdict, ok = m5.evaluate_gate(_draws(), BAND)
    assert ok is True
    assert all(verdict["rules"].values())


def test_q1_fails_when_a_draw_gate_fails():
    draws = _draws()
    draws[2]["verdict_pass"] = False
    verdict, ok = m5.evaluate_gate(draws, BAND)
    assert ok is False
    assert verdict["rules"]["Q1_all_draws_gate_pass"] is False


def test_q2_fails_on_weak_swap():
    verdict, ok = m5.evaluate_gate(_draws(dp=0.20), BAND)
    assert verdict["rules"]["Q2_positive_control"] is False
    assert ok is False


def test_q2_fails_on_low_flip():
    verdict, _ = m5.evaluate_gate(_draws(flip=0.50), BAND)
    assert verdict["rules"]["Q2_positive_control"] is False


def test_q3_bound_is_quant_aware():
    # n=16 -> bound = max(0.03, 1/16=0.0625) = 0.0625; sham 0.05 passes
    verdict, _ = m5.evaluate_gate(_draws(sham=0.05), BAND)
    assert verdict["rules"]["Q3_sham"] is True
    assert verdict["diagnostics"]["sham_bound"] == 0.0625
    # sham 0.07 > bound -> fails
    verdict2, _ = m5.evaluate_gate(_draws(sham=0.07), BAND)
    assert verdict2["rules"]["Q3_sham"] is False


def test_q4_fails_when_random_arm_beats_bandmin():
    # random arm HMR 2.0 <= jlens band-min 2.5 -> negative-control breach
    verdict, ok = m5.evaluate_gate(_draws(random_hmr=2.0), BAND)
    assert verdict["rules"]["Q4_negative_control"] is False
    assert ok is False


def test_q5_fails_without_logit_contrast():
    # jlens 2.5 but logit only 10.0 < 5x -> no task passes the contrast
    verdict, _ = m5.evaluate_gate(_draws(logit_hmr=10.0), BAND)
    assert verdict["rules"]["Q5_probing_contrast"] is False


def test_q5_needs_two_tasks():
    draws = _draws()
    # break the contrast on all but one task
    for k in (0, 1, 2):
        for t in ["capital-operand", "opposites", "word-pairs"]:
            draws[k]["probe"][t] = _probe_task(2.5, 10.0)
    verdict, _ = m5.evaluate_gate(draws, BAND)
    assert verdict["diagnostics"]["q5_passing_tasks"] == ["capital-recall"]
    assert verdict["rules"]["Q5_probing_contrast"] is False


def test_q6_fails_on_unstable_dp():
    draws = {0: _draw(dp=0.30), 1: _draw(dp=0.60), 2: _draw(dp=0.90)}
    verdict, ok = m5.evaluate_gate(draws, BAND)
    assert verdict["rules"]["Q6_draw_stability"] is False
    assert ok is False


def test_q6_fails_on_unstable_hmr():
    draws = {
        0: _draw(jlens_hmr=1.0), 1: _draw(jlens_hmr=2.5), 2: _draw(jlens_hmr=4.0),
    }
    verdict, _ = m5.evaluate_gate(draws, BAND)
    # dp stable but band-min HMR IQR large -> Q6 fails
    assert verdict["rules"]["Q6_draw_stability"] is False
