"""Model-free landing tests for the EXP-M5-2 operator-gate logic.

The pure functions (prefix_match / faithfulness / top1_agreement /
output_state_cosine / require_controlled / converged / derange_objects /
fixed_split) are validated on synthetic inputs — no model loaded (the vendored
`src` estimator is imported only inside the model-side functions). The real
wrapper-faithfulness landing check runs in `probe` before any gate number.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("m5_2", REPO_ROOT / "scripts" / "m5_2_operator_gate.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

BARS = {"agreement": 0.90, "output_cosine": 0.95, "faithfulness_pos": 0.60, "shuffled_neg": 0.10,
        "n_relations_certify": 6}


def test_prefix_match():
    assert m.prefix_match("Rome", "Rome")
    assert m.prefix_match(" Ro", "Rome")       # pred prefix of obj
    assert m.prefix_match("Rome", "Ro")        # obj prefix of pred
    assert not m.prefix_match("Paris", "Rome")
    assert not m.prefix_match("", "Rome")


def test_faithfulness():
    assert m.faithfulness(["Rome", "Paris", "x"], ["Rome", "Paris", "London"]) == 2 / 3
    assert m.faithfulness([], []) == 0.0


def test_top1_agreement():
    rows = [["Rome", "Rome", "Rome"], ["Paris", "Paris", "London"], ["a", "a", "a"]]
    assert m.top1_agreement(rows) == 2 / 3          # 2 of 3 subjects fully agree
    assert m.top1_agreement([["X", " x", "x "]]) == 1.0  # normalized equal


def test_output_state_cosine():
    v = torch.tensor([1.0, 0.0, 0.0])
    w = torch.tensor([0.0, 1.0, 0.0])
    assert abs(m.output_state_cosine([[v, v, v]]) - 1.0) < 1e-6      # identical -> 1
    assert abs(m.output_state_cosine([[v, w]]) - 0.0) < 1e-6         # orthogonal -> 0


def test_require_controlled():
    ok, rep = m.require_controlled([0.7, 0.65, 0.72], [0.0, 0.05, 0.02], BARS)
    assert ok and rep["positive_ok"] and rep["negative_ok"]
    ok2, rep2 = m.require_controlled([0.4, 0.45, 0.5], [0.0, 0.0, 0.0], BARS)   # pos too low
    assert not ok2 and not rep2["positive_ok"]
    ok3, rep3 = m.require_controlled([0.7, 0.7, 0.7], [0.3, 0.4, 0.5], BARS)    # shuffled faithful
    assert not ok3 and not rep3["negative_ok"]


def test_converged():
    assert m.converged(0.95, 0.97, True, BARS)
    assert not m.converged(0.85, 0.99, True, BARS)   # agreement below bar
    assert not m.converged(0.95, 0.90, True, BARS)   # cosine below bar
    assert not m.converged(0.95, 0.97, False, BARS)  # controls failed


def test_derange_objects_no_fixed_point():
    pairs = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    out = m.derange_objects(pairs, seed=7)
    assert [s for s, _ in out] == ["a", "b", "c", "d"]          # subjects preserved order
    assert all(out[i][1] != pairs[i][1] for i in range(len(pairs)))  # every object moved


def test_fixed_split_deterministic_disjoint():
    samples = [(f"s{i}", f"o{i}") for i in range(20)]
    probe1, pool1 = m.fixed_split(samples, n_probe=8, k_estimate=6)
    probe2, pool2 = m.fixed_split(samples, n_probe=8, k_estimate=6)
    assert probe1 == probe2 and pool1 == pool2                  # deterministic (seed 0)
    assert len(probe1) == 8
    assert set(probe1).isdisjoint(set(pool1))                   # held-out disjoint from pool
    assert len(pool1) > 6                                        # pool exceeds k_estimate (draws differ)
    # small relation: probe capped so the estimation pool stays > k_estimate
    probe3, pool3 = m.fixed_split(samples[:14], n_probe=30, k_estimate=6)
    assert len(pool3) > 6 and len(probe3) == 4
