"""Model-free unit tests for the EXP-M5-2b draw-marginalized / cosine criterion.

The pure functions (majority_vote / marginalized_faithfulness /
per_draw_faithfulness / certify / require_controlled) are validated on synthetic
inputs — no model loaded (the estimator/model path lives in the reused m5_2
module's runtime functions).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("m5_2b", REPO_ROOT / "scripts" / "m5_2b_operator_criterion.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

BARS = {"output_cosine": 0.95, "marginalized_faith": 0.60, "negative": 0.10, "n_relations_certify": 6}


def test_majority_vote():
    assert m.majority_vote(["Beijing", "Beijing", "Delhi"]) == "beijing"   # 2 vs 1
    assert m.majority_vote([" Paris", "paris", "Paris "]) == "paris"       # normalized unanimous
    assert m.majority_vote(["a", "b", "c"]) == "a"                         # 3-way tie -> earliest draw


def test_marginalized_faithfulness():
    # subject1: majority Beijing (matches); subject2: majority 'x' (no match)
    top1 = [["Beijing", "Beijing", "Delhi"], ["x", "x", "Lima"]]
    assert m.marginalized_faithfulness(top1, ["Beijing", "Lima"]) == 0.5
    # marginalization rescues a churny-but-majority-correct subject
    assert m.marginalized_faithfulness([["Rome", "Rome", "\n"]], ["Rome"]) == 1.0
    assert m.marginalized_faithfulness([], []) == 0.0


def test_per_draw_faithfulness_shows_churn():
    # each draw individually 0.5, but majority vote (below) would be 1.0 -> shows churn absorbed
    pd = m.per_draw_faithfulness([["Rome", "Paris", "Rome"], ["Paris", "Rome", "Paris"]], ["Rome", "Paris"])
    assert pd == [1.0, 0.0, 1.0]


def test_certify_all_three_gates():
    assert m.certify(0.96, 0.70, 0.05, BARS)             # all pass
    assert not m.certify(0.94, 0.70, 0.05, BARS)         # cosine under
    assert not m.certify(0.96, 0.50, 0.05, BARS)         # marg-faith under
    assert not m.certify(0.96, 0.70, 0.20, BARS)         # negative too faithful


def test_require_controlled():
    ok, rep = m.require_controlled(0.70, 0.05, BARS)
    assert ok and rep["positive_ok"] and rep["negative_ok"]
    ok2, rep2 = m.require_controlled(0.50, 0.05, BARS)   # positive fails
    assert not ok2 and not rep2["positive_ok"]
    ok3, rep3 = m.require_controlled(0.70, 0.30, BARS)   # unrelated donor too faithful
    assert not ok3 and not rep3["negative_ok"]


def test_close_but_under_is_fail():
    # the "no sixth criterion" spirit: 0.949 cosine is a FAIL, not a near-pass
    assert not m.certify(0.949, 0.99, 0.0, BARS)
