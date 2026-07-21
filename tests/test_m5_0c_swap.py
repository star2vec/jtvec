"""Landing test for EXP-M5-0c swap-decomposition verdict logic (model-free).

item_stats / draw_summary / aggregate_over_draws / require_controlled / decide
are pure functions over logit arrays and per-draw summaries; these tests feed
them synthetic inputs and assert the per-item statistics, the instrument
controls, and each decision branch (H-POTENCY / H-CONFOUND / MIXED /
VOID-CONTROL-FAIL) — no model loaded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "m5_0c_swap", REPO_ROOT / "scripts" / "m5_0c_swap.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


# --- item_stats -------------------------------------------------------------
def test_item_stats_gap_margin_and_flip():
    # vocab 5; answer=1, swap_answer=2. base: answer dominant, runner-up=token2.
    base = [0.0, 5.0, 1.0, 0.0, 0.0]
    swap = [0.0, 3.0, 4.0, 0.0, 0.0]      # swap_answer now argmax -> flip
    sham = [[0.0, 5.0, 1.0, 0.0, 0.0]]    # unmoved
    s = m.item_stats(base, swap, sham, answer_id=1, swap_answer_id=2)
    assert s["gap_base"] == -4.0            # logit(2)-logit(1) = 1-5
    assert s["gap_swap"] == 1.0             # 4-3
    assert s["gap_sham_mean"] == -4.0
    assert s["base_margin"] == 4.0          # 5 - 1 (runner-up token2)
    assert s["runner_up_id"] == 2
    assert s["sham_controlled_shift"] == 5.0  # 1 - (-4)
    assert s["margin_normalized_flip"] is True  # 5 > 4
    assert s["top1_flip"] is True
    assert s["base_top1_correct"] is True
    # per-arm swap-answer logits + target-specific sham push (clarified neg control)
    assert s["logit_swap_answer_base"] == 1.0 and s["logit_swap_answer_swap"] == 4.0
    assert s["sham_swap_answer_push"] == 0.0   # sham unmoved -> no logit target push
    assert s["real_swap_answer_push"] == 3.0
    assert s["dp_swap_answer_sham"] == 0.0     # sham == base -> prob push 0 (neg control)
    assert s["dp_swap_answer_real"] > 0.0      # real swap raises swap-answer prob


def test_item_stats_no_flip_and_margin_not_cleared():
    base = [0.0, 6.0, 2.0, 0.0, 0.0]      # margin 4
    swap = [0.0, 6.0, 5.0, 0.0, 0.0]      # gap_swap = -1, still no flip
    sham = [[0.0, 6.0, 2.0, 0.0, 0.0]]    # gap_sham = -4
    s = m.item_stats(base, swap, sham, answer_id=1, swap_answer_id=2)
    assert s["top1_flip"] is False
    assert s["sham_controlled_shift"] == 3.0        # -1 - (-4)
    assert s["margin_normalized_flip"] is False     # 3 < 4


def test_item_stats_base_incorrect_flagged():
    base = [0.0, 2.0, 9.0, 0.0, 0.0]      # top1 is token2, not the answer(1)
    s = m.item_stats(base, base, [base], answer_id=1, swap_answer_id=2)
    assert s["base_top1_correct"] is False


# --- helpers to build per-draw summaries ------------------------------------
def _draws(scs, sham_dp, flip, mflip, sham_gap=None, sham_logit_push=None):
    # sham_dp = prob-space target push (GATING neg control); sham_gap / sham_logit_push
    # = logit-space readings (informational).
    n = len(scs)
    sham_gap = sham_gap if sham_gap is not None else [0.0] * n
    sham_logit_push = sham_logit_push if sham_logit_push is not None else [0.0] * n
    return [
        {"n": 10, "mean_sham_controlled_shift": a, "median_sham_controlled_shift": a,
         "mean_sham_gap_shift": g, "mean_sham_swap_answer_push": lp, "mean_dp_swap_answer_sham": b,
         "top1_flip_rate": c, "margin_norm_flip_rate": d}
        for a, b, c, d, g, lp in zip(scs, sham_dp, flip, mflip, sham_gap, sham_logit_push)
    ]


# --- controls ---------------------------------------------------------------
def test_controls_pass():
    ref = m.aggregate_over_draws(_draws([0.5, 0.48, 0.52], [0.0, 0.01, -0.01], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]))
    test = m.aggregate_over_draws(_draws([0.2, 0.21, 0.19], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]))
    ok, report = m.require_controlled(ref, test)
    assert ok and report["positive_control_ok"] and report["negative_control_prob_ok"]


def test_positive_control_fail():
    ref = m.aggregate_over_draws(_draws([0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]))
    test = m.aggregate_over_draws(_draws([0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]))
    ok, report = m.require_controlled(ref, test)
    assert not ok and not report["positive_control_ok"]


def test_negative_control_prob_fail():
    # prob-space target push out of band -> gate fails
    ref = m.aggregate_over_draws(_draws([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]))
    test = m.aggregate_over_draws(_draws([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]))
    ok, report = m.require_controlled(ref, test)
    assert not ok and not report["negative_control_prob_ok"]


def test_logit_neg_readings_confound_does_not_gate():
    # logit gap-shift / push large (confounded, active sham), but prob push ~0:
    # gate PASSES (gates on prob-space); logit readings reported as informational.
    ref = m.aggregate_over_draws(_draws([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9],
                                        sham_gap=[3.7, 3.8, 3.6], sham_logit_push=[1.5, 1.6, 1.5]))
    test = m.aggregate_over_draws(_draws([0.2, 0.2, 0.2], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2],
                                         sham_gap=[2.0, 2.1, 1.9], sham_logit_push=[0.9, 1.0, 0.8]))
    ok, report = m.require_controlled(ref, test)
    assert ok is True
    assert report["negative_control_prob_ok"] is True
    assert abs(report["negative_control_logit_gap_shift"]["ref"]) > 0.03


# --- decision branches ------------------------------------------------------
def test_decide_h_potency():
    ref = m.aggregate_over_draws(_draws([0.50, 0.48, 0.52], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]))
    test = m.aggregate_over_draws(_draws([0.20, 0.18, 0.22], [0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]))
    d = m.decide(ref, test)
    assert d["label"] == "H-POTENCY"
    assert d["delta"] >= m.POTENCY_DELTA and d["iqr_non_overlap"] and d["margin_flip_low"]


def test_decide_h_confound():
    # comparable gap-shifts (overlapping IQRs), low raw flip, high margin-norm flip
    ref = m.aggregate_over_draws(_draws([0.50, 0.45, 0.55], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]))
    test = m.aggregate_over_draws(_draws([0.48, 0.44, 0.52], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]))
    d = m.decide(ref, test)
    assert d["label"] == "H-CONFOUND"
    assert not d["iqr_non_overlap"] and d["margin_flip_high"]


def test_decide_mixed():
    # non-overlapping (potency-shaped gap) BUT margin-flip high (not low) -> neither branch
    ref = m.aggregate_over_draws(_draws([0.50, 0.48, 0.52], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]))
    test = m.aggregate_over_draws(_draws([0.20, 0.18, 0.22], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]))
    d = m.decide(ref, test)
    assert d["label"] == "MIXED"
