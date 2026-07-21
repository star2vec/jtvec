"""Landing tests for EXP-M5-1c null-check pure helpers (jtvec.concept_gate:
scrambled_labels prefix-stability + group_by_label) and the scramble transform."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jtvec.concept_gate import group_by_label, scrambled_labels


def test_scrambled_labels_prefix_stable():
    """First T of an n>=T scramble equal an n=T scramble at the same seed
    (per-index rng), so a ladder rung is a prefix slice."""
    full = scrambled_labels(64, 8, seed=3)
    for t in (8, 16, 32):
        assert scrambled_labels(t, 8, seed=3) == full[:t]


def test_scrambled_labels_range_and_seed_variation():
    a = scrambled_labels(40, 8, seed=1)
    b = scrambled_labels(40, 8, seed=2)
    assert all(0 <= x < 8 for x in a)
    assert a != b


def test_group_by_label_splits_pos_neg():
    states = {4: torch.arange(10.0).reshape(5, 2)}   # 5 contexts, d_model=2
    labels = [0, 1, 0, 2, 0]
    pos, neg = group_by_label(states, labels, target=0)
    assert pos[4].shape == (3, 2) and neg[4].shape == (2, 2)
    # rows 0,2,4 are the target
    assert torch.equal(pos[4], states[4][[0, 2, 4]])
    assert torch.equal(neg[4], states[4][[1, 3]])


def test_group_by_label_none_when_side_empty():
    states = {4: torch.zeros(3, 2)}
    assert group_by_label(states, [0, 0, 0], target=0) is None   # no neg
    assert group_by_label(states, [1, 1, 1], target=0) is None   # no pos


def test_scramble_intermediates_permutes_and_avoids_fixed_points():
    from jvec.evals.tasks import Task
    from scripts.m5_1c_null_check import scramble_intermediates
    items = [{"name": f"i{i}", "prompt": f"p{i}", "intermediates": [f"w{i}"]}
             for i in range(6)]
    task = Task("t", "completion", items)
    scr = scramble_intermediates(task, seed=0)
    # every scrambled intermediate is a real one, and the multiset is preserved
    orig = sorted(it["intermediates"][0] for it in items)
    got = sorted(it["intermediates"][0] for it in scr.items)
    assert got == orig
    # no item probes its own original intermediate (derangement where possible)
    assert all(scr.items[i]["intermediates"][0] != items[i]["intermediates"][0]
               for i in range(6))
