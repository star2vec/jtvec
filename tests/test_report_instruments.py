"""EXP-M4-E2 report-instrument gate: control-rule logic contracts.

The rule is CI-level and quantization-free; these pin the positive arm
(coherent elevated above the neutral prior), the negative arm (coherent
specific to the coherent mapping vs BOTH the shuffled and other nulls — the
D-013 discriminator), the gated-iff-one-phrasing-does-both semantics, and
bootstrap determinism."""

from __future__ import annotations

import numpy as np
import pytest

from jtvec.report_instruments import (
    ReportScoreControlRule,
    bootstrap_ci,
    instrument_name,
)


def test_instrument_name_is_task_scoped():
    assert instrument_name("singular-plural") == "report-score-prior-corrected@singular-plural"


def test_bootstrap_ci_is_deterministic_and_ordered():
    vals = list(range(40))
    a = bootstrap_ci(vals, seed=0)
    b = bootstrap_ci(vals, seed=0)
    assert a == b
    mean, lo, hi = a
    assert lo < mean < hi
    with pytest.raises(ValueError):
        bootstrap_ci([])


def _rule():
    return ReportScoreControlRule(n_boot=2000, boot_seed=0)


def test_phrasing_gated_when_coherent_elevated_and_specific():
    rng = np.random.default_rng(1)
    coherent = list(rng.normal(3.0, 0.5, 40))   # well above 0
    shuffled = list(rng.normal(0.2, 0.5, 40))   # near prior (mapping destroyed)
    other = list(rng.normal(-0.1, 0.5, 40))     # at/below prior
    res = _rule().evaluate_phrasing(coherent=coherent, shuffled=shuffled, other=other)
    assert res["positive_pass"] and res["negative_pass"] and res["gated"]


def test_negative_fails_when_shuffled_matches_coherent():
    # The D-013 failure mode: the label is read from the inputs, so shuffling
    # the mapping does NOT lower the score. Coherent ~ shuffled -> not specific.
    rng = np.random.default_rng(2)
    coherent = list(rng.normal(3.0, 0.5, 40))
    shuffled = list(rng.normal(3.0, 0.5, 40))   # inputs alone carry the label
    other = list(rng.normal(-0.1, 0.5, 40))
    res = _rule().evaluate_phrasing(coherent=coherent, shuffled=shuffled, other=other)
    assert res["positive_pass"]           # detection still works
    assert not res["negative_pass"]       # but not specific to the mapping
    assert not res["gated"]


def test_positive_fails_when_coherent_not_above_prior():
    rng = np.random.default_rng(3)
    coherent = list(rng.normal(0.0, 0.5, 40))   # CI straddles 0
    shuffled = list(rng.normal(-2.0, 0.5, 40))
    other = list(rng.normal(-2.0, 0.5, 40))
    res = _rule().evaluate_phrasing(coherent=coherent, shuffled=shuffled, other=other)
    assert not res["positive_pass"] and not res["gated"]


def test_negative_requires_specificity_over_both_nulls():
    # Coherent beats shuffled but NOT other (a bare label prior inflates other).
    rng = np.random.default_rng(4)
    coherent = list(rng.normal(2.0, 0.4, 40))
    shuffled = list(rng.normal(-1.0, 0.4, 40))
    other = list(rng.normal(2.0, 0.4, 40))      # other as high as coherent
    res = _rule().evaluate_phrasing(coherent=coherent, shuffled=shuffled, other=other)
    assert not res["negative_pass"] and not res["gated"]


def test_verdict_gated_needs_one_phrasing_doing_both():
    rule = _rule()
    rng = np.random.default_rng(5)
    strong = rng.normal(3.0, 0.5, 40)
    near0 = rng.normal(0.1, 0.5, 40)
    low = rng.normal(-0.2, 0.5, 40)
    # P1: positive only (shuffled as high as coherent). P2: negative-shaped but
    # coherent not above 0. Neither phrasing does both -> not gated, though
    # 'positive' is true somewhere.
    p1 = rule.evaluate_phrasing(coherent=list(strong), shuffled=list(strong), other=list(low))
    p2 = rule.evaluate_phrasing(coherent=list(near0), shuffled=list(low), other=list(low))
    v = rule.verdict({"P1": p1, "P2": p2})
    assert v["positive"] and not v["gated"] and v["best_phrasing"] is None

    # Add a phrasing that does both -> gated, best_phrasing set.
    p3 = rule.evaluate_phrasing(coherent=list(strong), shuffled=list(near0), other=list(low))
    v2 = rule.verdict({"P1": p1, "P2": p2, "P3": p3})
    assert v2["gated"] and v2["best_phrasing"] == "P3" and v2["negative"]


def test_verdict_refuses_empty():
    with pytest.raises(ValueError):
        _rule().verdict({})
