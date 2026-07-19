"""EXP-M4-E2 dissociation decision-rule contracts: the cross-draw effect
DrawSet, the hurts/transfers predicates, and the double/one-way/none verdict
classification."""

from __future__ import annotations

import pytest

from jtvec.core.draws import DrawCountError, DrawSet
from jtvec.e2_dissociation import DissociationRule, effect_drawset


def test_effect_drawset_signs_and_seeds():
    ds = effect_drawset(0.9, {1: 0.0, 2: 0.1, 3: 0.05})
    assert ds.seeds == (1, 2, 3)
    # clean - ablated: bigger drop = bigger positive effect
    assert ds.values == (0.9, 0.8, 0.85)
    assert ds.median == pytest.approx(0.85)


def test_effect_drawset_requires_three_draws():
    with pytest.raises(DrawCountError):
        effect_drawset(0.9, {1: 0.0, 2: 0.1})


def _flat(v):
    # a 3-draw DrawSet with all-equal values (median=v, iqr=0)
    return DrawSet(values=(v, v, v), seeds=(1, 2, 3))


def test_double_dissociation_when_each_ablation_hits_its_own_measure():
    rule = DissociationRule()
    v = rule.verdict(
        fv_exec=_flat(0.90), fv_exec_sham=_flat(0.00),      # fv cuts exec
        fv_report=_flat(0.02), fv_report_sham=_flat(0.00),  # fv spares report
        jspace_exec=_flat(0.03), jspace_exec_sham=_flat(0.00),   # jspace spares exec
        jspace_report=_flat(0.30), jspace_report_sham=_flat(0.00),  # jspace cuts report
    )
    assert v["verdict"] == "DOUBLE-DISSOCIATION"
    assert v["direction1_fv_exec_not_report"] and v["direction2_jspace_report_not_exec"]
    assert v["cross_draw_transfer"]["fv_exec"] and v["cross_draw_transfer"]["jspace_report"]


def test_one_way_when_only_fv_direction_holds():
    rule = DissociationRule()
    v = rule.verdict(
        fv_exec=_flat(0.90), fv_exec_sham=_flat(0.00),
        fv_report=_flat(0.02), fv_report_sham=_flat(0.00),
        jspace_exec=_flat(0.03), jspace_exec_sham=_flat(0.00),
        jspace_report=_flat(0.04), jspace_report_sham=_flat(0.00),  # jspace does NOT cut report
    )
    assert v["verdict"] == "ONE-WAY"
    assert v["direction1_fv_exec_not_report"] and not v["direction2_jspace_report_not_exec"]


def test_no_dissociation_when_ablations_are_nonspecific():
    rule = DissociationRule()
    # fv cuts BOTH measures -> not a clean execution-only effect -> dir1 false.
    v = rule.verdict(
        fv_exec=_flat(0.90), fv_exec_sham=_flat(0.00),
        fv_report=_flat(0.30), fv_report_sham=_flat(0.00),  # fv also cuts report
        jspace_exec=_flat(0.03), jspace_exec_sham=_flat(0.00),
        jspace_report=_flat(0.04), jspace_report_sham=_flat(0.00),
    )
    assert not v["direction1_fv_exec_not_report"]
    assert v["verdict"] == "NO-DISSOCIATION"


def test_sham_subtracted_effect_must_clear_delta():
    rule = DissociationRule(min_exec_drop=0.15, min_report_drop=0.10)
    # exec effect 0.20 but sham also 0.12 -> net 0.08 < 0.15 -> does not hurt.
    assert not rule.hurts(_flat(0.20), _flat(0.12), 0.15)
    assert rule.hurts(_flat(0.30), _flat(0.12), 0.15)  # net 0.18 >= 0.15


def test_transfer_flag_false_when_a_draw_straddles_delta():
    rule = DissociationRule(min_report_drop=0.10)
    sham = _flat(0.0)
    # median 0.12 clears 0.10, but one draw (0.05) does not -> not transferring.
    effect = DrawSet(values=(0.20, 0.12, 0.05), seeds=(1, 2, 3))
    assert rule.hurts(effect, sham, 0.10)            # median 0.12 >= 0.10
    assert not rule.transfers(effect, sham, 0.10)    # draw 3 (0.05) fails
