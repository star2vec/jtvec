"""EXP-M4-emergence classification logic: onset detection and the per-scale
developmental verdict (dissociation / co-emergence / inconclusive)."""

from __future__ import annotations

import math

import pytest

from jtvec.emergence import (
    CheckpointRecord,
    EmergenceRule,
    onset_step_frac_plateau,
    onset_step_gate,
    scale_interaction,
)


def _rec(step, acc, gate, conv=0, ov=None):
    return CheckpointRecord(step=step, exec_acc=acc, gate_passed=gate,
                            converged_tasks=conv, outvocab_rank=ov)


def test_onset_frac_plateau_picks_earliest_reaching_fraction():
    recs = [_rec(1, 0.1, False), _rec(10, 0.5, False), _rec(100, 0.9, True),
            _rec(1000, 1.0, True)]
    # max 1.0, frac 0.8 -> target 0.8 -> first at step 100
    assert onset_step_frac_plateau(recs, lambda r: r.exec_acc, frac=0.8) == 100


def test_onset_frac_plateau_none_when_all_zero():
    recs = [_rec(1, 0.0, False), _rec(10, 0.0, False), _rec(100, 0.0, False)]
    assert onset_step_frac_plateau(recs, lambda r: r.exec_acc, frac=0.8) is None


def test_onset_gate_first_pass():
    recs = [_rec(1, 0.9, False), _rec(10, 0.9, False), _rec(100, 0.9, True),
            _rec(1000, 0.9, True)]
    assert onset_step_gate(recs) == 100
    assert onset_step_gate([_rec(1, 0.9, False), _rec(10, 0.9, False),
                            _rec(100, 0.9, False)]) is None


def test_dissociation_when_execution_precedes_fv_stability():
    # execution plateaus at step 10; FV stabilizes only at step 4000.
    recs = [_rec(1, 0.1, False), _rec(10, 0.95, False), _rec(100, 0.95, False),
            _rec(1000, 0.95, False), _rec(4000, 0.95, True), _rec(16000, 0.95, True)]
    v = EmergenceRule().classify_scale(recs)
    assert v["verdict"] == "DISSOCIATION"
    assert v["exec_onset_step"] == 10 and v["fv_stability_onset_step"] == 4000
    assert v["log10_gap"] == pytest.approx(math.log10(4000) - math.log10(10))


def test_co_emergence_when_onsets_coincide():
    # both cross around the same step; gap below the log threshold.
    recs = [_rec(1, 0.1, False), _rec(100, 0.95, True), _rec(200, 0.95, True),
            _rec(1000, 0.95, True)]
    v = EmergenceRule().classify_scale(recs)
    assert v["verdict"] == "CO-EMERGENCE"
    assert abs(v["log10_gap"]) < EmergenceRule().min_log10_gap


def test_dissociation_fv_never_when_gate_never_passes_but_executes():
    # >=2 gate passes required, but here 0 passes AND execution matures:
    # with <2 passes the rule reports INCONCLUSIVE-FV (can't assess), not
    # DISSOCIATION-FV-NEVER. DISSOCIATION-FV-NEVER needs the gate-passes bar met
    # elsewhere yet no onset -- unreachable, so INCONCLUSIVE guards it.
    recs = [_rec(1, 0.1, False), _rec(10, 0.95, False), _rec(100, 0.95, False)]
    v = EmergenceRule().classify_scale(recs)
    assert v["verdict"] == "INCONCLUSIVE-FV" and v["n_gate_passes"] == 0


def test_inconclusive_when_fewer_than_two_gate_passes():
    recs = [_rec(1, 0.1, False), _rec(10, 0.95, False), _rec(100, 0.95, True)]
    v = EmergenceRule().classify_scale(recs)  # only 1 gate pass
    assert v["verdict"] == "INCONCLUSIVE-FV"


def test_no_execution_when_task_never_learned():
    recs = [_rec(1, 0.0, False), _rec(10, 0.0, True), _rec(100, 0.0, True)]
    v = EmergenceRule().classify_scale(recs)
    assert v["verdict"] == "NO-EXECUTION"


def test_min_gate_passes_configurable():
    recs = [_rec(1, 0.1, False), _rec(10, 0.95, False), _rec(100, 0.95, True)]
    v = EmergenceRule(min_gate_passes=1).classify_scale(recs)
    assert v["verdict"] == "DISSOCIATION"  # 1 pass now suffices; gap 100 vs 10


def test_scale_interaction_monotonic_gap():
    per_scale = {
        "410M": {"log10_gap": 0.6},
        "1B": {"log10_gap": 1.0},
        "2.8B": {"log10_gap": 1.4},
    }
    si = scale_interaction(per_scale)
    assert si["n_scales_with_gap"] == 3 and si["gap_monotonic_nondecreasing"] is True

    per_scale2 = {"410M": {"log10_gap": 1.4}, "1B": {"log10_gap": 0.6}}
    assert scale_interaction(per_scale2)["gap_monotonic_nondecreasing"] is False

    per_scale3 = {"410M": {"log10_gap": None}, "1B": {"log10_gap": 0.6}}
    assert scale_interaction(per_scale3)["gap_monotonic_nondecreasing"] is None
