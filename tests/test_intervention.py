import pytest

from jtvec.core.draws import DrawSet
from jtvec.core.intervention import (
    InterventionResult,
    InterventionSpec,
    ShamMismatchError,
    ShamResult,
)

EFFECT = DrawSet(values=(0.60, 0.62, 0.58), seeds=(0, 1, 2))
SHAM_EFFECT = DrawSet(values=(0.01, 0.00, 0.02), seeds=(3, 4, 5))


def spec(**overrides):
    fields = dict(kind="swap", layers=(8, 9), positions=(-1,), n_directions=1, norm=12.5)
    fields.update(overrides)
    return InterventionSpec(**fields)


def test_result_without_sham_is_a_type_error():
    with pytest.raises(TypeError):
        InterventionResult(spec=spec(), effect=EFFECT)


def test_sham_must_match_layers():
    sham = ShamResult(spec=spec(layers=(1, 2)), effect=SHAM_EFFECT)
    with pytest.raises(ShamMismatchError):
        InterventionResult(spec=spec(), effect=EFFECT, sham=sham)


def test_sham_must_match_positions():
    sham = ShamResult(spec=spec(positions=(0,)), effect=SHAM_EFFECT)
    with pytest.raises(ShamMismatchError):
        InterventionResult(spec=spec(), effect=EFFECT, sham=sham)


def test_sham_must_match_norm_within_tolerance():
    sham = ShamResult(spec=spec(norm=14.0), effect=SHAM_EFFECT)
    with pytest.raises(ShamMismatchError):
        InterventionResult(spec=spec(), effect=EFFECT, sham=sham)


def test_matched_sham_admitted_and_rendered_in_same_row():
    sham = ShamResult(spec=spec(norm=12.55), effect=SHAM_EFFECT)
    result = InterventionResult(spec=spec(), effect=EFFECT, sham=sham)
    row = result.table_row()
    assert "sham:" in row
    assert "median=0.6" in row
