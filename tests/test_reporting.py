import pytest

from jtvec.core.draws import DrawSet
from jtvec.core.reporting import scoped, scoped_intervention

EFFECT = DrawSet(values=(0.60, 0.62, 0.58), seeds=(0, 1, 2))
SHAM = DrawSet(values=(0.01, 0.00, 0.02), seeds=(3, 4, 5))


def test_scope_arguments_are_mandatory():
    with pytest.raises(TypeError):
        scoped("dp", 0.6)  # no model/config/n -> no string


def test_scoped_renders_full_scope():
    s = scoped("dp(swap)", EFFECT, model="pythia-410m", config="skip4_n10", n=16)
    assert s.startswith("on pythia-410m, skip4_n10, N=16:")
    assert "median=0.6" in s


def test_intervention_string_always_contains_sham():
    s = scoped_intervention(
        "dp(swap)", EFFECT, SHAM, model="pythia-410m", config="skip4_n10", n=16
    )
    assert "(sham:" in s


def test_intervention_sham_is_mandatory():
    with pytest.raises(TypeError):
        scoped_intervention("dp(swap)", EFFECT, model="pythia-410m", config="c", n=16)
