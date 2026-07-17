import pytest

from jtvec.core.draws import DrawCountError, DrawIndependenceError, DrawSet, MIN_DRAWS


def test_single_draw_is_unrepresentable():
    with pytest.raises(DrawCountError):
        DrawSet(values=(0.5,), seeds=(0,))


def test_two_draws_rejected():
    with pytest.raises(DrawCountError):
        DrawSet(values=(0.5, 0.6), seeds=(0, 1))


def test_min_draws_is_three():
    assert MIN_DRAWS == 3
    ds = DrawSet(values=(0.5, 0.6, 0.7), seeds=(0, 1, 2))
    assert ds.n == 3


def test_duplicate_seeds_rejected():
    with pytest.raises(DrawIndependenceError):
        DrawSet(values=(0.5, 0.6, 0.7), seeds=(0, 0, 1))


def test_seed_count_must_match_value_count():
    with pytest.raises(DrawIndependenceError):
        DrawSet(values=(0.5, 0.6, 0.7), seeds=(0, 1))


def test_median_and_iqr():
    ds = DrawSet(values=(1.0, 2.0, 3.0, 4.0), seeds=(0, 1, 2, 3))
    assert ds.median == 2.5
    assert ds.iqr == pytest.approx(1.5)
    assert "median=2.5" in ds.summary()
    assert "n_draws=4" in ds.summary()
