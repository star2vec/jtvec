"""EXP-M4-E3 swap decision-rule contracts: redirection vs the random control,
cross-draw transfer, and the J-specific vs basis-agnostic classification."""

from __future__ import annotations

import pytest

from jtvec.core.draws import DrawCountError
from jtvec.e3_swap import SwapRedirectionRule, rate_drawset


def test_rate_drawset_requires_three():
    with pytest.raises(DrawCountError):
        rate_drawset({1: 0.9, 2: 0.9})
    ds = rate_drawset({1: 0.9, 2: 0.8, 3: 0.85})
    assert ds.median == pytest.approx(0.85) and ds.seeds == (1, 2, 3)


def test_redirects_basis_agnostic_when_lens_and_direct_both_work():
    # M3-shaped: none 0.0, lens ~0.9, direct ~0.93, random 0.0.
    v = SwapRedirectionRule().verdict(
        none_b=0.0,
        lens_b={1: 0.90, 2: 0.90, 3: 0.87},
        direct_b={1: 0.93, 2: 0.93, 3: 0.90},
        random_b={1: 0.0, 2: 0.0, 3: 0.0},
    )
    assert v["redirects"] and v["cross_draw_transfer"]
    assert not v["j_specific"]                      # direct as good as lens
    assert v["verdict"] == "REDIRECTS-BASIS-AGNOSTIC"


def test_redirects_j_specific_when_lens_beats_direct():
    v = SwapRedirectionRule().verdict(
        none_b=0.0,
        lens_b={1: 0.90, 2: 0.88, 3: 0.92},
        direct_b={1: 0.10, 2: 0.05, 3: 0.12},       # direct fails, lens works
        random_b={1: 0.0, 2: 0.0, 3: 0.0},
    )
    assert v["redirects"] and v["j_specific"]
    assert v["verdict"] == "REDIRECTS-J-SPECIFIC"


def test_no_redirection_when_random_also_elevates_b():
    # If the random target elevates B as much as the swap, redirection is not
    # established (one-sided control fires).
    v = SwapRedirectionRule().verdict(
        none_b=0.10,
        lens_b={1: 0.60, 2: 0.60, 3: 0.60},
        direct_b={1: 0.60, 2: 0.60, 3: 0.60},
        random_b={1: 0.58, 2: 0.60, 3: 0.62},       # random ~= swap
    )
    assert not v["redirects"] and v["verdict"] == "NO-REDIRECTION"


def test_no_redirection_when_gain_below_delta():
    v = SwapRedirectionRule(min_b_gain=0.20).verdict(
        none_b=0.10,
        lens_b={1: 0.20, 2: 0.22, 3: 0.18},         # gain ~0.10 < 0.20
        direct_b={1: 0.19, 2: 0.21, 3: 0.20},
        random_b={1: 0.10, 2: 0.10, 3: 0.10},
    )
    assert not v["redirects"] and v["verdict"] == "NO-REDIRECTION"


def test_transfer_flag_false_when_one_draw_lags():
    v = SwapRedirectionRule(min_b_gain=0.20).verdict(
        none_b=0.0,
        lens_b={1: 0.90, 2: 0.90, 3: 0.10},         # draw 3 lags
        direct_b={1: 0.90, 2: 0.90, 3: 0.12},
        random_b={1: 0.0, 2: 0.0, 3: 0.0},
    )
    # median gain still >= 0.20 (redirects), but draw 3 does not transfer
    assert v["redirects"] and not v["cross_draw_transfer"]


def test_mismatched_draws_raise():
    with pytest.raises(ValueError):
        SwapRedirectionRule().verdict(
            none_b=0.0, lens_b={1: 0.9, 2: 0.9, 3: 0.9},
            direct_b={1: 0.9, 2: 0.9, 4: 0.9}, random_b={1: 0.0, 2: 0.0, 3: 0.0},
        )
