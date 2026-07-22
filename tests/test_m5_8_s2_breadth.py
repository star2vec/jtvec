"""Model-free unit tests for the EXP-M5-8 S2 breadth verdict logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("m5_8", REPO_ROOT / "scripts" / "m5_8_s2_breadth.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

BARS = {"stability": 0.95, "lens_dark": 20, "potency_delta": 0.15, "n_certify": 2}


def test_draw_unstable():
    assert m.draw_unstable(0.50, 0.95)          # in the v1 0.43-0.61 range -> unstable
    assert not m.draw_unstable(0.997, 0.95)     # the ACTUAL certified fv_todd -> STABLE


def test_lens_dark():
    assert m.lens_dark(50, 20)                  # label unreadable via jlens
    assert not m.lens_dark(5, 20)


def test_cleared():
    assert m.cleared(0.70, 0.05, 0.15)
    assert not m.cleared(0.10, 0.05, 0.15)      # under sham+delta
    assert not m.cleared(0.19, 0.05, 0.15)      # 0.14 < 0.15


def test_potent_either_arm():
    assert m.potent(0.80, 0.02, 0.05, 0.02, 0.15)["potent"]    # injection fires
    assert m.potent(0.05, 0.02, 0.60, 0.03, 0.15)["potent"]    # ablation fires
    assert not m.potent(0.05, 0.02, 0.05, 0.03, 0.15)["potent"]  # neither
    p = m.potent(0.80, 0.02, 0.05, 0.02, 0.15)
    assert p["injection_potent"] and not p["ablation_potent"]


def test_s2_match_full_profile():
    # draw-unstable + lens-dark + potent -> match
    pot = m.potent(0.9, 0.0, 0.0, 0.0, 0.15)
    assert m.s2_match(0.50, 40, pot, BARS)["match"]
    # the ACTUAL certified case: draw-STABLE (0.997) breaks the draw-unstable leg
    assert not m.s2_match(0.997, 40, pot, BARS)["match"]
    # lens-readable (rank 3) breaks the lens-dark leg
    assert not m.s2_match(0.50, 3, pot, BARS)["match"]
    # inert breaks potent
    inert = m.potent(0.0, 0.0, 0.0, 0.0, 0.15)
    assert not m.s2_match(0.50, 40, inert, BARS)["match"]


def test_roster():
    assert m.roster_verdict([True, True, False], 2) == "PROFILE-REPRODUCES"
    assert m.roster_verdict([True, False, False], 2) == "HETEROGENEOUS"
    assert m.roster_verdict([False, False, False], 2) == "HETEROGENEOUS"
