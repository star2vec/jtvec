"""Model-free landing test for EXP-M5-0 qualification helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jtvec.qualification import admit, build_lre_prompt, em_hit


def test_build_lre_prompt():
    p = build_lre_prompt("The capital of {} is",
                         [("France", "Paris"), ("Italy", "Rome")], "Peru")
    assert p == "The capital of France is Paris\nThe capital of Italy is Rome\nThe capital of Peru is"


def test_em_hit_relaxed_prefix():
    assert em_hit(" Paris, the capital", "Paris") is True
    assert em_hit(" paris", "Paris") is True            # case-relaxed
    assert em_hit(" Lima", "Paris") is False
    assert em_hit(" Washington D.C. is", "Washington D.C.") is True  # multi-token object


def test_admit():
    a = admit({"r1": 0.7, "r2": 0.55, "r3": 0.6}, 0.6)
    assert a == {"r1": True, "r2": False, "r3": True}
