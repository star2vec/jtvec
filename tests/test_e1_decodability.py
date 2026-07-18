"""EXP-M4-E1 logic contracts: registered label sets equal the vendored v1
definitions, the rank statistic and its scale invariance, the readout
control rules (D-014 proportional inclusion), and the C1-C4 decision rule."""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import pytest
import torch

from jtvec.e1_decodability import (
    E1_TASKS,
    LABEL_SETS,
    RANDOM_SEEDS,
    E1DecisionRule,
    ReadoutNegativeRule,
    ReadoutPositiveRule,
    full_vocab_rank,
    label_rank,
    layer_set_stability,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class StubTokenizer:
    """Maps exact strings to token-id lists, mimicking the vendored
    tokenizer(form, add_special_tokens=False).input_ids call shape."""

    def __init__(self, vocab: dict[str, list[int]]):
        self.vocab = vocab

    def __call__(self, text: str, add_special_tokens: bool = True):
        class _Enc:
            def __init__(self, ids):
                self.input_ids = ids

        return _Enc(self.vocab.get(text, [0]))


def test_label_sets_equal_vendored_v1_definitions():
    spec = importlib.util.spec_from_file_location(
        "harden_exp1", REPO_ROOT / "scripts" / "14_harden_exp1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["harden_exp1"] = mod
    spec.loader.exec_module(mod)
    for set_name in ("set1", "set2", "set3"):
        for task in E1_TASKS:
            assert LABEL_SETS[set_name][task] == list(mod.LABEL_SETS[set_name][task]), (
                f"{set_name}/{task} diverges from the vendored registration"
            )


def test_random_seeds_are_the_registered_hundred():
    assert RANDOM_SEEDS == tuple(range(1000, 1100))


def test_full_vocab_rank():
    logits = torch.tensor([0.1, 3.0, 2.0, 2.0, -1.0])
    assert full_vocab_rank(logits, 1) == 1
    assert full_vocab_rank(logits, 0) == 4
    assert full_vocab_rank(logits, 4) == 5
    # ties: rank counts only strictly-greater entries
    assert full_vocab_rank(logits, 2) == 2
    assert full_vocab_rank(logits, 3) == 2


def test_label_rank_min_over_layers_words_tokens():
    tok = StubTokenizer({" plural": [2], "plural": [3], " Plural": [2]})
    readouts = {
        4: torch.tensor([5.0, 4.0, 3.0, 2.0]),  # id 2 rank 3
        5: torch.tensor([0.0, 1.0, 9.0, 2.0]),  # id 2 rank 1
    }
    assert label_rank(readouts, tok, ["plural"]) == 1
    with pytest.raises(ValueError):
        label_rank({}, tok, ["plural"])


def test_rank_statistic_is_scale_invariant():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, generator=g)
    for tid in (0, 17, 63):
        assert full_vocab_rank(logits, tid) == full_vocab_rank(logits * 7.3, tid)


def test_positive_rule_primary_matches_ratified_10_of_13():
    rule = ReadoutPositiveRule()
    ranks = {l: 1 for l in range(4, 17)}  # 13 layers, all pass
    res = rule.evaluate(ranks)
    assert res["needed"] == 10 and res["passed"]
    ranks_9 = {l: (1 if l < 13 else 999) for l in range(4, 17)}  # 9 pass
    res9 = rule.evaluate(ranks_9)
    assert len(res9["included_layers"]) == 9 and not res9["passed"]
    ranks_10 = {l: (1 if l < 14 else 999) for l in range(4, 17)}  # 10 pass
    assert rule.evaluate(ranks_10)["passed"]


def test_positive_rule_single_layer_variant():
    # D-014: skip16_n10's band overlap is one layer; ceil(0.75*1)=1.
    rule = ReadoutPositiveRule()
    assert rule.evaluate({16: 3})["passed"]
    assert not rule.evaluate({16: 11})["passed"]
    with pytest.raises(ValueError):
        rule.evaluate({})


def test_negative_rule_median_floor():
    rule = ReadoutNegativeRule()
    assert rule.evaluate([100] * 100)["passed"]
    assert not rule.evaluate([99] * 100)["passed"]
    mixed = [5] * 49 + [5000] * 51
    assert rule.evaluate(mixed)["median_rank"] == statistics.median(mixed)


def test_layer_set_stability():
    base = list(range(4, 17))
    ok = layer_set_stability({0: base, 1: base[:-1], 2: base[1:]})
    assert ok["passed"] and ok["max_symmetric_diff"] == 2
    bad = layer_set_stability({0: base, 1: base, 2: base[:-4]})
    assert not bad["passed"] and bad["max_symmetric_diff"] == 4


def _grids(
    *, jlens_rank=3, logit_rank=5000, beaten=100, ordering_pairs=None
):
    primary = {(i, j): jlens_rank for i in (1, 2, 3) for j in (0, 1, 2)}
    logit = {i: logit_rank for i in (1, 2, 3)}
    if ordering_pairs is None:
        ordering_pairs = {f"cell{k}": (jlens_rank, logit_rank) for k in range(36)}
    random_beaten = {i: beaten for i in (1, 2, 3)}
    return primary, logit, ordering_pairs, random_beaten


def test_decision_rule_supports():
    primary, logit, ordering, beaten = _grids()
    v = E1DecisionRule().verdict(
        primary_jlens=primary, logit_by_fv_draw=logit,
        ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
    )
    assert v["verdict"] == "DECODABLE-AND-SEPARATED"
    assert v["c1_decodable"] and v["c2_ordering"] and v["c3_logit_floor"] and v["c4_random_anchor"]


def test_decision_rule_not_decodable_on_c1_or_c4():
    primary, logit, ordering, beaten = _grids(jlens_rank=21)
    ordering = {k: (21, 5000) for k in ordering}
    v = E1DecisionRule().verdict(
        primary_jlens=primary, logit_by_fv_draw=logit,
        ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
    )
    assert v["verdict"] == "NOT-DECODABLE" and not v["c1_decodable"]

    primary, logit, ordering, beaten = _grids(beaten=94)
    v = E1DecisionRule().verdict(
        primary_jlens=primary, logit_by_fv_draw=logit,
        ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
    )
    assert v["verdict"] == "NOT-DECODABLE" and not v["c4_random_anchor"]


def test_decision_rule_not_separated_on_c2_or_c3():
    primary, logit, ordering, beaten = _grids(logit_rank=199)
    ordering = {k: (3, 199) for k in ordering}
    v = E1DecisionRule().verdict(
        primary_jlens=primary, logit_by_fv_draw=logit,
        ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
    )
    assert v["verdict"] == "NOT-SEPARATED" and not v["c3_logit_floor"]

    primary, logit, ordering, beaten = _grids()
    ordering["cell0"] = (10, 9)  # one cell breaks ordering
    v = E1DecisionRule().verdict(
        primary_jlens=primary, logit_by_fv_draw=logit,
        ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
    )
    assert v["verdict"] == "NOT-SEPARATED" and not v["c2_ordering"]


def test_decision_rule_refuses_partial_grids():
    primary, logit, ordering, beaten = _grids()
    del primary[(1, 0)]
    with pytest.raises(ValueError):
        E1DecisionRule().verdict(
            primary_jlens=primary, logit_by_fv_draw=logit,
            ordering_cells=ordering, random_beaten_by_fv_draw=beaten,
        )
