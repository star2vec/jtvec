"""M4-E1: FV label decodability — model-free statistics and decision rule.

E1 decodes M2-certified Todd FVs through the M1-gated J-lens transport
(unembed(J_l v), the jlens arm) and the logit lens (unembed(v)) and scores
the task-label full-vocab rank. This module holds the registered label sets
(v1's, restricted to the certified tasks), the rank statistics, the
readout's in-run instrument-control rules, and the preregistered per-task
decision rule C1-C4 (EXP-M4-E1; constants ratified D-014). Nothing here
touches a model; the orchestrator is scripts/m4_e1_gate.py.

Scale note pinned by tests: the rank statistic is invariant to positive
rescaling of the decoded vector (layer norm and comparisons both preserve
per-vector order), so norm-matching a random vector changes no rank.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch

from jvec.evals.fvprobe import TASK_LABEL_WORDS
from jvec.evals.tasks import surface_token_ids

E1_TASKS = ("capitalize", "singular-plural", "english-french")

#: Registered label sets — v1's (scripts/14_harden_exp1.py LABEL_SETS),
#: restricted to the certified tasks. Equality with the vendored definitions
#: is pinned by tests/test_e1_decodability.py; fixed before scoring.
LABEL_SETS: dict[str, dict[str, list[str]]] = {
    "set1": {t: list(TASK_LABEL_WORDS[t]) for t in E1_TASKS},
    "set2": {
        "capitalize": ["uppercase"],
        "singular-plural": ["plural"],
        "english-french": ["French"],
    },
    "set3": {
        "capitalize": sorted(set(TASK_LABEL_WORDS["capitalize"]) | {"capitalized", "caps"}),
        "singular-plural": sorted(set(TASK_LABEL_WORDS["singular-plural"]) | {"many", "pluralize"}),
        "english-french": sorted(set(TASK_LABEL_WORDS["english-french"]) | {"language", "foreign"}),
    },
}

#: Random-vector seeds (preregistered; v1 convention seed+1000).
RANDOM_SEEDS = tuple(range(1000, 1100))


def full_vocab_rank(logits: torch.Tensor, token_id: int) -> int:
    """Rank of token_id over the full vocab; 1 = top."""
    return int(1 + (logits > logits[token_id]).sum())


def label_rank(readouts: dict[int, torch.Tensor], tokenizer, words) -> int:
    """min over layers, label words, and surface tokens of full-vocab rank.

    The v1 Exp-1 statistic (scripts/14_harden_exp1.label_stat): surface
    tokens use the vendored case+space expansion — for a rank readout any
    surface variant of the label counts as verbalizing it (there is no
    cross-task collision to guard against here, unlike the D-012 execution
    scoring).
    """
    best: int | None = None
    for logits in readouts.values():
        for word in words:
            for tid in surface_token_ids(tokenizer, word):
                r = full_vocab_rank(logits, tid)
                best = r if best is None or r < best else best
    if best is None:
        raise ValueError("no readouts or no label words; statistic undefined")
    return best


def canonical_label_token(tokenizer, word: str) -> int:
    """The word's primary surface token id (vendored convention)."""
    return surface_token_ids(tokenizer, word)[0]


@dataclass(frozen=True)
class ReadoutPositiveRule:
    """Round-trip detection ceiling per lens instance (D-014).

    For each layer l of the instance's band-overlap layer set, the
    constructed label vector pinv(J_l) @ W_U[label] must decode its own
    label token at rank <= max_rank through unembed(J_l .). The control
    passes iff >= ceil(min_pass_fraction * n_layers) layers pass — for the
    primary 13-layer instances that is the ratified 10/13; for a robustness
    variant the same fraction applies to its own (possibly smaller) set.
    The passing layers become the instance's included layer set L.
    """

    max_rank: int = 10
    min_pass_fraction: float = 0.75

    def evaluate(self, per_layer_rank: dict[int, int]) -> dict:
        if not per_layer_rank:
            raise ValueError("no layers; positive control undefined")
        included = sorted(l for l, r in per_layer_rank.items() if r <= self.max_rank)
        needed = math.ceil(self.min_pass_fraction * len(per_layer_rank))
        return {
            "per_layer_rank": {int(l): int(r) for l, r in per_layer_rank.items()},
            "included_layers": included,
            "needed": needed,
            "n_layers": len(per_layer_rank),
            "passed": len(included) >= needed,
        }


@dataclass(frozen=True)
class ReadoutNegativeRule:
    """No label from noise: median random-vector label rank >= min_median."""

    min_median: float = 100.0

    def evaluate(self, random_ranks: list[int]) -> dict:
        if not random_ranks:
            raise ValueError("no random readings; negative control undefined")
        med = statistics.median(random_ranks)
        return {
            "median_rank": med,
            "n": len(random_ranks),
            "passed": med >= self.min_median,
        }


def layer_set_stability(
    included_by_draw: dict[int, list[int]], max_symmetric_diff: int = 3
) -> dict:
    """Included layer sets across the primary lens draws must agree.

    Pairwise symmetric difference > max_symmetric_diff on any pair means the
    readout is unstable: INSTRUMENT-VOID for all tasks (prereg).
    """
    draws = sorted(included_by_draw)
    worst = 0
    for a in draws:
        for b in draws:
            if a < b:
                diff = len(set(included_by_draw[a]) ^ set(included_by_draw[b]))
                worst = max(worst, diff)
    return {
        "included_by_draw": {int(k): list(v) for k, v in included_by_draw.items()},
        "max_symmetric_diff": worst,
        "passed": worst <= max_symmetric_diff,
    }


@dataclass(frozen=True)
class E1DecisionRule:
    """Preregistered per-task criteria C1-C4 (EXP-M4-E1, ratified D-014)."""

    max_jlens_median: float = 20.0  # C1
    min_logit_median: float = 200.0  # C3
    min_random_beaten: int = 95  # C4, out of n_random
    n_random: int = 100

    def verdict(
        self,
        *,
        primary_jlens: dict[tuple[int, int], int],
        logit_by_fv_draw: dict[int, int],
        ordering_cells: dict[str, tuple[int, int]],
        random_beaten_by_fv_draw: dict[int, int],
        not_evaluable_cells: tuple[str, ...] = (),
    ) -> dict:
        """primary_jlens: {(fv_draw, lens_draw): rank}, the 9 primary cells.
        ordering_cells: {cell_name: (r_jlens, r_logit)} — primary + robustness.
        not_evaluable_cells: robustness cells voided by a variant's failed
        control (recorded, excluded from C2 per the D-014 proportional rule).
        """
        if len(primary_jlens) != 9:
            raise ValueError(f"expected 9 primary cells, got {len(primary_jlens)}")
        if len(logit_by_fv_draw) != 3:
            raise ValueError(f"expected 3 logit draws, got {len(logit_by_fv_draw)}")
        if len(random_beaten_by_fv_draw) != 3:
            raise ValueError(
                f"expected 3 random-anchor draws, got {len(random_beaten_by_fv_draw)}"
            )
        jlens_median = statistics.median(primary_jlens.values())
        logit_median = statistics.median(logit_by_fv_draw.values())
        c1 = jlens_median <= self.max_jlens_median
        c2 = all(rj < rl for rj, rl in ordering_cells.values())
        c3 = logit_median >= self.min_logit_median
        c4 = all(
            beaten >= self.min_random_beaten
            for beaten in random_beaten_by_fv_draw.values()
        )
        if c1 and c2 and c3 and c4:
            verdict = "DECODABLE-AND-SEPARATED"
        elif not (c1 and c4):
            verdict = "NOT-DECODABLE"
        else:
            verdict = "NOT-SEPARATED"
        return {
            "jlens_median": jlens_median,
            "logit_median": logit_median,
            "c1_decodable": c1,
            "c2_ordering": c2,
            "c3_logit_floor": c3,
            "c4_random_anchor": c4,
            "n_ordering_cells": len(ordering_cells),
            "not_evaluable_cells": list(not_evaluable_cells),
            "verdict": verdict,
        }
