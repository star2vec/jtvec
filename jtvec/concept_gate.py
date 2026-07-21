"""EXP-M5-1: S1 concept-direction stability gate — pure logic + decision rule.

The S1 extractor (M5_SPEC §M5.1) is a residual-space mean-difference direction
over certified-task answer states, at the final position of the band layers:

    d(concept c) = mean(resid_final | answer == c) - mean(resid_final | answer != c)

computed per band layer, then (a) concatenated across band layers and unit-
normalised as the *identity* direction the convergence ladder tracks by cosine,
and (b) unit-scaled per layer to the natural mean answer-state norm as the
*injection* delta whose downstream Δp(c) potency the ladder tracks by IQR.

This module holds only the model-free logic: mean-difference assembly, the
context-stream / ladder prefix, the cross-draw cosine + effect-IQR statistics,
the preregistered convergence verdict (witness-rung semantics shared with
jtvec.fv_stability), the positive/negative controls with the quantisation-aware
bound, and the certificate payload. Model forward/hook work lives in
jvec.evals.concept; orchestration in scripts/m5_1_concept_gate.py.

Design note (flagged, D-015 text-only precedent): the prereg's "M1 swap
machinery (truncated pinv rcond 0.05)" clause is vacuous for a residual-space
mean-difference direction — no lens-coordinate write-back is needed to inject a
vector that already lives in the residual stream. Injection is therefore
norm-preserving residual activation-addition (jvec.evals.concept), reusing the
M1 norm-preservation idiom. No threshold or decision-rule change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jtvec.core.draws import DrawSet

# --- roster (EXP-M5-1 "Extractor under the gate"; ratified as drafted) -------

#: Capital family: >= 8 capitals, contexts from the certified capital-recall
#: templates. (country, capital) — capital is the answer token whose Δp the
#: readout tracks.
CAPITAL_ROSTER: tuple[tuple[str, str], ...] = (
    ("France", "Paris"), ("England", "London"), ("Italy", "Rome"),
    ("Germany", "Berlin"), ("Spain", "Madrid"), ("Austria", "Vienna"),
    ("Greece", "Athens"), ("Egypt", "Cairo"),
)

#: Few-shot pool for the capital-recall carrier prefixes (superset of the
#: roster; matches scripts/make_tasks.py CAPITALS). Shots for a target never
#: share the target's country or capital surface string.
CAPITAL_POOL: tuple[tuple[str, str], ...] = CAPITAL_ROSTER + (
    ("Russia", "Moscow"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("Poland", "Warsaw"), ("Ireland", "Dublin"), ("Portugal", "Lisbon"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Cuba", "Havana"),
    ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Finland", "Helsinki"),
)

CAP_TEMPLATE = "The capital of {} is {}. "
CAP_QUERY = "The capital of {} is"

#: Ladder rungs T (n_contexts per class). A pass at max(RUNGS) alone is not
#: convergence — the larger rungs are the stability witness.
RUNGS: tuple[int, ...] = (8, 16, 32, 64)


def capital_context_stream(target: tuple[str, str], seed: int, n: int,
                           *, n_shots: int = 3) -> list[str]:
    """Ordered stream of ``n`` capital-recall carrier prompts whose answer is
    ``target``'s capital, prefixes drawn under ``seed``.

    The stream is a prefix-stable function of (target, seed): the first T of an
    n>=T stream equal an n=T stream at the same seed (RNG-prefix property, so a
    ladder rung is a prefix slice — pinned in tests). Achieved by drawing each
    prompt's shots from an independent per-index rng, so the total count never
    perturbs an earlier index.
    """
    country, capital = target
    pool = [p for p in CAPITAL_POOL if not ({country, capital} & set(p))]
    prompts = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])  # per-index stream -> prefix-stable
        idx = rng.choice(len(pool), n_shots, replace=False)
        prefix = "".join(CAP_TEMPLATE.format(*pool[j]) for j in idx)
        prompts.append(prefix + CAP_QUERY.format(country))
    return prompts


def rung_prefix(stream: list, t: int) -> list:
    """First ``t`` contexts — the ladder rung as a prefix of the draw's stream."""
    if t > len(stream):
        raise ValueError(f"rung {t} exceeds {len(stream)} available contexts")
    return stream[:t]


# --- direction assembly ------------------------------------------------------


def mean_difference_by_layer(pos_by_layer: dict[int, torch.Tensor],
                             neg_by_layer: dict[int, torch.Tensor]
                             ) -> dict[int, torch.Tensor]:
    """Raw per-layer mean-difference d_l = mean(pos_l) - mean(neg_l).

    pos_by_layer/neg_by_layer: {layer: [n, d_model]} residual answer states.
    """
    if set(pos_by_layer) != set(neg_by_layer):
        raise ValueError("pos/neg layer sets differ")
    return {l: pos_by_layer[l].float().mean(0) - neg_by_layer[l].float().mean(0)
            for l in pos_by_layer}


def identity_direction(raw_by_layer: dict[int, torch.Tensor],
                       band_layers: list[int]) -> torch.Tensor:
    """Unit direction the ladder tracks by cosine: the per-layer raw
    mean-differences concatenated over band layers, then unit-normalised.

    A fully degenerate direction (concatenated norm ~ 0, i.e. the extractor
    found no separation between the answer classes) raises rather than emitting
    NaN — a genuine extraction failure to surface, not silently pass on.
    """
    d = torch.cat([raw_by_layer[l].flatten().float() for l in band_layers])
    norm = float(d.norm())
    if norm < 1e-8:
        raise ValueError("degenerate concept direction: concatenated mean-"
                         "difference norm ~ 0 (answer classes not separated)")
    return d / norm


def injection_deltas(raw_by_layer: dict[int, torch.Tensor],
                     natural_norm_by_layer: dict[int, float]) -> dict[int, torch.Tensor]:
    """Per-layer injection delta: unit(d_l) scaled to the layer's natural mean
    answer-state norm (the "natural norm" the prereg injects at). A layer whose
    mean-difference is ~ 0 contributes no injection (zeros) rather than NaN."""
    out = {}
    for l, d in raw_by_layer.items():
        d = d.float()
        norm = float(d.norm())
        out[l] = (d / norm * float(natural_norm_by_layer[l])) if norm >= 1e-8 else torch.zeros_like(d)
    return out


def natural_norms(states_by_layer: dict[int, torch.Tensor]) -> dict[int, float]:
    """Mean L2 norm of the answer states per layer (the injection scale)."""
    return {l: float(s.float().norm(dim=-1).mean()) for l, s in states_by_layer.items()}


def min_pairwise_cosine(units: list[torch.Tensor]) -> float:
    """Min cosine over the unordered pairs of unit directions (>= 3 draws)."""
    n = len(units)
    if n < 2:
        raise ValueError("need >= 2 directions for a pairwise cosine")
    return min(
        float(torch.dot(units[i].flatten().float(), units[j].flatten().float()))
        for i in range(n) for j in range(i + 1, n)
    )


# --- convergence rule (shared witness semantics with jtvec.fv_stability) ------


@dataclass(frozen=True)
class ConceptRungStats:
    """Cross-draw agreement at one rung (one concept)."""

    n_contexts: int
    min_pairwise_cosine: float
    effect_iqr: float  # IQR over draws of the downstream Δp(concept answer)


@dataclass(frozen=True)
class ConceptConvergenceRule:
    """Preregistered thresholds (EXP-M5-1 Decision rule; M5_SPEC values)."""

    min_pairwise_cosine: float  # 0.95
    max_effect_iqr: float       # 0.05

    def passes(self, s: ConceptRungStats) -> bool:
        return (s.min_pairwise_cosine >= self.min_pairwise_cosine
                and s.effect_iqr <= self.max_effect_iqr)


def convergence_verdict(per_rung: list[ConceptRungStats],
                        rule: ConceptConvergenceRule) -> dict:
    """converged_at = smallest passing rung with every larger rung also passing.

    A pass at the largest rung alone has no witness (``passes_at_max_rung_only``):
    preregistered as "extend the ladder", not convergence.
    """
    ordered = sorted(per_rung, key=lambda s: s.n_contexts)
    passes = [rule.passes(s) for s in ordered]
    converged_at = None
    for i in range(len(ordered) - 1):  # last rung alone cannot converge
        if passes[i] and all(passes[i:]):
            converged_at = ordered[i].n_contexts
            break
    return {
        "converged": converged_at is not None,
        "converged_at": converged_at,
        "passes_at_max_rung_only": bool(passes and passes[-1] and converged_at is None),
        "per_rung": {
            s.n_contexts: {"pass": ok, "min_pairwise_cosine": s.min_pairwise_cosine,
                           "effect_iqr": s.effect_iqr}
            for s, ok in zip(ordered, passes)
        },
    }


# --- instrument controls -----------------------------------------------------


@dataclass(frozen=True)
class ControlOutcome:
    passed: bool
    detail: str


def positive_control(effect: DrawSet, sham: DrawSet, min_gain: float) -> ControlOutcome:
    """The direction moves its own readout: Δp(concept answer) median >=
    +min_gain above the sham median (EXP-M5-1 Instruments)."""
    gain = effect.median - sham.median
    return ControlOutcome(
        gain >= min_gain,
        f"Δp(answer) median {effect.median:+.4f} - sham median {sham.median:+.4f} "
        f"= {gain:+.4f} vs bar +{min_gain}",
    )


def negative_control(random_dps: list[float], n_eval: int, base: float = 0.02
                     ) -> ControlOutcome:
    """Norm-matched random directions barely move the readout: max |Δp| <=
    max(base, 1/N) — quantisation-aware bound from the start (D-010 lesson)."""
    bound = max(base, 1.0 / n_eval)
    worst = max(abs(x) for x in random_dps)
    return ControlOutcome(
        worst <= bound,
        f"max |Δp| {worst:.4f} vs bound {bound:.4f} = max({base}, 1/{n_eval})",
    )


def certificate_payload(*, concept: str, model: str, converged_at: int,
                        n_draws: int, evidence_run: str, issued: str) -> dict:
    """JSON-serialisable S1 certificate inputs, concept-scoped in the estimator
    string (``s1_concept@Paris``) so a certificate can never be presented for a
    concept whose convergence study it does not cover."""
    return {
        "estimator": f"s1_concept@{concept}",
        "model": model,
        "converged_at": converged_at,
        "n_draws": n_draws,
        "evidence_run": evidence_run,
        "issued": issued,
    }
