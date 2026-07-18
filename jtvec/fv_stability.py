"""M2: FV extraction-stability gate — pure logic and the decision rule.

CONSTRAINTS lists as a known-unknown the AIE trial count at which Todd-style
FV extraction converges on the target model (v1 evidence at 25 trials:
cross-draw cosine 0.43-0.61 on identical weights). This module holds the
model-free logic of the M2 convergence study: rung derivation from stored
per-trial AIE tensors, cross-draw agreement statistics, the preregistered
convergence verdict, sham-twin construction, and the certificate plumbing
that makes a gated FV constructible (jtvec.core.gate).

Model work lives in scripts/m2_gate.py. Vendored extraction code (jvec.fv,
third_party/function_vectors) is called as a library and never modified.
Design reference: design_input/15_fv_stability_v1_untracked.py (v1,
unvalidated, never imported).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jtvec.core.gate import CertifiedArtifact, GateCertificate, UngatedEstimatorError

#: AIE-trial ladder. Extraction runs once per draw at max(RUNGS); every lower
#: rung is a prefix of the stored per-trial tensor. Because the extraction RNG
#: stream advances identically regardless of n_trials, the prefix equals what
#: a shorter run at the same seed would have produced (tests/test_m2_stability
#: pins the RNG-prefix property and the rung recomputation).
RUNGS: tuple[int, ...] = (25, 50, 100, 200)


def rung_slice(indirect_effect: torch.Tensor, n_trials: int) -> torch.Tensor:
    """First ``n_trials`` trials of a stored [n_trials_max, layers, heads] AIE tensor."""
    if indirect_effect.ndim != 3:
        raise ValueError(
            f"expected [n_trials, layers, heads], got shape {tuple(indirect_effect.shape)}"
        )
    stored = indirect_effect.shape[0]
    if not 1 <= n_trials <= stored:
        raise ValueError(f"rung {n_trials} outside stored trial range 1..{stored}")
    return indirect_effect[:n_trials]


def fv_at_rung(
    mean_activations: torch.Tensor,
    indirect_effect: torch.Tensor,
    n_trials: int,
    model,
    model_config: dict,
    n_top_heads: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Todd FV recomputed from the first ``n_trials`` stored AIE trials.

    Calls the vendored compute_function_vector unchanged; only the AIE tensor
    is prefix-sliced. mean_activations comes from the fixed n_trials_mean
    stage and is not laddered.
    """
    import jvec.fv  # noqa: F401, PLC0415  (sys.path setup for the Todd repo)
    from utils.extract_utils import compute_function_vector  # noqa: PLC0415

    fv, top_heads = compute_function_vector(
        mean_activations,
        rung_slice(indirect_effect, n_trials),
        model,
        model_config,
        n_top_heads=n_top_heads,
    )
    return fv.squeeze().float().cpu(), [(int(l), int(h)) for l, h, _ in top_heads]


def pairwise_cosines(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Cosine for every unordered pair, keyed "a|b" (design-ref convention)."""
    names = list(vectors)
    return {
        f"{a}|{b}": float(
            torch.nn.functional.cosine_similarity(
                vectors[a].flatten().float(), vectors[b].flatten().float(), dim=0
            )
        )
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def top_head_overlap(
    heads_a: list[tuple[int, int]], heads_b: list[tuple[int, int]]
) -> int:
    """|intersection| of two top-head lists, compared as (layer, head) sets."""
    return len({(l, h) for l, h in heads_a} & {(l, h) for l, h in heads_b})


def sham_twin(fv: torch.Tensor, seed: int) -> torch.Tensor:
    """Norm-matched random direction: the auto-generated sham for FV injection.

    Matched on norm by construction; the caller injects it at the same layer
    and position as the real FV, satisfying the sham LAW's matching clause.
    """
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(fv.shape, generator=gen, dtype=torch.float32)
    return v * (float(fv.float().norm()) / float(v.norm()))


@dataclass(frozen=True)
class RungStats:
    """Cross-draw agreement at one rung (one task)."""

    n_trials: int
    min_pairwise_cosine: float
    gain_iqr: float
    min_top_head_overlap: int  # descriptive; not part of the decision rule


@dataclass(frozen=True)
class ConvergenceRule:
    """Preregistered thresholds (EXP-M2, Decision rule section)."""

    min_pairwise_cosine: float
    max_gain_iqr: float

    def passes(self, stats: RungStats) -> bool:
        return (
            stats.min_pairwise_cosine >= self.min_pairwise_cosine
            and stats.gain_iqr <= self.max_gain_iqr
        )


def convergence_verdict(per_rung: list[RungStats], rule: ConvergenceRule) -> dict:
    """Apply the preregistered rule to one task's ladder.

    converged_at = the smallest rung that passes with every larger rung also
    passing (the larger rungs are the stability witness). A pass at the
    largest rung alone has no witness: reported as non-convergence with
    ``passes_at_max_rung_only`` set, which preregisters as "extend the ladder"
    rather than "converged".
    """
    ordered = sorted(per_rung, key=lambda s: s.n_trials)
    passes = [rule.passes(s) for s in ordered]
    converged_at = None
    for i in range(len(ordered) - 1):  # the last rung alone cannot converge
        if passes[i] and all(passes[i:]):
            converged_at = ordered[i].n_trials
            break
    return {
        "converged": converged_at is not None,
        "converged_at": converged_at,
        "passes_at_max_rung_only": bool(passes and passes[-1] and converged_at is None),
        "per_rung": {
            s.n_trials: {
                "pass": ok,
                "min_pairwise_cosine": s.min_pairwise_cosine,
                "gain_iqr": s.gain_iqr,
                "min_top_head_overlap": s.min_top_head_overlap,
            }
            for s, ok in zip(ordered, passes)
        },
    }


def certificate_payload(
    *,
    estimator: str,
    task: str,
    model: str,
    converged_at: int,
    n_draws: int,
    evidence_run: str,
    issued: str,
) -> dict:
    """JSON-serializable GateCertificate inputs, task-scoped.

    GateCertificate has no task field, so per-task scope is encoded in the
    estimator string ("fv_todd@capitalize"): a certificate can never be
    presented for a task whose convergence study it does not cover.
    """
    return {
        "estimator": f"{estimator}@{task}",
        "model": model,
        "converged_at": converged_at,
        "n_draws": n_draws,
        "evidence_run": evidence_run,
        "issued": issued,
    }


def load_certificate(payload: dict) -> GateCertificate:
    """Reconstruct (and re-validate) a GateCertificate from its payload."""
    return GateCertificate(**payload)


class StabilityGatedFV(CertifiedArtifact):
    """The M2 FV artifact: a task FV inseparable from its gate certificate."""

    def __init__(
        self,
        certificate: GateCertificate,
        vector: torch.Tensor,
        task: str,
        draw_seed: int,
    ) -> None:
        super().__init__(certificate)
        if not certificate.estimator.endswith(f"@{task}"):
            raise UngatedEstimatorError(
                f"certificate estimator '{certificate.estimator}' does not cover "
                f"task '{task}'; the stability gate is per-task"
            )
        self.vector = vector
        self.task = task
        self.draw_seed = draw_seed
