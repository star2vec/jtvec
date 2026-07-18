"""M3: intervention-instrument gate — model-free logic and control criteria.

M1 gated the lens readout, M2 gated the FV estimator; M3 gates the
intervention instruments the M4 experiments consume: FV-direction ablation,
J-space ablation, the forced-choice report probe, and the FV swap (FV
injection was gated at M2). Deliverable: a passing positive+negative
ControlRecord pair per instrument (jtvec.core.instruments), after a lens
re-materialization verified against M1's committed manifest.

Vendored intervention code (jvec.evals.exp3, jvec.evals.fvswap,
jvec.evals.swap) is called as a library and never modified. Scope ruled by
D-011. Control bounds are quantization-aware from the start (D-010 lesson):
every sham-deviation bound is max(base, 1/N).
"""

from __future__ import annotations

from dataclasses import dataclass

from jvec.lens_cache import IDENTITY_KEYS
from jtvec.fv_stability import StabilityGatedFV, load_certificate

M3_INSTRUMENTS = (
    "fv-direction-ablation",
    "jspace-ablation",
    "report-probe-forced-choice",
    "fv-swap",
)


def verify_lens_manifest(refit: dict, reference: dict) -> dict[str, tuple]:
    """Identity mismatches between a refit lens manifest and M1's committed
    one; empty dict = same instrument identity. IDENTITY_KEYS is the vendored
    lens cache's own definition, so device/wall-clock/provenance fields are
    excluded by construction (the MPS->CUDA move changes numerics, which the
    functional spot-check covers; identity covers the inputs)."""
    return {
        k: (refit.get(k), reference.get(k))
        for k in IDENTITY_KEYS
        if refit.get(k) != reference.get(k)
    }


def load_certified_fv(
    cfg, task: str, revision: str, certificates: dict, draw_k: int = 1
) -> StabilityGatedFV:
    """A StabilityGatedFV from M2's certificates plus the draw-k cache.

    M2 certified the estimator per task; the concrete artifact M3/M4 use is
    draw draw_k's full-trial FV (n_trials_aie=200 >= converged_at). Manifest
    identity is enforced by the vendored loader; a missing cache fails loudly.
    """
    import dataclasses as _dc  # noqa: PLC0415
    from jvec.fv import load_cached_fv  # noqa: PLC0415

    if task not in certificates:
        raise KeyError(f"no M2 certificate for task '{task}'")
    cert = load_certificate(certificates[task])
    dcfg = _dc.replace(cfg, cache_dir=f"cache/m2/draw{draw_k}")
    cached = load_cached_fv(dcfg, task, revision)
    if cached is None:
        raise FileNotFoundError(
            f"no cached FV for '{task}' under cache/m2/draw{draw_k}; "
            "re-run scripts/m2_gate.py (extractions cache-hit)"
        )
    return StabilityGatedFV(
        cert,
        cached["fv_todd"].float(),
        task=task,
        draw_seed=cfg.seed * 1000 + draw_k,
    )


def explicit_rule_context(bos: str, label_word: str, shuffled_pairs) -> str:
    """Positive-control context for the report probe (D-011).

    The task label is present ONLY as an explicit rule statement; the Q/A
    pairs are label-shuffled so they carry no coherent task. The probe must
    read the stated label. This is a detection-ceiling control for the
    readout — deliberately not a statement about what ICL contexts carry.
    """
    rule = (
        f"Rule: in the list below, every answer is the question word's "
        f"{label_word}.\n\n"
    )
    body = "".join(f"Q: {x}\nA: {y}\n\n" for x, y in shuffled_pairs)
    return bos + rule + body


def quantized_bound(base: float, n: int) -> float:
    """A deviation bound that can never sit below one readout quantum (D-010)."""
    if n <= 0:
        raise ValueError(f"n={n}; need a positive cell size")
    return max(base, 1.0 / n)


@dataclass(frozen=True)
class AblationControlRule:
    """Positive: ablation drops execution by >= min_exec_drop.
    Negative: the matched sham stays within a quantized bound of clean."""

    min_exec_drop: float
    max_sham_dev: float

    def verdict(self, *, none_acc: float, ablated_acc: float, sham_acc: float, n: int) -> dict:
        bound = quantized_bound(self.max_sham_dev, n)
        return {
            "none": none_acc,
            "ablated": ablated_acc,
            "sham": sham_acc,
            "n": n,
            "sham_bound": bound,
            "positive_pass": (none_acc - ablated_acc) >= self.min_exec_drop,
            "negative_pass": abs(none_acc - sham_acc) <= bound,
        }


@dataclass(frozen=True)
class ReportProbeControlRule:
    """Positive: best phrasing reads the explicitly stated label >= min_detection.
    Negative: shuffled-context accuracy sits within a quantized margin of the
    label prior (1/n_candidates)."""

    min_detection: float
    max_shuffled_above_prior: float

    def verdict(
        self, *, detection_by_phrasing: dict[str, float], shuffled_acc: float,
        prior: float, n: int,
    ) -> dict:
        margin = quantized_bound(self.max_shuffled_above_prior, n)
        best = max(detection_by_phrasing.values())
        return {
            "detection_by_phrasing": dict(detection_by_phrasing),
            "best_detection": best,
            "shuffled_acc": shuffled_acc,
            "prior": prior,
            "n": n,
            "shuffled_margin": margin,
            "positive_pass": best >= self.min_detection,
            "negative_pass": (shuffled_acc - prior) <= margin,
        }


@dataclass(frozen=True)
class SwapControlRule:
    """Positive: either swap kind lifts the task-B-correct rate over clean by
    >= min_b_gain. Negative: the norm-matched random-target arm stays within a
    quantized bound of clean."""

    min_b_gain: float
    max_random_dev: float

    def verdict(self, *, b_rates: dict[str, float], n: int) -> dict:
        bound = quantized_bound(self.max_random_dev, n)
        gain = max(
            b_rates["direct_swap"] - b_rates["none"],
            b_rates["lens_swap"] - b_rates["none"],
        )
        return {
            "b_rates": dict(b_rates),
            "n": n,
            "best_gain": gain,
            "random_bound": bound,
            "positive_pass": gain >= self.min_b_gain,
            "negative_pass": abs(b_rates["random_target"] - b_rates["none"]) <= bound,
        }


def shared_query_map(ds_a, ds_b) -> dict[str, tuple[str, str]]:
    """Queries valid under both tasks with distinct targets: q -> (y_a, y_b).

    Same purpose as the io_map/shared logic in vendored scripts/12_fv_swap.py:
    the swap control needs trials where "followed task B" is well defined.
    """

    def io_map(ds):
        mapping: dict[str, str] = {}
        for split in ("train", "valid", "test"):
            data = ds[split][:]
            for x, y in zip(data["input"], data["output"]):
                mapping[str(x)] = str(y)
        return mapping

    a, b = io_map(ds_a), io_map(ds_b)
    return {q: (a[q], b[q]) for q in sorted(set(a) & set(b)) if a[q] != b[q]}
