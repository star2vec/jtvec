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


def answer_first_tokens(tokenizer, answer: str, *, case_sensitive: bool = True) -> set[int]:
    """First-token ids of an answer's canonical completion form (D-012).

    Exact-match scoring for execution controls: the leading-space form plus
    the bare form, in the answer's OWN case only (case_sensitive=True). This
    is deliberately narrower than jvec.evals.tasks.surface_token_ids, whose
    case-expansion made an uppercase capitalize output (" K" for "Kettle")
    collide with a plural target's capitalized variant (" Kettles") in the
    cross-task swap, and whose leading-space relaxation let a jspace-broken
    "Te" still count as a hit for " Tehran". add_special_tokens=False because
    the jlens tokenizer prepends BOS by default.
    """
    forms = [f" {answer}", answer]
    if not case_sensitive:
        forms += [f" {answer.capitalize()}", f" {answer.lower()}",
                  answer.capitalize(), answer.lower()]
    ids = set()
    for form in forms:
        toks = tokenizer(form, add_special_tokens=False).input_ids
        if toks:
            ids.add(toks[0])
    return ids


def random_word_null_context(bos: str, input_words, pool, rng) -> str:
    """Report-probe negative-control context with random-word outputs (D-012).

    Replaces the v1 shuffled-context baseline, which did not null a task whose
    outputs are all one morphological class (shuffling plurals among plurals
    still exemplifies "plural"; observed report-probe null = 1.0 on
    singular-plural). Here each input is paired with a word drawn from `pool`
    (the union of the OTHER tasks' outputs — no coherent class), so no task
    label is systematically exemplified.
    """
    outs = [pool[int(i)] for i in rng.integers(0, len(pool), size=len(input_words))]
    return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in zip(input_words, outs))


def execution_answer(item: dict) -> str:
    """The clean-execution answer token of a lens task item, across schemas.

    completion tasks (capital-recall) carry ``target``; swap tasks
    (swap-capitals) carry ``answer`` (the clean answer; swap_* fields are for
    the M1 causal-swap eval, not execution). Raises if neither is present so a
    schema drift fails loudly rather than scoring against nothing.
    """
    for key in ("target", "answer"):
        if item.get(key):
            return item[key]
    raise KeyError(
        f"task item '{item.get('name', '?')}' has neither 'target' nor 'answer'"
    )


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
    Negative: random-word-null accuracy sits within a quantized margin of the
    label prior (1/n_candidates). (D-012: the null is the random-word context,
    not the v1 shuffled context.)"""

    min_detection: float
    max_null_above_prior: float

    def verdict(
        self, *, detection_by_phrasing: dict[str, float], null_acc: float,
        prior: float, n: int,
    ) -> dict:
        margin = quantized_bound(self.max_null_above_prior, n)
        best = max(detection_by_phrasing.values())
        return {
            "detection_by_phrasing": dict(detection_by_phrasing),
            "best_detection": best,
            "null_acc": null_acc,
            "prior": prior,
            "n": n,
            "null_margin": margin,
            "positive_pass": best >= self.min_detection,
            "negative_pass": (null_acc - prior) <= margin,
        }


@dataclass(frozen=True)
class SwapControlRule:
    """Positive: either swap kind lifts the task-B-correct rate over clean by
    >= min_b_gain. Negative (D-012, one-sided): the norm-matched random-target
    arm does not ELEVATE the task-B rate over clean by more than a quantized
    bound. A random swap that destroys computation (drives B to 0) is expected
    per CONSTRAINTS and must not count as a control failure — only a random
    swap that itself produces the task-B effect does."""

    min_b_gain: float
    max_random_elevation: float

    def verdict(self, *, b_rates: dict[str, float], n: int) -> dict:
        bound = quantized_bound(self.max_random_elevation, n)
        gain = max(
            b_rates["direct_swap"] - b_rates["none"],
            b_rates["lens_swap"] - b_rates["none"],
        )
        return {
            "b_rates": dict(b_rates),
            "n": n,
            "best_gain": gain,
            "random_bound": bound,
            "random_elevation": b_rates["random_target"] - b_rates["none"],
            "positive_pass": gain >= self.min_b_gain,
            "negative_pass": (b_rates["random_target"] - b_rates["none"]) <= bound,
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
