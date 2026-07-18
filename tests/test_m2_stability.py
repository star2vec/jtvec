"""M2 stability-gate logic: rungs, agreement stats, verdict, certificates.

Model-free except the rung-recomputation check, which drives the vendored
compute_function_vector on the tiny NeoX fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jtvec.core.gate import GateCertificate, UngatedEstimatorError
from jvec.utils import set_seed
from jtvec.fv_stability import (
    RUNGS,
    ConvergenceRule,
    RungStats,
    StabilityGatedFV,
    certificate_payload,
    convergence_verdict,
    fv_at_rung,
    load_certificate,
    pairwise_cosines,
    rung_slice,
    sham_twin,
    top_head_overlap,
)

RULE = ConvergenceRule(min_pairwise_cosine=0.95, max_gain_iqr=0.05)


def _stats(spec: dict[int, bool]) -> list[RungStats]:
    """RungStats that pass/fail RULE per the given {rung: should_pass} spec."""
    return [
        RungStats(
            n_trials=t,
            min_pairwise_cosine=0.99 if ok else 0.50,
            gain_iqr=0.01 if ok else 0.30,
            min_top_head_overlap=10 if ok else 3,
        )
        for t, ok in spec.items()
    ]


# --- rung derivation ---------------------------------------------------------


def test_rung_slice_is_prefix():
    ie = torch.arange(24.0).reshape(6, 2, 2)
    assert torch.equal(rung_slice(ie, 4), ie[:4])


def test_rung_slice_rejects_bad_shapes_and_ranges():
    with pytest.raises(ValueError):
        rung_slice(torch.zeros(6, 2), 2)
    ie = torch.zeros(6, 2, 2)
    with pytest.raises(ValueError):
        rung_slice(ie, 0)
    with pytest.raises(ValueError):
        rung_slice(ie, 7)


def test_rng_prefix_property():
    """The extraction loop's RNG stream is length-independent, so the first T
    stored trials equal a T-trial run at the same seed (the ladder's premise)."""

    def draw(n):
        set_seed(11)
        return [np.random.choice(100, 10, replace=False) for _ in range(n)]

    long, short = draw(8), draw(4)
    for a, b in zip(short, long[:4]):
        assert np.array_equal(a, b)


def test_fv_at_rung_matches_direct_computation(tiny_neox):
    model, _, model_config = tiny_neox
    import jvec.fv  # noqa: F401  (sys.path for the Todd repo)
    from utils.extract_utils import compute_function_vector

    gen = torch.Generator().manual_seed(5)
    L, H, D = model_config["n_layers"], model_config["n_heads"], model_config["resid_dim"]
    mean_acts = torch.randn(L, H, 3, D // H, generator=gen)
    ie = torch.randn(8, L, H, generator=gen)

    fv4, heads4 = fv_at_rung(mean_acts, ie, 4, model, model_config, n_top_heads=3)
    direct, direct_heads = compute_function_vector(
        mean_acts, ie[:4], model, model_config, n_top_heads=3
    )
    assert torch.allclose(fv4, direct.squeeze().float().cpu())
    assert heads4 == [(int(l), int(h)) for l, h, _ in direct_heads]

    fv8, _ = fv_at_rung(mean_acts, ie, 8, model, model_config, n_top_heads=3)
    assert not torch.allclose(fv4, fv8)  # different rungs, different estimates


# --- agreement statistics ----------------------------------------------------


def test_pairwise_cosines_keys_and_values():
    v = torch.tensor([1.0, 0.0])
    w = torch.tensor([0.0, 1.0])
    cos = pairwise_cosines({"a": v, "b": w, "c": v})
    assert set(cos) == {"a|b", "a|c", "b|c"}
    assert cos["a|c"] == pytest.approx(1.0)
    assert cos["a|b"] == pytest.approx(0.0, abs=1e-6)


def test_top_head_overlap_counts_layer_head_pairs():
    a = [(0, 1), (2, 3), (4, 5)]
    b = [(2, 3), (4, 5), (6, 7)]
    assert top_head_overlap(a, b) == 2
    assert top_head_overlap(a, a) == 3


def test_sham_twin_norm_matched_and_seeded():
    fv = torch.randn(64, generator=torch.Generator().manual_seed(1)) * 7
    sham = sham_twin(fv, seed=42)
    assert float(sham.norm()) == pytest.approx(float(fv.norm()), rel=1e-5)
    assert torch.equal(sham, sham_twin(fv, seed=42))  # deterministic
    cos = torch.nn.functional.cosine_similarity(fv, sham, dim=0)
    assert abs(float(cos)) < 0.9  # a twin, not a copy


# --- the preregistered verdict -----------------------------------------------


def test_verdict_converges_at_first_witnessed_rung():
    v = convergence_verdict(_stats({25: False, 50: True, 100: True, 200: True}), RULE)
    assert v["converged"] and v["converged_at"] == 50


def test_verdict_ignores_unwitnessed_early_pass():
    v = convergence_verdict(_stats({25: True, 50: False, 100: True, 200: True}), RULE)
    assert v["converged_at"] == 100


def test_verdict_never_converges():
    v = convergence_verdict(_stats({t: False for t in RUNGS}), RULE)
    assert not v["converged"]
    assert v["converged_at"] is None
    assert not v["passes_at_max_rung_only"]
    assert set(v["per_rung"]) == set(RUNGS)


def test_verdict_max_rung_alone_is_not_convergence():
    v = convergence_verdict(_stats({25: False, 50: False, 100: False, 200: True}), RULE)
    assert not v["converged"]
    assert v["passes_at_max_rung_only"]


def test_verdict_nonmonotone_pass_needs_all_larger_rungs():
    v = convergence_verdict(_stats({25: False, 50: True, 100: False, 200: True}), RULE)
    assert not v["converged"]
    assert v["passes_at_max_rung_only"]


# --- certificates and the gated artifact --------------------------------------


def test_certificate_roundtrip(tmp_path):
    payload = certificate_payload(
        estimator="fv_todd",
        task="capitalize",
        model="EleutherAI/pythia-410m@9879c9b",
        converged_at=100,
        n_draws=3,
        evidence_run=str(tmp_path),
        issued="2026-07-18",
    )
    assert payload["estimator"] == "fv_todd@capitalize"
    cert = load_certificate(payload)
    assert isinstance(cert, GateCertificate)
    assert cert.converged_at == 100


def test_gated_fv_requires_matching_task(tmp_path):
    cert = load_certificate(
        certificate_payload(
            estimator="fv_todd",
            task="capitalize",
            model="m",
            converged_at=100,
            n_draws=3,
            evidence_run=str(tmp_path),
            issued="2026-07-18",
        )
    )
    fv = StabilityGatedFV(cert, torch.zeros(4), task="capitalize", draw_seed=1)
    assert fv.certificate is cert
    with pytest.raises(UngatedEstimatorError):
        StabilityGatedFV(cert, torch.zeros(4), task="english-french", draw_seed=1)


def test_gated_fv_unconstructible_without_certificate():
    with pytest.raises(UngatedEstimatorError):
        StabilityGatedFV(None, torch.zeros(4), task="capitalize", draw_seed=1)  # type: ignore[arg-type]
