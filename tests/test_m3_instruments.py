"""M3 landing/property tests for the vendored intervention hooks, plus the
m3 control-rule logic. The hooks are pure tensor operations — every contract
here runs model-free.

Landing contracts (the transformers-5 lesson, applied before first use):
each hook must actually edit a plain-tensor block output, must handle the
tuple form, and must touch only the final position.
"""

from __future__ import annotations

import pytest
import torch

from jvec.evals.exp3 import JSpaceAblateHook, MatchedNoiseHook, ProjectOutHook
from jvec.evals.fvswap import _FVSwapHook
from jtvec.m3_instruments import (
    AblationControlRule,
    ReportProbeControlRule,
    SwapControlRule,
    explicit_rule_context,
    load_certified_fv,
    quantized_bound,
    shared_query_map,
    verify_lens_manifest,
)

D = 16


def _hidden(seq: int = 5, d: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, d, generator=g)


# --- ProjectOutHook -----------------------------------------------------------


def test_project_out_removes_span_and_only_last_position():
    dirs = torch.eye(D)[:2]  # span{e0, e1}
    h = _hidden()
    before = h.clone()
    ProjectOutHook(dirs)(None, None, h)
    assert torch.allclose(h[0, -1, :2], torch.zeros(2), atol=1e-5)
    assert torch.allclose(h[0, -1, 2:], before[0, -1, 2:], atol=1e-5)
    assert torch.equal(h[0, :-1], before[0, :-1])


def test_project_out_idempotent_and_handles_tuple():
    g = torch.Generator().manual_seed(3)
    dirs = torch.randn(3, D, generator=g)
    h = _hidden(seed=1)
    hook = ProjectOutHook(dirs)
    hook(None, None, h)
    once = h[0, -1].clone()
    hook(None, None, h)
    assert torch.allclose(h[0, -1], once, atol=1e-5)  # projection is idempotent

    tup = (_hidden(seed=2), "attn-extra")
    before = tup[0].clone()
    hook(None, None, tup)
    assert not torch.allclose(tup[0][0, -1], before[0, -1])  # tuple form edited


# --- JSpaceAblateHook ---------------------------------------------------------


def test_jspace_ablation_zeroes_top_atom_readout():
    g = torch.Generator().manual_seed(4)
    J = torch.randn(D, D, generator=g)
    W_U = torch.randn(40, D, generator=g)
    m = 5
    h = _hidden(seed=5)
    top = (h[0, -1] @ J.T @ W_U.T).topk(m).indices  # what the hook will target
    JSpaceAblateHook(J, W_U, m)(None, None, h)
    residual_readout = W_U[top] @ (J @ h[0, -1])
    assert residual_readout.abs().max() < 1e-3  # atom span removed exactly


# --- MatchedNoiseHook ---------------------------------------------------------


def test_matched_noise_preserves_norm_and_hits_target_delta():
    rel = 0.3
    h = _hidden(seed=6)
    before = h[0, -1].clone()
    MatchedNoiseHook(rel, D, seed=7)(None, None, h)
    after = h[0, -1]
    assert float(after.norm()) == pytest.approx(float(before.norm()), rel=1e-4)
    disp = float((after - before).norm() / before.norm())
    assert 0.5 * rel < disp < 1.5 * rel

    h2 = _hidden(seed=6)
    MatchedNoiseHook(rel, D, seed=7)(None, None, h2)
    assert torch.allclose(h2[0, -1], after, atol=1e-6)  # deterministic per seed


# --- _FVSwapHook ---------------------------------------------------------------


def test_fv_swap_direct_moves_a_component_onto_b():
    d = 8
    a, b = torch.eye(d)[0], torch.eye(d)[1]
    h = torch.zeros(1, 3, d)
    h[0, -1] = 2 * a + 3 * b + 4 * torch.eye(d)[2]
    norm_before = float(h[0, -1].norm())
    _FVSwapHook(a, b)(None, None, h)
    assert float(h[0, -1] @ a) == pytest.approx(0.0, abs=1e-5)  # a-component gone
    assert float(h[0, -1] @ b) > 0  # moved onto b
    assert float(h[0, -1].norm()) == pytest.approx(norm_before, rel=1e-5)


def test_fv_swap_lens_path_with_identity_matches_direct():
    d = 8
    g = torch.Generator().manual_seed(8)
    a = torch.randn(d, generator=g)
    b = torch.randn(d, generator=g)
    h1 = _hidden(seq=3, d=d, seed=9)
    h2 = h1.clone()
    _FVSwapHook(a, b)(None, None, h1)
    _FVSwapHook(a, b, J=torch.eye(d), J_pinv=torch.eye(d))(None, None, h2)
    assert torch.allclose(h1[0, -1], h2[0, -1], atol=1e-4)


# --- m3 logic -------------------------------------------------------------------


def test_verify_lens_manifest_flags_identity_fields_only():
    ref = {
        "model_name": "m", "model_revision": "r", "calibration_sha256": ["a", "b"],
        "n_prompts": 10, "max_seq_len": 128, "dim_batch": 8, "skip_first": 4,
        "source_layers": None, "target_layer": None, "seed": 0,
        "jlens_commit": "581d398", "device": "mps", "wall_clock_s": 919.3,
    }
    refit = dict(ref, device="cuda", wall_clock_s=200.0)  # provenance may differ
    assert verify_lens_manifest(refit, ref) == {}
    bad = dict(refit, calibration_sha256=["a", "X"])
    assert set(verify_lens_manifest(bad, ref)) == {"calibration_sha256"}


def test_quantized_bound_floors_at_one_item():
    assert quantized_bound(0.05, 100) == 0.05
    assert quantized_bound(0.05, 10) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        quantized_bound(0.05, 0)


def test_ablation_rule_verdict():
    rule = AblationControlRule(min_exec_drop=0.15, max_sham_dev=0.05)
    v = rule.verdict(none_acc=0.9, ablated_acc=0.5, sham_acc=0.88, n=30)
    assert v["positive_pass"] and v["negative_pass"]
    v = rule.verdict(none_acc=0.9, ablated_acc=0.8, sham_acc=0.7, n=30)
    assert not v["positive_pass"] and not v["negative_pass"]
    # quantization: at n=16 a one-item sham wobble (0.0625) is inside the bound
    v = rule.verdict(none_acc=0.9, ablated_acc=0.5, sham_acc=0.9 - 1 / 16, n=16)
    assert v["negative_pass"]


def test_report_probe_rule_verdict():
    rule = ReportProbeControlRule(min_detection=0.8, max_shuffled_above_prior=0.15)
    v = rule.verdict(
        detection_by_phrasing={"P1": 0.6, "P2": 0.9, "P3": 0.7},
        shuffled_acc=0.2, prior=0.125, n=12,
    )
    assert v["positive_pass"] and v["negative_pass"]
    v = rule.verdict(
        detection_by_phrasing={"P1": 0.5, "P2": 0.6, "P3": 0.7},
        shuffled_acc=0.5, prior=0.125, n=12,
    )
    assert not v["positive_pass"] and not v["negative_pass"]


def test_swap_rule_verdict():
    rule = SwapControlRule(min_b_gain=0.2, max_random_dev=0.05)
    v = rule.verdict(
        b_rates={"none": 0.1, "direct_swap": 0.5, "lens_swap": 0.15, "random_target": 0.12},
        n=30,
    )
    assert v["positive_pass"] and v["negative_pass"]
    v = rule.verdict(
        b_rates={"none": 0.1, "direct_swap": 0.2, "lens_swap": 0.2, "random_target": 0.4},
        n=30,
    )
    assert not v["positive_pass"] and not v["negative_pass"]


def test_explicit_rule_context_states_label_only_in_rule():
    ctx = explicit_rule_context("<BOS>", "plural", [("dog", "RUN"), ("cat", "blue")])
    assert ctx.startswith("<BOS>Rule:")
    assert "plural" in ctx.split("\n\n")[0]  # label in the rule sentence
    assert "Q: dog\nA: RUN" in ctx  # shuffled pairs carried verbatim


def test_load_certified_fv_requires_certificate():
    with pytest.raises(KeyError):
        load_certified_fv(None, "capitalize", "rev", certificates={})


def test_shared_query_map_intersects_with_distinct_targets():
    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, sl):
            return {
                "input": [x for x, _ in self.rows],
                "output": [y for _, y in self.rows],
            }

    ds_a = {"train": FakeSplit([("dog", "DOG")]), "valid": FakeSplit([]), "test": FakeSplit([("cat", "CAT")])}
    ds_b = {"train": FakeSplit([("dog", "dogs")]), "valid": FakeSplit([("cat", "CAT")]), "test": FakeSplit([])}
    shared = shared_query_map(ds_a, ds_b)
    assert shared == {"dog": ("DOG", "dogs")}  # cat dropped: identical target
