"""Landing tests for EXP-M5-1 S1 concept-direction gate logic (jtvec.concept_gate)
and the model-side forward/hook ops (jvec.evals.concept)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jtvec.concept_gate import (
    CAPITAL_ROSTER,
    ConceptConvergenceRule,
    ConceptRungStats,
    capital_context_stream,
    certificate_payload,
    convergence_verdict,
    identity_direction,
    injection_deltas,
    mean_difference_by_layer,
    min_pairwise_cosine,
    natural_norms,
    negative_control,
    positive_control,
    rung_prefix,
)
from jtvec.core.draws import DrawSet


# --- context stream / ladder prefix -----------------------------------------

def test_capital_context_stream_prefix_property():
    """A ladder rung is a prefix slice: stream(n=64)[:T] == stream(n=T)."""
    target = CAPITAL_ROSTER[0]  # France/Paris
    full = capital_context_stream(target, seed=1, n=64)
    for t in (8, 16, 32):
        assert capital_context_stream(target, seed=1, n=t) == full[:t]


def test_capital_context_stream_answers_target_and_excludes_it():
    country, capital = CAPITAL_ROSTER[0]
    for prompt in capital_context_stream(CAPITAL_ROSTER[0], seed=3, n=16):
        assert prompt.endswith(f"The capital of {country} is")
        prefix = prompt[: -len(f"The capital of {country} is")]
        assert country not in prefix and capital not in prefix  # no leakage in shots


def test_stream_varies_with_seed():
    a = capital_context_stream(CAPITAL_ROSTER[0], seed=1, n=8)
    b = capital_context_stream(CAPITAL_ROSTER[0], seed=2, n=8)
    assert a != b


def test_rung_prefix_and_bounds():
    assert rung_prefix([1, 2, 3, 4], 2) == [1, 2]
    with pytest.raises(ValueError):
        rung_prefix([1, 2], 3)


# --- direction assembly ------------------------------------------------------

def test_mean_difference_by_layer():
    pos = {0: torch.tensor([[1.0, 0.0], [3.0, 0.0]])}   # mean [2,0]
    neg = {0: torch.tensor([[0.0, 0.0], [0.0, 4.0]])}   # mean [0,2]
    d = mean_difference_by_layer(pos, neg)
    assert torch.allclose(d[0], torch.tensor([2.0, -2.0]))


def test_mean_difference_layer_mismatch_raises():
    with pytest.raises(ValueError):
        mean_difference_by_layer({0: torch.zeros(1, 2)}, {1: torch.zeros(1, 2)})


def test_identity_direction_is_unit_and_concatenates_band():
    raw = {4: torch.tensor([3.0, 0.0]), 5: torch.tensor([0.0, 4.0])}
    u = identity_direction(raw, [4, 5])
    assert u.shape == (4,)
    assert abs(float(u.norm()) - 1.0) < 1e-6
    # 3 and 4 over the combined norm 5
    assert torch.allclose(u, torch.tensor([0.6, 0.0, 0.0, 0.8]), atol=1e-6)


def test_injection_deltas_scaled_to_natural_norm():
    raw = {0: torch.tensor([3.0, 4.0])}  # norm 5
    deltas = injection_deltas(raw, {0: 10.0})
    assert abs(float(deltas[0].norm()) - 10.0) < 1e-5
    assert torch.allclose(deltas[0], torch.tensor([6.0, 8.0]), atol=1e-5)


def test_identity_direction_degenerate_raises():
    with pytest.raises(ValueError):
        identity_direction({4: torch.zeros(2), 5: torch.zeros(2)}, [4, 5])


def test_injection_deltas_zero_layer_is_zeros_not_nan():
    deltas = injection_deltas({0: torch.zeros(3), 1: torch.tensor([3.0, 4.0, 0.0])},
                              {0: 10.0, 1: 5.0})
    assert torch.equal(deltas[0], torch.zeros(3))       # degenerate layer -> no injection
    assert not torch.isnan(deltas[0]).any()
    assert abs(float(deltas[1].norm()) - 5.0) < 1e-5


def test_natural_norms():
    states = {0: torch.tensor([[3.0, 4.0], [0.0, 5.0]])}  # norms 5, 5
    assert abs(natural_norms(states)[0] - 5.0) < 1e-6


def test_min_pairwise_cosine():
    e0, e1 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert abs(min_pairwise_cosine([e0, e0, e0]) - 1.0) < 1e-6
    assert abs(min_pairwise_cosine([e0, e0, e1]) - 0.0) < 1e-6


# --- convergence verdict (witness-rung semantics) ---------------------------

def _rung(t, cos, iqr):
    return ConceptRungStats(n_contexts=t, min_pairwise_cosine=cos, effect_iqr=iqr)


RULE = ConceptConvergenceRule(min_pairwise_cosine=0.95, max_effect_iqr=0.05)


def test_rule_passes():
    assert RULE.passes(_rung(16, 0.96, 0.02))
    assert not RULE.passes(_rung(16, 0.94, 0.02))   # cosine short
    assert not RULE.passes(_rung(16, 0.96, 0.06))   # IQR too wide


def test_converged_at_witness():
    v = convergence_verdict(
        [_rung(8, 0.90, 0.10), _rung(16, 0.96, 0.02),
         _rung(32, 0.97, 0.01), _rung(64, 0.98, 0.01)], RULE)
    assert v["converged"] and v["converged_at"] == 16
    assert not v["passes_at_max_rung_only"]


def test_pass_at_max_rung_only_is_not_convergence():
    v = convergence_verdict(
        [_rung(8, 0.5, 0.2), _rung(16, 0.5, 0.2),
         _rung(32, 0.5, 0.2), _rung(64, 0.99, 0.01)], RULE)
    assert not v["converged"] and v["converged_at"] is None
    assert v["passes_at_max_rung_only"]


def test_all_rungs_pass_converges_at_smallest():
    v = convergence_verdict(
        [_rung(t, 0.99, 0.01) for t in (8, 16, 32, 64)], RULE)
    assert v["converged_at"] == 8


# --- controls ----------------------------------------------------------------

def test_positive_control():
    effect = DrawSet((0.20, 0.25, 0.30), (1, 2, 3))   # median 0.25
    sham = DrawSet((0.00, 0.02, 0.01), (1, 2, 3))     # median 0.01
    assert positive_control(effect, sham, min_gain=0.10).passed
    assert not positive_control(effect, sham, min_gain=0.30).passed


def test_negative_control_quantization_bound():
    # N=40 -> 1/N = 0.025 > base 0.02 -> bound 0.025
    assert negative_control([0.02, -0.024, 0.01], n_eval=40).passed
    assert not negative_control([0.02, -0.03, 0.01], n_eval=40).passed
    # large N -> base floor 0.02 dominates the quantum
    out = negative_control([0.019], n_eval=1000)
    assert out.passed and "max(0.02, 1/1000)" in out.detail


def test_certificate_payload_scopes_concept_in_estimator():
    p = certificate_payload(concept="Paris", model="EleutherAI/pythia-410m@9879c9b",
                            converged_at=16, n_draws=3,
                            evidence_run="results/m5/x", issued="2026-07-21")
    assert p["estimator"] == "s1_concept@Paris"
    assert p["converged_at"] == 16 and p["n_draws"] == 3


# --- model-side ops (tiny random GPT-NeoX via jlens) -------------------------

@pytest.fixture
def tiny_lens(tiny_neox):
    import jlens
    model, tokenizer, _ = tiny_neox
    return jlens.from_hf(model, tokenizer)


def test_answer_states_shapes(tiny_lens):
    from jvec.evals.concept import answer_states
    band = [1, 2]
    states = answer_states(tiny_lens, ["w1 w2 w3", "w4 w5 w6", "w7 w8 w9"], band)
    assert set(states) == set(band)
    for l in band:
        assert states[l].shape == (3, tiny_lens.d_model)


def test_injected_zero_delta_is_identity(tiny_lens):
    from jvec.evals.concept import injected_final_probs
    base = injected_final_probs(tiny_lens, "w1 w2 w3", {})
    same = injected_final_probs(tiny_lens, "w1 w2 w3", {})
    assert torch.allclose(base, same, atol=1e-6)
    assert abs(float(base.sum()) - 1.0) < 1e-4


def test_injected_nonzero_delta_changes_and_preserves_norm(tiny_lens):
    from jvec.evals.concept import injected_final_probs
    base = injected_final_probs(tiny_lens, "w1 w2 w3", {})
    delta = {1: torch.ones(tiny_lens.d_model) * 5.0}
    hooked = injected_final_probs(tiny_lens, "w1 w2 w3", delta)
    assert not torch.allclose(base, hooked, atol=1e-4)
    assert abs(float(hooked.sum()) - 1.0) < 1e-4
