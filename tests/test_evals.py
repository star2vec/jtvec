"""Unit tests for the eval harness math (no model, no network)."""

import torch

from jvec.evals.controls import random_matrices_like, random_unit_vector
from jvec.evals.probe import harmonic_mean, summarize_ranks
from jvec.evals.swap import _SwapHook


class FakeTokenizer:
    """Maps a word to a fixed token id via a small vocabulary."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=True, **kw):
        class Enc:
            pass

        enc = Enc()
        enc.input_ids = [self.vocab[text]]
        return enc


def test_rank_of_word_hand_example():
    from jvec.evals.tasks import rank_of_word

    tok = FakeTokenizer({" x": 2, "x": 2, " X": 2, "X": 2})
    logits = torch.tensor([0.5, 3.0, 1.0, 2.0])  # token 2 has the 3rd-highest logit
    assert rank_of_word(logits, tok, "x") == 3


def test_harmonic_mean():
    assert harmonic_mean([1, 1]) == 1.0
    assert abs(harmonic_mean([1, 100]) - 2 / (1 + 0.01)) < 1e-9


def test_summarize_ranks_pass_k():
    per_item = [
        {"name": "a", "ranks": {"jlens": {0: 1, 1: 50}}},
        {"name": "b", "ranks": {"jlens": {0: 20, 1: 5}}},
    ]
    m = summarize_ranks(per_item, layers=[0, 1], arms=["jlens"], pass_k=10)
    assert m["jlens"]["per_layer"][0]["pass@10"] == 0.5
    assert m["jlens"]["min_over_layers"]["pass@10"] == 1.0  # min ranks: 1 and 5


def test_random_matrices_norm_matched():
    J = {0: torch.randn(16, 16) * 3, 5: torch.eye(16)}
    R = random_matrices_like(J, seed=0)
    for l in J:
        assert abs(R[l].norm().item() - J[l].norm().item()) < 1e-4
    # deterministic given the seed
    R2 = random_matrices_like(J, seed=0)
    assert torch.equal(R[0], R2[0])
    assert not torch.equal(random_matrices_like(J, seed=1)[0], R[0])


def test_swap_hook_identity_jacobian_moves_component():
    """With J = J^+ = I and alpha=1, the hook moves the e_src component onto
    e_dst exactly (up to the norm-preserving rescale)."""
    d = 8
    e_src = torch.zeros(d); e_src[0] = 1.0
    e_dst = torch.zeros(d); e_dst[1] = 1.0
    hook = _SwapHook(torch.eye(d), torch.eye(d), e_src, e_dst, alpha=1.0, positions=[0])
    h = torch.zeros(1, 2, d)
    h[0, 0, 0] = 3.0  # pure e_src content at edited position
    h[0, 1, 0] = 3.0  # untouched position
    hook(None, None, (h,))
    edited, untouched = h[0, 0], h[0, 1]
    assert torch.allclose(untouched, torch.tensor([3.0] + [0.0] * (d - 1)))
    assert abs(edited[0].item()) < 1e-6  # src component removed
    assert abs(edited[1].item() - 3.0) < 1e-6  # moved to dst (norm preserved)


def test_random_unit_vector_deterministic():
    v = random_unit_vector(32, seed=7)
    assert abs(v.norm().item() - 1.0) < 1e-6
    assert torch.equal(v, random_unit_vector(32, seed=7))
