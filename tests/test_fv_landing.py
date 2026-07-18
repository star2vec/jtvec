"""The FV-injection landing test required by CONSTRAINTS before any M2 use.

CONSTRAINTS (VERIFIED): the Todd repo's ``add_function_vector`` silently
no-ops under transformers 5.x — NeoX blocks return plain tensors, the hook
only edited tuples, and v1 observed induction gain of exactly +0.0%.
jvec.fv patches the hook at import time. These tests prove, through the
real entry point (``function_vector_intervention``) on a GPT-NeoX module
tree, that the patched hook actually lands the vector: outputs change when
a vector is injected, do not change when a zero vector is injected, and the
residual stream at the edit layer moves by exactly the injected vector.

Under the unpatched original, test_nonzero_fv_changes_logits and
test_fv_lands_at_edit_layer_with_correct_value fail (clean == intervened).
"""

from __future__ import annotations

import pytest
import torch

import jvec.fv  # noqa: F401  (sys.path setup for the Todd repo + the hook patch)
import utils.intervention_utils as iu
from baukit import TraceDict

EDIT_LAYER = 1


def _fv(dim: int, scale: float, seed: int = 3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=gen) * scale


def test_patch_is_installed():
    # The module-global that function_vector_intervention resolves at call
    # time must be jvec.fv's patched closure, not the vendored original.
    assert iu.add_function_vector.__module__ == "jvec.fv"


def test_hook_edits_plain_tensor_output():
    vec = torch.arange(8.0)
    fn = iu.add_function_vector(2, vec.reshape(1, 8), "cpu")
    out = torch.zeros(1, 5, 8)
    edited = fn(out.clone(), "gpt_neox.layers.2")
    assert torch.equal(edited[0, -1], vec)
    assert torch.equal(edited[0, :-1], out[0, :-1])  # only idx position moves


def test_hook_edits_tuple_output():
    vec = torch.arange(8.0)
    fn = iu.add_function_vector(2, vec.reshape(1, 8), "cpu")
    hidden = torch.zeros(1, 5, 8)
    edited = fn((hidden.clone(), "extra"), "gpt_neox.layers.2")
    assert isinstance(edited, tuple)
    assert torch.equal(edited[0][0, -1], vec)


def test_hook_leaves_other_layers_untouched():
    vec = torch.ones(1, 8)
    fn = iu.add_function_vector(2, vec, "cpu")
    out = torch.zeros(1, 5, 8)
    edited = fn(out.clone(), "gpt_neox.layers.1")
    assert torch.equal(edited, out)


def test_zero_fv_keeps_logits_identical(tiny_neox):
    model, tokenizer, model_config = tiny_neox
    clean, intervened = iu.function_vector_intervention(
        "w1 w2 w3", "w4", EDIT_LAYER,
        torch.zeros(model_config["resid_dim"]),
        model, model_config, tokenizer,
    )
    assert torch.allclose(clean, intervened, atol=1e-6)


def test_nonzero_fv_changes_logits(tiny_neox):
    # THE landing assertion: v1's observed failure mode was clean == intervened.
    model, tokenizer, model_config = tiny_neox
    fv = _fv(model_config["resid_dim"], scale=10.0)
    clean, intervened = iu.function_vector_intervention(
        "w1 w2 w3", "w4", EDIT_LAYER, fv, model, model_config, tokenizer,
    )
    assert (clean - intervened).abs().max() > 1e-3


def test_fv_lands_at_edit_layer_with_correct_value(tiny_neox):
    # The residual stream after the edit layer must move by exactly fv at the
    # last token; the block before the edit layer must not move at all.
    model, tokenizer, model_config = tiny_neox
    dim = model_config["resid_dim"]
    fv = _fv(dim, scale=1.0)
    inputs = tokenizer("w1 w2 w3", return_tensors="pt")

    clean = model(**inputs, output_hidden_states=True)
    hook = iu.add_function_vector(EDIT_LAYER, fv.reshape(1, dim), "cpu")
    with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=hook):
        intervened = model(**inputs, output_hidden_states=True)

    # hidden_states[l + 1] is the output of block l.
    delta_edit = (
        intervened.hidden_states[EDIT_LAYER + 1][0, -1]
        - clean.hidden_states[EDIT_LAYER + 1][0, -1]
    )
    delta_before = (
        intervened.hidden_states[EDIT_LAYER][0, -1]
        - clean.hidden_states[EDIT_LAYER][0, -1]
    )
    assert torch.allclose(delta_edit, fv, rtol=1e-4, atol=1e-5)
    assert delta_before.abs().max() == 0.0


def test_landing_positions_only_last_token(tiny_neox):
    # idx=-1 is the M2 injection position; earlier positions must be untouched.
    model, tokenizer, model_config = tiny_neox
    dim = model_config["resid_dim"]
    fv = _fv(dim, scale=1.0)
    inputs = tokenizer("w1 w2 w3", return_tensors="pt")

    clean = model(**inputs, output_hidden_states=True)
    hook = iu.add_function_vector(EDIT_LAYER, fv.reshape(1, dim), "cpu")
    with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=hook):
        intervened = model(**inputs, output_hidden_states=True)

    edit_out_clean = clean.hidden_states[EDIT_LAYER + 1][0, :-1]
    edit_out_interv = intervened.hidden_states[EDIT_LAYER + 1][0, :-1]
    assert torch.allclose(edit_out_clean, edit_out_interv, atol=1e-6)
