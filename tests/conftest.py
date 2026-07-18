"""Shared fixtures (v2-original): a tiny random GPT-NeoX for hook/FV tests.

The tiny model exists so tests can drive the *real* vendored code paths
(baukit TraceDict hooks, Todd's compute_function_vector and
function_vector_intervention) without a network or a real checkpoint. Its
module tree matches Pythia's (gpt_neox.layers.<L>...), which the hook-name
parsing in those code paths depends on.
"""

from __future__ import annotations

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

TINY_LAYERS = 4
TINY_HEADS = 4
TINY_DIM = 32


@pytest.fixture(scope="session")
def tiny_neox():
    """(model, tokenizer, model_config) — random weights, deterministic seed."""
    torch.manual_seed(0)
    config = transformers.GPTNeoXConfig(
        vocab_size=128,
        hidden_size=TINY_DIM,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = transformers.GPTNeoXForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    words = ["[UNK]", "[EOS]"] + [f"w{i}" for i in range(60)]
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Same shape as jvec.fv.load_fv_model's MODEL_CONFIG for GPT-NeoX.
    model_config = {
        "n_heads": TINY_HEADS,
        "n_layers": TINY_LAYERS,
        "resid_dim": TINY_DIM,
        "name_or_path": "tiny-gpt-neox",
        "attn_hook_names": [
            f"gpt_neox.layers.{l}.attention.dense" for l in range(TINY_LAYERS)
        ],
        "layer_hook_names": [f"gpt_neox.layers.{l}" for l in range(TINY_LAYERS)],
        "prepend_bos": False,
    }
    return model, tokenizer, model_config
