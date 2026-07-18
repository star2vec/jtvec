"""Experiment 1: read function vectors through the J-lens vs the logit lens.

For each task's FV (Todd attention-head FV: one d-vector; Hendel hidden-state
vector: one d-vector per layer) and each lens layer l we decode three ways:

- ``jlens``:  unembed(J_l @ v)   — the J-lens readout
- ``logit``:  unembed(v)         — Nadaf's logit-lens baseline
- ``random-<s>``: same readouts on a norm-matched random vector

and score (1) the best rank of a small set of task-label words ("opposite",
"French", ...) — can the lens *verbalize the task*; (2) the mean rank of the
task's output-token cloud — does the FV point at the task's output vocabulary.
Ranks are over the full vocab, 1 = top.
"""

from __future__ import annotations

import torch
from jlens import JacobianLens

from jvec.evals.tasks import rank_of_word, surface_token_ids

#: Words that verbalize each task. Scored as best (min) rank over the set.
TASK_LABEL_WORDS: dict[str, list[str]] = {
    "antonym": ["opposite", "opposites", "antonym", "reverse", "contrary"],
    "synonym": ["synonym", "synonyms", "same", "similar", "equivalent"],
    "country-capital": ["capital", "capitals", "city"],
    "country-currency": ["currency", "currencies", "money"],
    "english-french": ["French", "France", "translate", "translation"],
    "english-spanish": ["Spanish", "Spain", "translate", "translation"],
    "present-past": ["past", "tense", "verb"],
    "singular-plural": ["plural", "plurals", "singular", "noun"],
    "capitalize": ["capital", "capitalize", "uppercase", "case"],
    "landmark-country": ["country", "countries", "location"],
    "person-sport": ["sport", "sports", "plays"],
}


def output_token_ids(dataset, tokenizer, max_outputs: int = 50) -> list[int]:
    """First token ids of the task's output vocabulary (from the test split)."""
    outputs = []
    for pair in dataset["test"]:
        out = pair["output"]
        outputs.append(out[0] if isinstance(out, list) else out)
    unique = sorted(set(map(str, outputs)))[:max_outputs]
    ids = {surface_token_ids(tokenizer, o)[0] for o in unique}
    return sorted(ids)


def _mean_rank_of_ids(logits: torch.Tensor, ids: list[int]) -> float:
    ranks = torch.zeros(len(ids))
    for j, t in enumerate(ids):
        ranks[j] = 1 + (logits > logits[t]).sum()
    return float(ranks.mean())


@torch.no_grad()
def decode_vector(
    model, tokenizer, lens: JacobianLens, vector: torch.Tensor, task: str,
    out_ids: list[int], *, layers: list[int], topk: int = 10,
) -> dict:
    """Decode one residual-space vector through every arm at every layer."""
    device = model.input_device
    v = vector.float().to(device)
    label_words = TASK_LABEL_WORDS[task]
    per_layer = {}
    for l in layers:
        readouts = {
            "jlens": model.unembed(lens.transport(v, l)).float().cpu(),
            "logit": model.unembed(v).float().cpu(),
        }
        entry = {}
        for arm, logits in readouts.items():
            entry[arm] = {
                "label_rank": min(rank_of_word(logits, tokenizer, w) for w in label_words),
                "output_mean_rank": _mean_rank_of_ids(logits, out_ids),
                "topk": [tokenizer.decode([t]) for t in logits.topk(topk).indices],
            }
        per_layer[l] = entry
    return per_layer


def random_like(vector: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vector.shape, generator=g, dtype=torch.float32)
    return r * (vector.norm() / r.norm())
