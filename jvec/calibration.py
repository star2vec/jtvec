"""Seeded selection of calibration and held-out prompts from a text corpus.

Selection is deterministic given (corpus, seed, n_prompts, n_heldout, model
tokenizer): documents are shuffled with a dedicated RNG, then the first
``n_prompts`` docs with >= ``min_tokens`` tokens become calibration prompts and
the next ``n_heldout`` become held-out report prompts. Exact texts and sha256
hashes are returned so the cache manifest can pin them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import datasets
import numpy as np

from jvec.config import Config


@dataclass(frozen=True)
class PromptSet:
    calibration: list[str]
    heldout: list[str]
    calibration_sha256: list[str]
    heldout_sha256: list[str]
    corpus: str
    seed: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_prompts(cfg: Config, tokenizer, *, min_tokens: int | None = None) -> PromptSet:
    """Pick calibration + held-out documents (see module docstring)."""
    if min_tokens is None:
        min_tokens = cfg.fit.max_seq_len
    ds = datasets.load_dataset(cfg.calibration.corpus, split=cfg.calibration.split)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(ds))

    needed = cfg.calibration.n_prompts + cfg.calibration.n_heldout
    picked: list[str] = []
    for idx in order:
        text = ds[int(idx)]["text"]
        # Token count via the model's tokenizer; cheap upper bound first.
        if len(text) < min_tokens:  # < 1 char per token is impossible
            continue
        n_tok = len(tokenizer(text, truncation=True, max_length=min_tokens + 1).input_ids)
        if n_tok >= min_tokens:
            picked.append(text)
        if len(picked) == needed:
            break
    if len(picked) < needed:
        raise ValueError(
            f"only {len(picked)}/{needed} documents in {cfg.calibration.corpus} "
            f"had >= {min_tokens} tokens"
        )

    calib = picked[: cfg.calibration.n_prompts]
    heldout = picked[cfg.calibration.n_prompts :]
    return PromptSet(
        calibration=calib,
        heldout=heldout,
        calibration_sha256=[_sha256(t) for t in calib],
        heldout_sha256=[_sha256(t) for t in heldout],
        corpus=cfg.calibration.corpus,
        seed=cfg.seed,
    )
