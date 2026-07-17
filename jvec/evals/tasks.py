"""Task file loading and token-level scoring helpers.

Task JSON schema (superset of the jacobian-lens eval schema)::

    {"task": str, "protocol": "completion" | "typo", "items": [
        {"name": str, "prompt": str, "target": str?, "clean_prompt": str?,
         "intermediates": [str, ...]}]}

Scoring convention: prompts end mid-sentence without trailing space, so a
word's canonical surface form is its leading-space variant. Ranks and matches
use the *first token* of each surface form, taking the best (minimum) rank over
the leading-space and bare variants — the same relaxation the repo protocol
uses via synonym expansion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from jvec.utils import REPO_ROOT

TASKS_DIR = REPO_ROOT / "tasks"


@dataclass(frozen=True)
class Task:
    name: str
    protocol: str
    items: list[dict]


def load_tasks(directory: Path | None = None) -> list[Task]:
    directory = directory or TASKS_DIR
    tasks = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        tasks.append(Task(name=raw["task"], protocol=raw["protocol"], items=raw["items"]))
    if not tasks:
        raise FileNotFoundError(f"no task files in {directory}; run scripts/make_tasks.py")
    return tasks


def surface_token_ids(tokenizer, word: str) -> list[int]:
    """First-token ids of the word's surface forms.

    Leading-space and bare variants, in both the written case and the
    title/lower-cased form ("Fish and" -> " Chips" is a correct completion of
    target "chips"). The expansion is applied uniformly to the baseline gate
    and to every lens arm, mirroring the repo protocol's synonym expansion.
    """
    variants = [word, word.capitalize(), word.lower()]
    ids = []
    for v in variants:
        for form in (f" {v}", v):
            # add_special_tokens=False: jlens's from_hf sets add_bos_token=True
            # on the shared tokenizer, which would otherwise make ids[0] the BOS.
            first = tokenizer(form, add_special_tokens=False).input_ids[0]
            if first not in ids:
                ids.append(first)
    return ids


def rank_of_word(logits: torch.Tensor, tokenizer, word: str) -> int:
    """Best (min) rank of the word's surface tokens in a [vocab] logit vector.

    Rank is 1-indexed: rank 1 = argmax.
    """
    return int(
        min(
            1 + (logits > logits[t]).sum()
            for t in surface_token_ids(tokenizer, word)
        )
    )


def top1_matches_word(logits: torch.Tensor, tokenizer, word: str) -> bool:
    return int(logits.argmax()) in surface_token_ids(tokenizer, word)
