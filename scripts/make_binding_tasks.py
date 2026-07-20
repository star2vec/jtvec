"""Generate the D-026 binding battery (bind2, bind3) — templates approved by
Ecaterina 2026-07-21. Feng & Steinhardt-style object-container binding: n
single-token objects bound to numbered boxes, query one object's box; greedy
exact-match of the box number. 10-shot ICL (prior completed instances prefix
the query). Deterministic. Writes tasks/binding/*.json.

Usage: uv run python scripts/make_binding_tasks.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tasks" / "binding"
N_ITEMS = 60
N_SHOTS = 10

OBJECTS = [
    "apple", "key", "book", "coin", "ball", "cup", "pen", "ring", "cake", "lamp",
    "hat", "sock", "fork", "drum", "kite", "boat", "leaf", "rope", "bell", "card",
]


def _instance(rng: random.Random, n: int) -> tuple[str, str]:
    """One binding instance -> (completed_text, query_stem_with_target). n
    objects bound to boxes 1..n (shuffled objects), query one object."""
    objs = rng.sample(OBJECTS, n)
    binding = ". ".join(f"The {o} is in Box {i + 1}" for i, o in enumerate(objs)) + ". "
    q_idx = rng.randrange(n)
    q_obj, q_box = objs[q_idx], q_idx + 1
    stem = binding + f"The {q_obj} is in Box"
    return stem + f" {q_box}.", stem, str(q_box)


def build(n: int) -> list[dict]:
    rng = random.Random(1000 + n)
    items = []
    for k in range(N_ITEMS):
        shots = [_instance(rng, n)[0] for _ in range(N_SHOTS)]
        _, stem, target = _instance(rng, n)
        items.append({
            "name": f"bind{n}-{k:02d}",
            "prompt": " ".join(shots) + " " + stem,
            "target": target,
            "intermediates": [target],
        })
    return items


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n, name in [(2, "bind2"), (3, "bind3")]:
        items = build(n)
        path = OUT / f"{name}.json"
        path.write_text(json.dumps({"task": name, "protocol": "completion", "items": items}, indent=1))
        print(f"wrote {path} ({len(items)} items, chance 1/{n})")


if __name__ == "__main__":
    main()
