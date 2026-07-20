"""EXP-M5-0 qualification helpers: LRE relation eval (greedy EM over the
evandez/relations data, D-022) and admission logic. Prompt construction and
admission are pure functions with a model-free landing test
(tests/test_qualification.py); the model-calling accuracy pass is thin.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

RELATIONS_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "relations" / "data"


def load_relation(rel_path: str) -> dict:
    """rel_path = '<category>/<file>' (no .json)."""
    return json.loads((RELATIONS_ROOT / f"{rel_path}.json").read_text())


def build_lre_prompt(template: str, exemplars: list[tuple[str, str]], query_subject: str) -> str:
    """10-shot ICL prompt: each exemplar is 'template.format(subj) object', one
    per line, then the query stem template.format(query_subject)."""
    shots = "".join(f"{template.format(s)} {o}\n" for s, o in exemplars)
    return shots + template.format(query_subject)


def em_hit(generated: str, obj: str) -> bool:
    """Greedy exact-match (D-012), case/space-relaxed prefix: the object is the
    start of the greedy continuation."""
    g = generated.strip().lower()
    o = obj.strip().lower()
    return g.startswith(o)


@torch.no_grad()
def lre_relation_accuracy(model, tok, relation: dict, n_shots: int, n_test: int,
                          device, seed: int = 0) -> dict:
    """10-shot greedy-EM accuracy on held-out subjects of one relation."""
    import random
    template = relation["prompt_templates"][0]
    samples = [(s["subject"], s["object"]) for s in relation["samples"]]
    rng = random.Random(seed)
    rng.shuffle(samples)
    exemplars = samples[:n_shots]
    heldout = samples[n_shots:n_shots + n_test]
    if not heldout:
        return {"accuracy": 0.0, "n": 0, "per_item": []}
    per_item = []
    for subj, obj in heldout:
        prompt = build_lre_prompt(template, exemplars, subj)
        ids = tok(prompt, return_tensors="pt").to(device)
        max_new = len(tok(" " + obj, add_special_tokens=False)["input_ids"]) + 2
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:])
        hit = em_hit(gen, obj)
        per_item.append({"subject": subj, "object": obj, "gen": gen.strip()[:40], "hit": hit})
    acc = sum(i["hit"] for i in per_item) / len(per_item)
    return {"accuracy": acc, "n": len(per_item), "per_item": per_item}


def admit(scores: dict[str, float], bar: float) -> dict[str, bool]:
    """Per-item admission at a bar."""
    return {k: v >= bar for k, v in scores.items()}
