"""Baseline gate: can GPT-2-small actually do each task in-context?

- ``completion`` protocol: greedy top-1 next token matches the item's target.
- ``typo`` protocol: greedy top-1 after the misspelled prompt matches the
  greedy top-1 after the clean prompt (behavioral read-through of the typo).

Tasks whose mean accuracy falls below the config threshold are excluded from
the lens evals and reported as dropped.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder

from jvec.evals.tasks import Task, top1_matches_word


@torch.no_grad()
def _final_logits(model, prompt: str) -> torch.Tensor:
    """Model's next-token logits at the last prompt position.

    Reads the final block's output via ActivationRecorder and unembeds it —
    the same path jlens uses (``model.forward`` returns the text module's
    ``last_hidden_state``, which already has the final norm applied, so
    ``unembed`` on it would double-apply the norm).
    """
    final = model.n_layers - 1
    input_ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final][0, -1].detach()
    return model.unembed(residual.float()).float().cpu()


def score_task(model, tokenizer, task: Task) -> dict:
    per_item = []
    for item in task.items:
        logits = _final_logits(model, item["prompt"])
        top1 = int(logits.argmax())
        if task.protocol == "completion":
            correct = top1_matches_word(logits, tokenizer, item["target"])
            expected = item["target"]
        elif task.protocol == "typo":
            clean_logits = _final_logits(model, item["clean_prompt"])
            correct = top1 == int(clean_logits.argmax())
            expected = tokenizer.decode([int(clean_logits.argmax())])
        elif task.protocol == "swap":
            # Gate on the model knowing the *base* fact it is asked to swap.
            correct = top1_matches_word(logits, tokenizer, item["answer"])
            expected = item["answer"]
        else:
            raise ValueError(f"unknown protocol {task.protocol!r}")
        per_item.append(
            {
                "name": item["name"],
                "correct": bool(correct),
                "top1": tokenizer.decode([top1]),
                "expected": expected,
            }
        )
    accuracy = sum(i["correct"] for i in per_item) / len(per_item)
    return {"task": task.name, "protocol": task.protocol, "accuracy": accuracy, "per_item": per_item}
