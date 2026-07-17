"""Probing eval: where in the layer stack does each lens surface the
intermediate token, and how does that compare to the logit lens and to a
random-matrix control?

For each item we run ONE forward pass, record the residual at every fitted
layer at the readout position (last prompt token), then decode it through each
arm: ``jlens`` (J_l @ h), ``logit`` (h unchanged), and ``random-<seed>``
(norm-matched Gaussian R_l @ h). Metrics per task x arm x layer: harmonic mean
rank (primary) and pass@k; plus the repo-protocol min-over-layers pass@k.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder, JacobianLens

from jvec.evals.controls import random_matrices_like
from jvec.evals.tasks import Task, rank_of_word


@torch.no_grad()
def _residuals_at_readout(model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    input_ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        return {
            l: recorder.activations[l][0, -1].detach().float() for l in layers
        }


@torch.no_grad()
def probe_task(
    model,
    tokenizer,
    lens: JacobianLens,
    task: Task,
    *,
    pass_k: int,
    n_random_seeds: int,
) -> dict:
    """Per-item, per-layer, per-arm ranks of the item's intermediates."""
    layers = lens.source_layers
    device = model.input_device
    arms: dict[str, dict[int, torch.Tensor]] = {
        "jlens": {l: lens.jacobians[l].to(device) for l in layers},
        "logit": {l: None for l in layers},
    }
    for seed in range(n_random_seeds):
        mats = random_matrices_like(lens.jacobians, seed=seed)
        arms[f"random-{seed}"] = {l: mats[l].to(device) for l in layers}

    per_item = []
    for item in task.items:
        residuals = _residuals_at_readout(model, item["prompt"], layers)
        ranks: dict[str, dict[int, int]] = {}
        for arm_name, mats in arms.items():
            arm_ranks = {}
            for l in layers:
                h = residuals[l]
                readout = h if mats[l] is None else h @ mats[l].T
                logits = model.unembed(readout).float().cpu()
                arm_ranks[l] = min(
                    rank_of_word(logits, tokenizer, word)
                    for word in item["intermediates"]
                )
            ranks[arm_name] = arm_ranks
        per_item.append({"name": item["name"], "ranks": ranks})

    return {
        "task": task.name,
        "layers": layers,
        "arms": sorted(arms),
        "per_item": per_item,
        "metrics": summarize_ranks(per_item, layers, sorted(arms), pass_k),
    }


def harmonic_mean(values: list[int]) -> float:
    return len(values) / sum(1.0 / v for v in values)


def summarize_ranks(
    per_item: list[dict], layers: list[int], arms: list[str], pass_k: int
) -> dict:
    metrics: dict[str, dict] = {}
    for arm in arms:
        per_layer = {}
        for l in layers:
            ranks = [item["ranks"][arm][l] for item in per_item]
            per_layer[l] = {
                "hmr": round(harmonic_mean(ranks), 2),
                f"pass@{pass_k}": round(
                    sum(r <= pass_k for r in ranks) / len(ranks), 4
                ),
            }
        min_ranks = [min(item["ranks"][arm].values()) for item in per_item]
        metrics[arm] = {
            "per_layer": per_layer,
            "min_over_layers": {
                "hmr": round(harmonic_mean(min_ranks), 2),
                f"pass@{pass_k}": round(
                    sum(r <= pass_k for r in min_ranks) / len(min_ranks), 4
                ),
            },
        }
    return metrics
