"""Causal swap eval through lens coordinates (pseudoinverse write-back).

For a swap item (prompt "... The capital of France is", intermediate "France",
swap_to "Italy", swap_answer "Rome"):

1. In the final-layer basis, take unit directions e_A, e_B for the surface
   tokens of ``intermediate`` and ``swap_to`` from the unembedding rows (the
   standard logit-lens approximation that ignores the final norm).
2. At each band layer l, at the prompt positions where the source token
   occurs, transport the residual with J_l and move its component along e_A
   onto e_B:  t' = t + alpha * <t, e_A> * (e_B - e_A).
3. Write the edit back into layer-l coordinates with a *truncated*
   pseudoinverse (``rcond``), then rescale the edited residual to its original
   norm.

The truncation and norm preservation are load-bearing: diagnostics
(2026-07-14) showed the full pseudoinverse injects huge components along J's
weak singular directions, destroying the computation generically (p(answer)
-> 0, p(swap_answer) unmoved, 0/6 flips at any alpha), while rcond=0.05 +
norm preservation reaches 5-6/6 top-1 flips — the same effect size as a
direct token-patching control (+0.37 dp).

Metric: delta-p(swap_answer) and delta-p(original answer) at the final
position, vs. the unhooked forward. Control: same edit energy, but the
A-component is moved onto a random unit direction instead of e_B.

``torch.linalg.pinv`` runs on CPU fp32 (flaky on MPS); results are moved to
the model device once per task.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder, JacobianLens

from jvec.evals.controls import random_unit_vector
from jvec.evals.tasks import Task, surface_token_ids


def _unembed_direction(model, tokenizer, word: str) -> torch.Tensor:
    """Unit direction in the final basis for the word's primary surface token."""
    token_id = surface_token_ids(tokenizer, word)[0]
    row = model._lm_head.weight[token_id].detach().float().cpu()
    return row / row.norm()


def pinv_jacobians(
    lens: JacobianLens, layers: list[int], *, rcond: float = 0.0
) -> dict[int, torch.Tensor]:
    return {
        l: torch.linalg.pinv(lens.jacobians[l].cpu().float(), rcond=rcond)
        for l in layers
    }


@torch.no_grad()
def _final_probs(model, input_ids) -> torch.Tensor:
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final][0, -1].detach()
    return torch.softmax(model.unembed(residual.float()).float().cpu(), dim=-1)


class _SwapHook:
    """Norm-preserving lens-coordinate swap at selected positions of one block."""

    def __init__(self, J, J_pinv, e_src, e_dst, alpha, positions):
        self.J, self.J_pinv = J, J_pinv
        self.e_src, self.e_dst = e_src, e_dst
        self.alpha = alpha
        self.positions = positions

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden.float()[:, self.positions, :]
        norms = h.norm(dim=-1, keepdim=True)
        transported = h @ self.J.T
        coeff = transported @ self.e_src  # [batch, n_positions]
        delta_t = self.alpha * coeff[..., None] * (self.e_dst - self.e_src)
        edited = h + delta_t @ self.J_pinv.T
        edited = edited * (norms / edited.norm(dim=-1, keepdim=True))
        hidden[:, self.positions, :] = edited.to(hidden.dtype)


@torch.no_grad()
def swap_task(
    model,
    tokenizer,
    lens: JacobianLens,
    task: Task,
    *,
    band: tuple[int, int],
    alpha: float,
    rcond: float,
    n_random_seeds: int,
) -> dict:
    device = model.input_device
    band_layers = [l for l in lens.source_layers if band[0] <= l <= band[1]]
    J = {l: lens.jacobians[l].to(device) for l in band_layers}
    J_pinv = {
        l: p.to(device)
        for l, p in pinv_jacobians(lens, band_layers, rcond=rcond).items()
    }

    per_item = []
    for item in task.items:
        input_ids = model.encode(item["prompt"])
        src_token = surface_token_ids(tokenizer, item["intermediate"])[0]
        positions = (input_ids[0] == src_token).nonzero().flatten().tolist()
        if not positions:
            per_item.append({"name": item["name"], "skipped": "source token not in prompt"})
            continue
        e_src = _unembed_direction(model, tokenizer, item["intermediate"]).to(device)
        e_swap = _unembed_direction(model, tokenizer, item["swap_to"]).to(device)
        answer_id = surface_token_ids(tokenizer, item["answer"])[0]
        swap_answer_id = surface_token_ids(tokenizer, item["swap_answer"])[0]

        p_base = _final_probs(model, input_ids)

        def run_with_direction(e_dst: torch.Tensor) -> torch.Tensor:
            handles = [
                model.layers[l].register_forward_hook(
                    _SwapHook(J[l], J_pinv[l], e_src, e_dst, alpha, positions)
                )
                for l in band_layers
            ]
            try:
                return _final_probs(model, input_ids)
            finally:
                for h in handles:
                    h.remove()

        p_swap = run_with_direction(e_swap)
        random_dps = []
        for seed in range(n_random_seeds):
            e_rand = random_unit_vector(lens.d_model, seed=seed).to(device)
            p_rand = run_with_direction(e_rand)
            random_dps.append(float(p_rand[swap_answer_id] - p_base[swap_answer_id]))

        per_item.append(
            {
                "name": item["name"],
                "n_edit_positions": len(positions),
                "p_base_answer": float(p_base[answer_id]),
                "p_base_swap_answer": float(p_base[swap_answer_id]),
                "dp_swap_answer": float(p_swap[swap_answer_id] - p_base[swap_answer_id]),
                "dp_answer": float(p_swap[answer_id] - p_base[answer_id]),
                "dp_swap_answer_random_ctrl": random_dps,
                "swap_top1_hit": bool(int(p_swap.argmax()) == swap_answer_id),
            }
        )

    scored = [i for i in per_item if "skipped" not in i]
    n = len(scored)
    mean = lambda key: sum(i[key] for i in scored) / n  # noqa: E731
    all_random = [dp for i in scored for dp in i["dp_swap_answer_random_ctrl"]]
    return {
        "task": task.name,
        "band_layers": band_layers,
        "alpha": alpha,
        "rcond": rcond,
        "per_item": per_item,
        "metrics": {
            "n_scored": n,
            "mean_dp_swap_answer": round(mean("dp_swap_answer"), 5),
            "mean_dp_answer": round(mean("dp_answer"), 5),
            "mean_dp_swap_answer_random_ctrl": round(
                sum(all_random) / len(all_random), 5
            ),
            "swap_top1_rate": round(sum(i["swap_top1_hit"] for i in scored) / n, 4),
        },
    }
