"""S1 concept-direction model-side ops (EXP-M5-1 / M5_SPEC §M5.1).

Two forward-only operations on a jlens LensModel:

- ``answer_states``: residual at the final position of each band layer, per
  context — the raw material of the mean-difference direction.
- ``injected_final_probs``: final-position softmax with a fixed residual
  direction added at the final position of each band layer, norm-preserving
  (the M1 norm-preservation idiom from jvec.evals.swap). No lens-coordinate
  write-back: the concept direction is already residual-space, so the prereg's
  truncated-pinv clause is vacuous (see jtvec.concept_gate design note).

Pure numeric logic (ladder, cosine, verdict, controls) lives in
jtvec.concept_gate; this module holds only model forward/hook work.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder


@torch.no_grad()
def answer_states(model_j, prompts: list[str], band_layers: list[int],
                  *, max_length: int = 1024) -> dict[int, torch.Tensor]:
    """{layer: [n_prompts, d_model]} residuals at the final position of each
    band layer, on CPU fp32."""
    acc: dict[int, list[torch.Tensor]] = {l: [] for l in band_layers}
    for prompt in prompts:
        ids = model_j.encode(prompt, max_length=max_length)
        with ActivationRecorder(model_j.layers, at=band_layers) as rec:
            model_j.forward(ids)
            for l in band_layers:
                acc[l].append(rec.activations[l][0, -1].detach().float().cpu())
    return {l: torch.stack(v) for l, v in acc.items()}


class _AddHook:
    """Add a fixed residual direction at the final position, norm-preserving."""

    def __init__(self, delta: torch.Tensor) -> None:
        self.delta = delta.float()

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden[:, -1].float().cpu()
        norms = h.norm(dim=-1, keepdim=True)
        edited = h + self.delta
        edited = edited * (norms / edited.norm(dim=-1, keepdim=True))
        hidden[:, -1] = edited.to(hidden.device, hidden.dtype)


@torch.no_grad()
def injected_final_probs(model_j, prompt: str, deltas: dict[int, torch.Tensor],
                         *, max_length: int = 1024) -> torch.Tensor:
    """Final-position softmax over the vocab with per-band-layer residual
    additions (``deltas`` empty -> the unhooked forward)."""
    final = model_j.n_layers - 1
    handles = [model_j.layers[l].register_forward_hook(_AddHook(d))
               for l, d in deltas.items()]
    try:
        ids = model_j.encode(prompt, max_length=max_length)
        with ActivationRecorder(model_j.layers, at=[final]) as rec:
            model_j.forward(ids)
            resid = rec.activations[final][0, -1].detach()
        return torch.softmax(model_j.unembed(resid.float()).float().cpu(), dim=-1)
    finally:
        for h in handles:
            h.remove()
