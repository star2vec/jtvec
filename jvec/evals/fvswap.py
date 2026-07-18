"""Experiment 4: cross-basis FV swap (PREREGISTRATION §C).

On task-A 10-shot ICL prompts, replace the FV_A component of the residual
with FV_B, and measure whether the model's answer follows task B for the
same query. Task pairs are chosen so the same query input is valid under
both tasks (e.g. english-french <-> english-spanish: "dog" -> "chien" vs
"perro"), so "task-B correct output" is well defined per trial.

Conditions:
- ``lens_swap``: swap performed in lens coordinates — transport h with J_l,
  move the component along (J_l fv_A) onto (J_l fv_B), write the delta back
  with the truncated pseudoinverse (rcond), renormalize h. The J-specific
  manipulation.
- ``direct_swap``: same component move performed directly in residual space
  (no lens coordinates). Baseline that decides whether J-space coordinates
  specifically carry task identity or any basis works.
- ``random_target``: lens_swap toward a norm-matched random vector instead
  of fv_B (control).
- ``none``: clean run.

Edits at the final position of each band layer, norm-preserving throughout.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder, JacobianLens


class _FVSwapHook:
    """Move the h-component along direction a onto direction b at h[:, -1].

    In lens coordinates when (J, J_pinv) are given, directly in residual
    space when they are None. Norm-preserving.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, J=None, J_pinv=None):
        self.a = a / a.norm()
        self.b = b / b.norm()
        self.J, self.J_pinv = J, J_pinv

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden[:, -1].float().cpu()
        norms = h.norm(dim=-1, keepdim=True)
        if self.J is None:
            coeff = h @ self.a
            edited = h + coeff[:, None] * (self.b - self.a)
        else:
            t = h @ self.J.T
            t_a = self.J @ self.a
            t_a = t_a / t_a.norm()
            t_b = self.J @ self.b
            t_b = t_b / t_b.norm()
            coeff = t @ t_a
            delta_t = coeff[:, None] * (t_b - t_a)
            edited = h + delta_t @ self.J_pinv.T
        edited = edited * (norms / edited.norm(dim=-1, keepdim=True))
        hidden[:, -1] = edited.to(hidden.device, hidden.dtype)


@torch.no_grad()
def final_logits(model_j, prompt: str, hooks: dict[int, object]) -> torch.Tensor:
    final = model_j.n_layers - 1
    handles = [model_j.layers[l].register_forward_hook(h) for l, h in hooks.items()]
    try:
        ids = model_j.encode(prompt, max_length=1024)
        with ActivationRecorder(model_j.layers, at=[final]) as rec:
            model_j.forward(ids)
            residual = rec.activations[final][0, -1].detach()
        return model_j.unembed(residual.float()).float().cpu()
    finally:
        for h in handles:
            h.remove()


def make_swap_hooks(
    condition: str,
    band_layers: list[int],
    lens: JacobianLens,
    fv_a: torch.Tensor,
    fv_b: torch.Tensor,
    pinvs: dict[int, torch.Tensor],
    *,
    seed: int = 0,
) -> dict[int, object]:
    if condition == "none":
        return {}
    if condition == "random_target":
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(fv_b.shape, generator=g)
        fv_b = r * (fv_b.norm() / r.norm())
        condition = "lens_swap"
    hooks: dict[int, object] = {}
    for l in band_layers:
        if condition == "lens_swap":
            hooks[l] = _FVSwapHook(
                fv_a.float(), fv_b.float(),
                J=lens.jacobians[l].float().cpu(), J_pinv=pinvs[l],
            )
        elif condition == "direct_swap":
            hooks[l] = _FVSwapHook(fv_a.float(), fv_b.float())
        else:
            raise ValueError(condition)
    return hooks
