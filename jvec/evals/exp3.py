"""Experiment 3: metacognitive report + double ablation (report vs execution).

Report protocol (scaled to Pythia-410M): the model sees the SAME Todd-format
10-shot ICL context used for execution, but the tail is a forced-choice report
probe (e.g. "\nEach answer above is the question word's") scored over the
fixed set of task-label first-tokens. Report accuracy = argmax over the
candidate label tokens; chance = 1/n_tasks. A shuffled-context baseline
controls for label priors.

Ablation conditions (applied at the final position of each band layer via
forward hooks):
- ``jspace``: project out the span of the top-m J-lens-readout atoms at that
  layer/position (contents computed on the fly per layer);
- ``fv``:     project out the task's unit Todd-FV direction;
- ``sham_jspace`` / ``sham_fv``: matched counts of random unit directions;
- ``none``:   clean run.

Double dissociation = jspace hurts report >> execution while fv hurts
execution >> report, both relative to their shams.
"""

from __future__ import annotations

import torch
from jlens import ActivationRecorder, JacobianLens

#: One primary label word per task; scored by first token of " word".
REPORT_LABELS: dict[str, str] = {
    "antonym": "opposite",
    "english-french": "French",
    "english-spanish": "Spanish",
    "present-past": "past",
    "singular-plural": "plural",
    "capitalize": "uppercase",
    "landmark-country": "country",
    "person-sport": "sport",
}

REPORT_PROBES: dict[str, str] = {
    "P1": "\nEach answer above is the question word's",
    "P2": "\nQuestion: How does each answer relate to its question?\nAnswer: It is the question word's",
    "P3": "\nThe rule of the list above: the answer is always the question word's",
}


def label_token_ids(tokenizer) -> dict[str, int]:
    return {
        task: tokenizer(" " + word, add_special_tokens=False).input_ids[0]
        for task, word in REPORT_LABELS.items()
    }


class MatchedNoiseHook:
    """Magnitude-matched sham: displace h[:, -1] along a random direction by a
    target *relative* magnitude, then rescale to the original norm.

    This matches a real ablation's per-layer relative displacement
    (PREREGISTRATION §A1 criterion iv) without sharing its content direction.
    """

    def __init__(self, rel_delta: float, d_model: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(d_model, generator=g)
        self.direction = v / v.norm()
        self.rel_delta = rel_delta

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden[:, -1].float().cpu()
        norms = h.norm(dim=-1, keepdim=True)
        h = h + self.rel_delta * norms * self.direction
        h = h * (norms / h.norm(dim=-1, keepdim=True))
        hidden[:, -1] = h.to(hidden.device, hidden.dtype)


class ProjectOutHook:
    """h[:, -1] -= P h[:, -1], where P projects onto span(directions)."""

    def __init__(self, directions: torch.Tensor):
        # Orthonormalize once: [m, d] -> Q [d, r]
        Q, _ = torch.linalg.qr(directions.T.float().cpu())
        self.Q = Q

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden[:, -1].float().cpu()
        h = h - (h @ self.Q) @ self.Q.T
        hidden[:, -1] = h.to(hidden.device, hidden.dtype)


class JSpaceAblateHook:
    """Project out the top-m J-lens-readout atoms at this layer, final position.

    Contents are dynamic: read h through J_l, take the top-m vocabulary
    tokens, and remove the span of their dictionary atoms (rows of W_U J_l).
    """

    def __init__(self, J: torch.Tensor, W_U: torch.Tensor, m: int):
        self.J = J.float().cpu()
        self.W_U = W_U.float().cpu()
        self.m = m

    def __call__(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        h = hidden[:, -1].float().cpu()  # [batch, d]
        transported = h @ self.J.T
        logits = transported @ self.W_U.T  # [batch, vocab]
        for b in range(h.shape[0]):
            top = logits[b].topk(self.m).indices
            atoms = self.W_U[top] @ self.J  # [m, d] rows of W_U J_l
            Q, _ = torch.linalg.qr(atoms.T)
            h[b] = h[b] - (h[b] @ Q) @ Q.T
        hidden[:, -1] = h.to(hidden.device, hidden.dtype)


def make_hooks(
    condition: str,
    band_layers: list[int],
    lens: JacobianLens,
    W_U: torch.Tensor,
    fv: torch.Tensor,
    *,
    m_top: int,
    seed: int = 0,
) -> dict[int, object]:
    """Per-layer hook objects for a condition (empty dict = clean run)."""
    if condition == "none":
        return {}
    g = torch.Generator().manual_seed(seed)
    hooks: dict[int, object] = {}
    for l in band_layers:
        if condition == "jspace":
            hooks[l] = JSpaceAblateHook(lens.jacobians[l], W_U, m_top)
        elif condition == "sham_jspace":
            dirs = torch.randn(m_top, W_U.shape[1], generator=g)
            hooks[l] = ProjectOutHook(dirs)
        elif condition == "fv":
            hooks[l] = ProjectOutHook(fv.reshape(1, -1))
        elif condition == "sham_fv":
            hooks[l] = ProjectOutHook(torch.randn(1, W_U.shape[1], generator=g))
        else:
            raise ValueError(condition)
    return hooks


@torch.no_grad()
def final_logits_under(model_j, prompt: str, hooks: dict[int, object]) -> torch.Tensor:
    """Model's final-position logits with the given per-layer hooks active."""
    final = model_j.n_layers - 1
    handles = [
        model_j.layers[l].register_forward_hook(hook) for l, hook in hooks.items()
    ]
    try:
        ids = model_j.encode(prompt, max_length=1024)
        with ActivationRecorder(model_j.layers, at=[final]) as rec:
            model_j.forward(ids)
            residual = rec.activations[final][0, -1].detach()
        return model_j.unembed(residual.float()).float().cpu()
    finally:
        for h in handles:
            h.remove()
