"""Experiment 2: sparse decomposition against the J-lens dictionary.

Per the workspace paper: the J-lens dictionary at layer l is the rows of
``W_U @ J_l`` — one direction in layer-l residual space per vocabulary token —
and a residual vector h is decomposed by gradient pursuit into a sparse
NON-NEGATIVE combination of at most k atoms (paper: k <= 25). The J-space
fraction is the share of squared norm captured by the reconstruction.

The paper does not specify atom normalization; we L2-normalize atoms (standard
for pursuit selection) — coefficients are reported on unit atoms.

Everything here runs on CPU fp32 (cheap: k matmuls of [vocab, d]) so it can
coexist with extraction jobs on the GPU/MPS device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jlens import JacobianLens


def jlens_dictionary(lens: JacobianLens, W_U: torch.Tensor, layer: int) -> torch.Tensor:
    """[vocab, d] dictionary at ``layer``: L2-normalized rows of W_U @ J_l."""
    D = W_U.float().cpu() @ lens.jacobians[layer].float().cpu()
    return D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)


@dataclass
class Decomposition:
    indices: list[int]      # selected atom (token) ids, in selection order
    coefficients: list[float]
    fraction: float         # 1 - ||residual||^2 / ||h||^2


@torch.no_grad()
def gradient_pursuit(h: torch.Tensor, D: torch.Tensor, k: int = 25) -> Decomposition:
    """Sparse non-negative gradient pursuit (Blumensath & Davies style).

    Args:
        h: [d] vector to decompose.
        D: [n_atoms, d] dictionary with unit-norm rows.
        k: maximum number of atoms.
    """
    h = h.float().cpu()
    residual = h.clone()
    support: list[int] = []
    coeffs = torch.zeros(0)

    for _ in range(k):
        correlations = D @ residual
        if support:
            correlations[torch.tensor(support)] = -torch.inf
        best = int(correlations.argmax())
        if correlations[best] <= 0:
            break
        support.append(best)
        D_S = D[torch.tensor(support)]           # [s, d]
        coeffs = torch.cat([coeffs, torch.zeros(1)])
        # Gradient step restricted to the support, then project to a >= 0.
        gradient = D_S @ residual                 # [s]
        step_dir = D_S.T @ gradient               # [d]
        denom = float(step_dir @ step_dir)
        if denom <= 0:
            break
        alpha = float(residual @ step_dir) / denom
        coeffs = (coeffs + alpha * gradient).clamp_min(0.0)
        residual = h - D_S.T @ coeffs

    fraction = 1.0 - float(residual @ residual) / max(float(h @ h), 1e-12)
    return Decomposition(
        indices=support,
        coefficients=[float(c) for c in coeffs],
        fraction=max(fraction, 0.0),
    )
