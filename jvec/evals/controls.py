"""Random-direction controls for the probing and swap evals."""

from __future__ import annotations

import torch


def random_matrices_like(
    jacobians: dict[int, torch.Tensor], seed: int
) -> dict[int, torch.Tensor]:
    """Per-layer Gaussian random matrices, Frobenius-norm-matched to ``J_l``.

    A lens readout through these answers: "would *any* norm-matched linear map
    into the final basis surface the token?"
    """
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for layer, J in jacobians.items():
        R = torch.randn(J.shape, generator=generator, dtype=torch.float32)
        out[layer] = R * (J.norm() / R.norm())
    return out


def random_unit_vector(d_model: int, seed: int) -> torch.Tensor:
    """A random unit direction in the final-layer basis (for swap controls)."""
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=generator, dtype=torch.float32)
    return v / v.norm()
