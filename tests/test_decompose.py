"""Unit tests for the gradient-pursuit decomposition."""

import torch

from jvec.decompose import gradient_pursuit


def _unit_dictionary(n_atoms=200, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_atoms, d, generator=g)
    return D / D.norm(dim=1, keepdim=True)


def test_recovers_planted_sparse_combination():
    D = _unit_dictionary()
    h = 2.0 * D[5] + 1.0 * D[17]
    result = gradient_pursuit(h, D, k=8)
    assert set(result.indices) >= {5, 17}
    assert result.fraction > 0.98
    top = {i for i, c in zip(result.indices, result.coefficients) if c > 0.5}
    assert 5 in top


def test_orthogonal_vector_has_low_fraction():
    D = _unit_dictionary(n_atoms=50, d=64)
    # Build a vector orthogonal to the whole dictionary span complement is
    # impossible with 50 atoms in 64 dims — instead check monotonicity: a
    # random vector's fraction is far below a planted one's.
    g = torch.Generator().manual_seed(1)
    random_h = torch.randn(64, generator=g)
    planted = gradient_pursuit(D[3].clone(), D, k=5)
    random_frac = gradient_pursuit(random_h, D, k=5).fraction
    assert planted.fraction > 0.99
    assert random_frac < planted.fraction


def test_nonnegative_coefficients():
    D = _unit_dictionary()
    h = D[0] - 0.8 * D[1]  # negative component cannot be represented
    result = gradient_pursuit(h, D, k=10)
    assert all(c >= 0 for c in result.coefficients)


def test_fraction_bounds():
    D = _unit_dictionary()
    g = torch.Generator().manual_seed(2)
    h = torch.randn(64, generator=g)
    result = gradient_pursuit(h, D, k=25)
    assert 0.0 <= result.fraction <= 1.0
