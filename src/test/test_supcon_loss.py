"""
Tests for src/train/supcon_loss.py.
Run with: PYTHONPATH=$(pwd) python -m pytest src/test/test_supcon_loss.py
"""
import types

import numpy as np
import torch
import torch.nn.functional as F

from src.train.supcon_loss import SupConSampler, ordinal_supcon_loss


def _reference_supcon(emb, y, tau):
    """Khosla et al. SupCon (L_out), written out longhand."""
    z = F.normalize(emb, dim=1)
    n = len(y)
    total, count = 0.0, 0
    for i in range(n):
        others = [j for j in range(n) if j != i]
        denom = torch.logsumexp(torch.stack([z[i] @ z[j] / tau for j in others]), 0)
        pos = [j for j in others if y[j] == y[i]]
        if not pos:
            continue
        total += -sum((z[i] @ z[j] / tau - denom) for j in pos) / len(pos)
        count += 1
    return total / count


def test_sigma_zero_is_standard_supcon():
    torch.manual_seed(0)
    emb, y = torch.randn(12, 8), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
    a = ordinal_supcon_loss(emb, y, tau=0.2, sigma=0)
    b = _reference_supcon(emb, y, tau=0.2)
    assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_clustered_embedding_scores_lower_than_shuffled():
    torch.manual_seed(0)
    y = torch.arange(4).repeat_interleave(8)
    centers = F.normalize(torch.randn(4, 16), dim=1)
    clustered = centers[y] + 0.05 * torch.randn(32, 16)
    shuffled = clustered[torch.randperm(32)]
    for sigma in (0.0, 0.75):
        assert ordinal_supcon_loss(clustered, y, sigma=sigma) < ordinal_supcon_loss(shuffled, y, sigma=sigma)


def test_ordinal_weighting_prefers_ordered_layout():
    # classes on a line in the right order vs the same points with classes
    # assigned out of order: the ordinal loss must prefer the ordered one,
    # plain SupCon (sigma=0) is nearly indifferent to the order.
    torch.manual_seed(0)
    y = torch.arange(4).repeat_interleave(6)
    pos = torch.tensor([0.0, 1.0, 2.0, 3.0])
    def layout(order):
        p = pos[torch.tensor(order)][y]
        e = torch.stack([torch.cos(p * 0.6), torch.sin(p * 0.6)], 1)
        return e + 0.02 * torch.randn_like(e)
    ordered, scrambled = layout([0, 1, 2, 3]), layout([0, 2, 3, 1])
    assert ordinal_supcon_loss(ordered, y, sigma=0.75) < ordinal_supcon_loss(scrambled, y, sigma=0.75)


def test_sampler_is_class_balanced_with_two_views():
    pool = types.SimpleNamespace(pool_x=torch.rand(30, 3, 32, 32),
                                 pool_y=np.array([0] * 3 + [1] * 7 + [2] * 10 + [3] * 10),
                                 n_classes=4)
    x, y = SupConSampler(pool, per_class=5, seed=0).sample()
    assert x.shape == (40, 3, 32, 32)
    assert torch.equal(torch.bincount(y), torch.tensor([10, 10, 10, 10]))
    assert torch.isfinite(x).all()
