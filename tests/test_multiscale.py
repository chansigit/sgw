import numpy as np
import torch
import pytest
from sgw._multiscale import fps_downsample, upsample_plan


def test_fps_downsample_shapes():
    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.normal(size=(100, 5)).astype(np.float32))
    indices, assignments = fps_downsample(X, n=20)
    assert indices.shape == (20,)
    assert assignments.shape == (100,)
    assert torch.all(assignments >= 0)
    assert torch.all(assignments < 20)


def test_fps_downsample_covers_selected():
    """Selected points should be assigned to themselves."""
    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.normal(size=(50, 3)).astype(np.float32))
    indices, assignments = fps_downsample(X, n=10)
    for i, idx in enumerate(indices):
        assert assignments[idx.item()].item() == i


def test_fps_downsample_n_equals_N():
    """When n == N, all points are selected."""
    X = torch.randn(20, 3)
    indices, assignments = fps_downsample(X, n=20)
    assert indices.shape == (20,)
    assert len(torch.unique(indices)) == 20


def test_upsample_plan_shapes():
    T_coarse = torch.rand(10, 12)
    T_coarse /= T_coarse.sum()
    assign_src = torch.randint(0, 10, (50,))
    assign_tgt = torch.randint(0, 12, (60,))
    p = torch.ones(50) / 50
    q = torch.ones(60) / 60

    T_fine = upsample_plan(T_coarse, assign_src, assign_tgt, p, q)
    assert T_fine.shape == (50, 60)
    assert torch.all(T_fine >= 0)


def test_upsample_plan_marginals():
    """Upsampled plan should approximately match target marginals."""
    T_coarse = torch.rand(10, 12, dtype=torch.float64)
    T_coarse /= T_coarse.sum()
    assign_src = torch.randint(0, 10, (50,))
    assign_tgt = torch.randint(0, 12, (60,))
    p = torch.ones(50, dtype=torch.float64) / 50
    q = torch.ones(60, dtype=torch.float64) / 60

    T_fine = upsample_plan(T_coarse, assign_src, assign_tgt, p, q)
    row_sums = T_fine.sum(dim=1)
    col_sums = T_fine.sum(dim=0)
    assert torch.allclose(row_sums, p, atol=1e-6)
    assert torch.allclose(col_sums, q, atol=1e-6)
