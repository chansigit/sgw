import numpy as np
import torch
from torchgw._sampling import sample_pairs_from_plan, sample_pairs_gpu


def test_sample_pairs_returns_correct_count():
    rng = np.random.default_rng(0)
    T = rng.random((50, 40)).astype(np.float32)
    T /= T.sum()
    rows, cols = sample_pairs_from_plan(T, M=100)
    assert len(rows) == 100
    assert len(cols) == 100


def test_sample_pairs_indices_in_range():
    rng = np.random.default_rng(0)
    T = rng.random((50, 40)).astype(np.float32)
    T /= T.sum()
    rows, cols = sample_pairs_from_plan(T, M=200)
    assert all(0 <= r < 50 for r in rows)
    assert all(0 <= c < 40 for c in cols)


def test_sample_pairs_concentrates_on_high_mass():
    """If T has mass concentrated in one cell, sampling should reflect that."""
    T = np.zeros((10, 10), dtype=np.float32)
    T[3, 7] = 1.0  # all mass here
    rows, cols = sample_pairs_from_plan(T, M=50)
    assert all(r == 3 for r in rows)
    assert all(c == 7 for c in cols)


# ── GPU sampling tests ────────────────────────────────────────────────

def test_gpu_sample_returns_correct_count():
    T = torch.rand(50, 40)
    T /= T.sum()
    rows, cols = sample_pairs_gpu(T, M=100)
    assert len(rows) == 100
    assert len(cols) == 100


def test_gpu_sample_indices_in_range():
    T = torch.rand(50, 40)
    T /= T.sum()
    rows, cols = sample_pairs_gpu(T, M=200)
    assert all(0 <= r < 50 for r in rows)
    assert all(0 <= c < 40 for c in cols)


def test_gpu_sample_concentrates_on_high_mass():
    T = torch.zeros(10, 10)
    T[3, 7] = 1.0
    rows, cols = sample_pairs_gpu(T, M=50)
    assert all(r == 3 for r in rows)
    assert all(c == 7 for c in cols)


def test_gpu_sample_near_zero_plan():
    T = torch.zeros(10, 10)
    rows, cols = sample_pairs_gpu(T, M=20)
    assert len(rows) == 20
    assert len(cols) == 20
