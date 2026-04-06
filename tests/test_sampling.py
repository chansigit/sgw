import numpy as np
from torchgw._sampling import sample_pairs_from_plan


def test_sample_pairs_returns_correct_count():
    rng = np.random.default_rng(0)
    T = rng.random((50, 40)).astype(np.float32)
    T /= T.sum()
    pairs = sample_pairs_from_plan(T, M=100)
    assert len(pairs) == 100


def test_sample_pairs_indices_in_range():
    rng = np.random.default_rng(0)
    T = rng.random((50, 40)).astype(np.float32)
    T /= T.sum()
    pairs = sample_pairs_from_plan(T, M=200)
    rows, cols = zip(*pairs)
    assert all(0 <= r < 50 for r in rows)
    assert all(0 <= c < 40 for c in cols)


def test_sample_pairs_concentrates_on_high_mass():
    """If T has mass concentrated in one cell, sampling should reflect that."""
    T = np.zeros((10, 10), dtype=np.float32)
    T[3, 7] = 1.0  # all mass here
    pairs = sample_pairs_from_plan(T, M=50)
    rows, cols = zip(*pairs)
    assert all(r == 3 for r in rows)
    assert all(c == 7 for c in cols)
