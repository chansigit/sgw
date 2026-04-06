import sys
import numpy as np
import pytest

# Prevent TensorFlow from being auto-imported by the POT (ot) backend,
# which causes a segfault due to a LLVM version conflict on this system.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]


@pytest.fixture
def two_clusters():
    """Two well-separated 2D Gaussian clusters, 100 points each."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.3, size=(100, 2)),
        rng.normal(loc=[5, 5], scale=0.3, size=(100, 2)),
    ]).astype(np.float32)
    return X


@pytest.fixture
def two_datasets():
    """Two datasets: source (3 clusters) and target (3 clusters, rotated)."""
    rng = np.random.default_rng(42)
    centers_src = [[0, 0], [3, 0], [0, 3]]
    centers_tgt = [[0, 0], [0, 3], [-3, 0]]  # rotated
    X_src = np.vstack([rng.normal(loc=c, scale=0.3, size=(50, 2)) for c in centers_src]).astype(np.float32)
    X_tgt = np.vstack([rng.normal(loc=c, scale=0.3, size=(50, 2)) for c in centers_tgt]).astype(np.float32)
    return X_src, X_tgt
