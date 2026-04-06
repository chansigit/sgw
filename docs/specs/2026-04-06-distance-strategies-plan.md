# Distance Strategies + Fused GW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three distance strategies (precomputed, dijkstra, spectral) and Fused GW to `sampled_gw`, with torch tensor as primary I/O type.

**Architecture:** Extract distance logic from `_solver.py` into a provider pattern in `_distances.py`. Each provider implements `get_distances(src_idx, tgt_idx) -> (D_X, D_Y)`. The solver picks the provider based on `distance_mode`, then blends with `C_linear` if `fgw_alpha > 0`.

**Tech Stack:** PyTorch, scipy (Dijkstra, eigsh), sklearn (kNN), numpy

**Spec:** `docs/specs/2026-04-06-distance-strategies-design.md`

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `sgw/_distances.py` | Distance provider classes | **New** |
| `sgw/_solver.py` | Main solver | **Modify** — new params, provider dispatch, FGW blending, tensor I/O |
| `sgw/_graph.py` | kNN graph builder | No change |
| `sgw/_sampling.py` | Anchor pair sampling | No change |
| `sgw/_embedding.py` | Joint embedding | No change |
| `sgw/_utils.py` | Device utils | No change |
| `sgw/__init__.py` | Public API | No change |
| `tests/test_distances.py` | Unit tests for providers | **New** |
| `tests/test_solver.py` | Solver tests | **Modify** — tensor output, new modes, FGW |
| `tests/test_spiral_swissroll.py` | Integration test | **Modify** — tensor output, remove graph params |
| `tests/test_embedding.py` | Embedding tests | **Modify** — tensor output, remove graph params |
| `tests/conftest.py` | Fixtures | No change |

---

## Task 1: Create `_distances.py` with `DijkstraProvider`

Extract the existing Dijkstra logic from `_solver.py` into a provider class. This is a pure refactor — no behavior change yet.

**Files:**
- Create: `sgw/_distances.py`
- Create: `tests/test_distances.py`

- [ ] **Step 1: Write failing test for DijkstraProvider**

```python
# tests/test_distances.py
import numpy as np
import torch
import pytest
from sgw._graph import build_knn_graph
from sgw._distances import DijkstraProvider


def test_dijkstra_provider_shapes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    Y = rng.normal(size=(60, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = DijkstraProvider(g_x, g_y)
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3, 7, 9])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (50, 3)
    assert D_Y.shape == (60, 4)
    assert D_X.dtype == torch.float32
    assert D_Y.dtype == torch.float32
    assert D_X.device.type == "cpu"


def test_dijkstra_provider_nonnegative():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(X.copy(), k=10)

    provider = DijkstraProvider(g_x, g_y)
    D_X, D_Y = provider.get_distances(np.array([0, 1]), np.array([0, 1]), torch.device("cpu"))

    assert torch.all(D_X >= 0)
    assert torch.all(D_Y >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distances.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 3: Implement DijkstraProvider**

```python
# sgw/_distances.py
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_DIJKSTRA_PARALLEL_THRESHOLD = 64


def _batch_dijkstra(graph: csr_matrix, sources: np.ndarray, parallel: bool) -> np.ndarray:
    """Run Dijkstra from multiple sources. Returns (len(sources), N) array."""
    if not parallel or len(sources) < _DIJKSTRA_PARALLEL_THRESHOLD:
        return dijkstra(csgraph=graph, directed=False, indices=sources)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(dijkstra)(graph, directed=False, indices=int(s)) for s in sources
    )
    return np.vstack(results)


class DijkstraProvider:
    """Compute distances on-the-fly via Dijkstra on kNN graphs."""

    def __init__(self, graph_source: csr_matrix, graph_target: csr_matrix):
        self.graph_source = graph_source
        self.graph_target = graph_target
        self._parallel = max(graph_source.shape[0], graph_target.shape[0]) > 1000

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (D_X, D_Y) as float32 tensors on device.

        D_X : (N, len(src_indices))
        D_Y : (K, len(tgt_indices))
        """
        unique_src, src_inv = np.unique(src_indices, return_inverse=True)
        unique_tgt, tgt_inv = np.unique(tgt_indices, return_inverse=True)

        D_src_all = _batch_dijkstra(self.graph_source, unique_src, self._parallel)
        D_tgt_all = _batch_dijkstra(self.graph_target, unique_tgt, self._parallel)

        D_X = torch.from_numpy(D_src_all).float().to(device).T[:, src_inv]
        D_Y = torch.from_numpy(D_tgt_all).float().to(device).T[:, tgt_inv]

        return D_X, D_Y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distances.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_distances.py tests/test_distances.py
git commit -m "feat: add DijkstraProvider in _distances.py"
```

---

## Task 2: Add `PrecomputedProvider`

**Files:**
- Modify: `sgw/_distances.py`
- Modify: `tests/test_distances.py`

- [ ] **Step 1: Write failing tests for PrecomputedProvider**

Append to `tests/test_distances.py`:

```python
from sgw._distances import PrecomputedProvider


def test_precomputed_provider_from_matrices():
    """User passes dist_source and dist_target directly."""
    rng = np.random.default_rng(42)
    C_X = rng.random((40, 40)).astype(np.float32)
    C_Y = rng.random((50, 50)).astype(np.float32)

    provider = PrecomputedProvider(
        dist_source=torch.from_numpy(C_X),
        dist_target=torch.from_numpy(C_Y),
    )
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (40, 3)
    assert D_Y.shape == (50, 2)
    # Should be exact column indexing
    torch.testing.assert_close(D_X[:, 0], torch.from_numpy(C_X[:, 0]))
    torch.testing.assert_close(D_Y[:, 1], torch.from_numpy(C_Y[:, 3]))


def test_precomputed_provider_from_graphs():
    """User doesn't pass dist matrices — provider computes all-pairs Dijkstra."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    Y = rng.normal(size=(35, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = PrecomputedProvider(graph_source=g_x, graph_target=g_y)
    D_X, D_Y = provider.get_distances(np.array([0, 1]), np.array([0, 1]), torch.device("cpu"))

    assert D_X.shape == (30, 2)
    assert D_Y.shape == (35, 2)
    assert torch.all(torch.isfinite(D_X))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distances.py::test_precomputed_provider_from_matrices -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Implement PrecomputedProvider**

Add to `sgw/_distances.py`:

```python
class PrecomputedProvider:
    """Look up distances from precomputed full pairwise matrices."""

    def __init__(
        self,
        dist_source: torch.Tensor | None = None,
        dist_target: torch.Tensor | None = None,
        graph_source: csr_matrix | None = None,
        graph_target: csr_matrix | None = None,
    ):
        if dist_source is not None and dist_target is not None:
            self.C_X = dist_source.float()
            self.C_Y = dist_target.float()
        elif graph_source is not None and graph_target is not None:
            C_X_np = dijkstra(csgraph=graph_source, directed=False)
            C_Y_np = dijkstra(csgraph=graph_target, directed=False)
            self.C_X = torch.from_numpy(C_X_np).float()
            self.C_Y = torch.from_numpy(C_Y_np).float()
        else:
            raise ValueError(
                "PrecomputedProvider requires either (dist_source, dist_target) "
                "or (graph_source, graph_target)"
            )

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D_X = self.C_X[:, src_indices].to(device)
        D_Y = self.C_Y[:, tgt_indices].to(device)
        return D_X, D_Y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_distances.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_distances.py tests/test_distances.py
git commit -m "feat: add PrecomputedProvider in _distances.py"
```

---

## Task 3: Add `SpectralProvider`

**Files:**
- Modify: `sgw/_distances.py`
- Modify: `tests/test_distances.py`

- [ ] **Step 1: Write failing tests for SpectralProvider**

Append to `tests/test_distances.py`:

```python
from sgw._distances import SpectralProvider


def test_spectral_provider_shapes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    Y = rng.normal(size=(60, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = SpectralProvider(g_x, g_y, spectral_dim=10)
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3, 7, 9])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (50, 3)
    assert D_Y.shape == (60, 4)
    assert D_X.dtype == torch.float32
    assert torch.all(D_X >= 0)
    assert torch.all(D_Y >= 0)


def test_spectral_provider_self_distance_zero():
    """Distance from a point to itself should be zero."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)

    provider = SpectralProvider(g_x, g_x, spectral_dim=10)
    D_X, _ = provider.get_distances(np.array([5]), np.array([0]), torch.device("cpu"))

    assert D_X[5, 0].item() == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distances.py::test_spectral_provider_shapes -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Implement SpectralProvider**

Add to `sgw/_distances.py`:

```python
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def _spectral_embed(graph: csr_matrix, dim: int) -> np.ndarray:
    """Compute spectral embedding of a connected graph.

    Returns (N, dim) array of coordinates from the smallest
    non-trivial eigenvectors of the normalized Laplacian.
    """
    W = graph.copy().astype(np.float64)
    W.data = 1.0 / (W.data + 1e-10)
    W = W.maximum(W.T)
    degrees = np.asarray(W.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1.0
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_norm = diags(np.ones(W.shape[0])) - D_inv_sqrt @ W @ D_inv_sqrt

    # Smallest eigenvalues of normalized Laplacian (skip first = trivial)
    n_eigvecs = min(dim + 1, W.shape[0] - 1)
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigvecs, which="SM")

    # Sort by eigenvalue ascending, skip the first (constant) eigenvector
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    Z = eigenvectors[:, 1 : dim + 1]  # (N, dim)

    return Z.astype(np.float32)


class SpectralProvider:
    """Approximate graph distances via Euclidean distance in spectral embedding space."""

    def __init__(self, graph_source: csr_matrix, graph_target: csr_matrix, spectral_dim: int = 50):
        self.Z_X = torch.from_numpy(_spectral_embed(graph_source, spectral_dim))
        self.Z_Y = torch.from_numpy(_spectral_embed(graph_target, spectral_dim))

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Z_X = self.Z_X.to(device)
        Z_Y = self.Z_Y.to(device)

        # (N, M): Euclidean distance from each point to each sampled anchor
        D_X = torch.cdist(Z_X, Z_X[src_indices])
        D_Y = torch.cdist(Z_Y, Z_Y[tgt_indices])

        return D_X, D_Y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_distances.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_distances.py tests/test_distances.py
git commit -m "feat: add SpectralProvider in _distances.py"
```

---

## Task 4: Refactor `_solver.py` — provider dispatch and tensor I/O

Replace the inline Dijkstra code with provider dispatch, change I/O to tensor, remove `graph_source`/`graph_target` params.

**Files:**
- Modify: `sgw/_solver.py`
- Modify: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests for new solver API**

Replace `tests/test_solver.py` with:

```python
import numpy as np
import torch
import pytest
from sgw._solver import sampled_gw


def test_sampled_gw_returns_tensor(two_datasets):
    """Output should be a torch.Tensor in v0.2.0."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_sampled_gw_accepts_tensor_input(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        torch.from_numpy(X_src), torch.from_numpy(X_tgt),
        s_shared=50, M=30, max_iter=10,
    )
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)


def test_sampled_gw_mass_bounded(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert T.sum().item() <= 1.0 + 1e-6


def test_distance_mode_dijkstra(two_datasets):
    """Default mode should work as before."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, distance_mode="dijkstra", s_shared=50, M=30, max_iter=10)
    assert T.shape == (150, 150)


def test_distance_mode_precomputed_with_matrices(two_datasets):
    """Pass precomputed distance matrices, no X needed."""
    from sgw._graph import build_knn_graph
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    X_src, X_tgt = two_datasets
    g_src = build_knn_graph(X_src, k=10)
    g_tgt = build_knn_graph(X_tgt, k=10)
    C_X = sp_dijkstra(csgraph=g_src, directed=False).astype(np.float32)
    C_Y = sp_dijkstra(csgraph=g_tgt, directed=False).astype(np.float32)

    T = sampled_gw(
        dist_source=C_X, dist_target=C_Y,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_distance_mode_precomputed_without_matrices(two_datasets):
    """Pass X, let solver compute all-pairs Dijkstra internally."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_distance_mode_spectral(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="spectral",
        spectral_dim=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_invalid_distance_mode(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="distance_mode"):
        sampled_gw(X_src, X_tgt, distance_mode="invalid", M=30, max_iter=10)


def test_precomputed_missing_params():
    """precomputed mode with no dist and no X should raise."""
    with pytest.raises(ValueError):
        sampled_gw(distance_mode="precomputed", M=30, max_iter=10)


def test_sampled_gw_identity_alignment():
    """Aligning identical data should produce near-diagonal transport plan."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5)).astype(np.float32)
    np.random.seed(2)
    T = sampled_gw(X, X.copy(), s_shared=30, M=20, max_iter=50, k=5,
                   alpha=0.5, epsilon=0.01)
    row_argmax = T.argmax(dim=1).numpy()
    diagonal_fraction = np.mean(row_argmax == np.arange(30))
    assert diagonal_fraction > 0.5, f"Only {diagonal_fraction:.0%} on diagonal"


def test_sampled_gw_log_returns_tuple(two_datasets):
    X_src, X_tgt = two_datasets
    result = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10, log=True)
    assert isinstance(result, tuple)
    T, log_dict = result
    assert isinstance(T, torch.Tensor)
    assert "err_list" in log_dict
    assert "n_iter" in log_dict
    assert "gw_cost" in log_dict


def test_public_import():
    from sgw import sampled_gw, build_knn_graph, joint_embedding
    assert callable(sampled_gw)
    assert callable(build_knn_graph)
    assert callable(joint_embedding)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_solver.py -v`
Expected: multiple FAILs (tensor output, new params not accepted)

- [ ] **Step 3: Refactor `_solver.py`**

Key changes to `sgw/_solver.py`:

1. Remove `_batch_dijkstra` and `_DIJKSTRA_PARALLEL_THRESHOLD` (moved to `_distances.py`).
2. Remove `graph_source` and `graph_target` parameters.
3. Add new parameters: `distance_mode`, `dist_source`, `dist_target`, `spectral_dim`.
4. Add input coercion helper at top of function.
5. Create provider based on `distance_mode`.
6. Replace inline Dijkstra block with `provider.get_distances()`.
7. Return `torch.Tensor` instead of numpy.

The full updated `sampled_gw` function:

```python
from sgw._distances import DijkstraProvider, PrecomputedProvider, SpectralProvider

def _to_tensor(x):
    """Convert numpy array or tensor to torch.Tensor. Pass through None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


def sampled_gw(
    X_source: np.ndarray | torch.Tensor | None = None,
    X_target: np.ndarray | torch.Tensor | None = None,
    p: np.ndarray | torch.Tensor | None = None,
    q: np.ndarray | torch.Tensor | None = None,
    *,
    distance_mode: str = "dijkstra",
    dist_source: np.ndarray | torch.Tensor | None = None,
    dist_target: np.ndarray | torch.Tensor | None = None,
    spectral_dim: int = 50,
    fgw_alpha: float = 0.0,
    C_linear: np.ndarray | torch.Tensor | None = None,
    s_shared: int | None = None,
    M: int = 50,
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1e-5,
    epsilon: float = 0.001,
    k: int = 30,
    min_iter_before_converge: int = 50,
    device: torch.device | None = None,
    verbose: bool = False,
    verbose_every: int = 20,
    log: bool = False,
    differentiable: bool = False,
    semi_relaxed: bool = False,
    rho: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    # docstring omitted for brevity — update to document new params

    if device is None:
        device = get_device()

    # ── Input coercion ──
    X_source = _to_tensor(X_source)
    X_target = _to_tensor(X_target)
    dist_source = _to_tensor(dist_source)
    dist_target = _to_tensor(dist_target)
    C_linear_t = _to_tensor(C_linear)

    # ── Infer N, K ──
    if dist_source is not None and dist_target is not None:
        N, K = dist_source.shape[0], dist_target.shape[0]
    elif X_source is not None and X_target is not None:
        N, K = X_source.shape[0], X_target.shape[0]
    elif C_linear_t is not None:
        N, K = C_linear_t.shape
    else:
        raise ValueError(
            "Cannot infer N and K. Provide (X_source, X_target), "
            "(dist_source, dist_target), or C_linear."
        )

    # ── Validate distance_mode ──
    if distance_mode not in ("precomputed", "dijkstra", "spectral"):
        raise ValueError(
            f"distance_mode must be 'precomputed', 'dijkstra', or 'spectral', "
            f"got '{distance_mode}'"
        )

    # ── Validate FGW params ──
    if fgw_alpha > 0 and C_linear_t is None:
        raise ValueError("fgw_alpha > 0 requires C_linear")

    # ── Build distance provider ──
    if fgw_alpha >= 1.0:
        # Pure Wasserstein — no structural distances needed
        provider = None
    elif distance_mode == "precomputed":
        if dist_source is not None and dist_target is not None:
            provider = PrecomputedProvider(
                dist_source=dist_source,
                dist_target=dist_target,
            )
        elif X_source is not None and X_target is not None:
            graphs = (
                build_knn_graph(X_source.cpu().numpy(), k=k),
                build_knn_graph(X_target.cpu().numpy(), k=k),
            )
            provider = PrecomputedProvider(
                graph_source=graphs[0],
                graph_target=graphs[1],
            )
        else:
            raise ValueError(
                "distance_mode='precomputed' requires (dist_source, dist_target) "
                "or (X_source, X_target)"
            )
    elif distance_mode == "dijkstra":
        if X_source is None or X_target is None:
            raise ValueError("distance_mode='dijkstra' requires X_source and X_target")
        graph_source = build_knn_graph(X_source.cpu().numpy(), k=k)
        graph_target = build_knn_graph(X_target.cpu().numpy(), k=k)
        provider = DijkstraProvider(graph_source, graph_target)
    elif distance_mode == "spectral":
        if X_source is None or X_target is None:
            raise ValueError("distance_mode='spectral' requires X_source and X_target")
        graph_source = build_knn_graph(X_source.cpu().numpy(), k=k)
        graph_target = build_knn_graph(X_target.cpu().numpy(), k=k)
        provider = SpectralProvider(graph_source, graph_target, spectral_dim=spectral_dim)

    s_eff = s_shared if s_shared is not None and s_shared <= min(N, K) else min(N, K)

    # ── Marginals (float64 for Sinkhorn) ──
    if p is not None:
        p_real = _to_tensor(p).to(device=device, dtype=torch.float64)
    else:
        p_real = torch.ones(N, device=device, dtype=torch.float64) / N
    if q is not None:
        q_real = _to_tensor(q).to(device=device, dtype=torch.float64)
    else:
        q_real = torch.ones(K, device=device, dtype=torch.float64) / K

    m_frac = s_eff / max(N, K)
    slack_p = max(q_real.sum().item() - m_frac, 1e-10)
    slack_q = max(p_real.sum().item() - m_frac, 1e-10)
    p_aug = torch.cat([p_real, torch.tensor([slack_p], device=device, dtype=torch.float64)])
    q_aug = torch.cat([q_real, torch.tensor([slack_q], device=device, dtype=torch.float64)])

    T_real = torch.outer(p_real, q_real)

    initial_reg = epsilon if epsilon > 0 else 1e-4
    final_reg = min(5e-4, initial_reg)
    decay = (final_reg / initial_reg) ** (1 / max(1, max_iter))

    # ── Prepare FGW linear cost ──
    if C_linear_t is not None and fgw_alpha > 0:
        C_lin_device = C_linear_t.float().to(device)
    else:
        C_lin_device = None

    err_list = []
    gw_cost_val = 0.0
    n_iter = 0
    ctx = torch.no_grad() if not differentiable else torch.enable_grad()
    _sinkhorn_fn = _sinkhorn_differentiable if differentiable else _sinkhorn_torch

    with ctx:
        for i in range(max_iter):
            current_reg = initial_reg * (decay ** i)
            T_prev = T_real.clone()

            # ── Sample anchor pairs ──
            T_cpu = T_real.detach().cpu().numpy()
            pairs = sample_pairs_from_plan(T_cpu, M)
            j_left, l_target = zip(*pairs)
            j_left = np.asarray(j_left)
            l_target = np.asarray(l_target)

            # ── Get distances ──
            if provider is not None:
                D_left, D_tgt = provider.get_distances(j_left, l_target, device)

                # Handle inf and normalize
                for D in [D_left, D_tgt]:
                    inf_mask = torch.isinf(D)
                    if torch.any(inf_mask):
                        D[inf_mask] = D[~inf_mask].max() * 1.5
                    mx = D.max()
                    if mx > 0:
                        D /= mx

                # GW cost matrix (float32 on GPU)
                term_A = torch.mean(D_left ** 2, dim=1, keepdim=True)
                term_C = torch.mean(D_tgt ** 2, dim=1, keepdim=True).T
                term_B = -2 * (D_left @ D_tgt.T) / M
                Lambda_gw = term_A + term_B + term_C
            else:
                Lambda_gw = None

            # ── Fused GW blending ──
            if fgw_alpha >= 1.0:
                Lambda = C_lin_device
            elif fgw_alpha > 0:
                Lambda = (1 - fgw_alpha) * Lambda_gw + fgw_alpha * C_lin_device
            else:
                Lambda = Lambda_gw

            # ── Augment with slack ──
            Lambda_aug = torch.zeros(N + 1, K + 1, device=device, dtype=torch.float64)
            Lambda_aug[:N, :K] = Lambda.to(torch.float64)
            max_val = Lambda.max().item()
            penalty = 100.0 * max_val if max_val > 0 else 100.0
            Lambda_aug[:-1, -1] = penalty
            Lambda_aug[-1, :-1] = penalty

            # ── Sinkhorn ──
            T_aug = _sinkhorn_fn(p_aug, q_aug, Lambda_aug, current_reg,
                                 semi_relaxed=semi_relaxed, rho=rho)
            T_new = T_aug[:-1, :-1]

            T_real = (1 - alpha) * T_prev + alpha * T_new

            gw_cost_val = (Lambda.to(torch.float64) * T_real[:N, :K]).sum().item()

            err = torch.linalg.norm(T_real - T_prev).item()
            err_list.append(err)
            n_iter = i + 1
            if verbose and (n_iter % verbose_every == 0 or i == max_iter - 1):
                print(f"  iter {n_iter:>4}/{max_iter} | err: {err:.4e}")

            if err < tol and i >= min_iter_before_converge:
                if verbose:
                    print(f"  converged at iteration {n_iter} (err={err:.4e})")
                break

            # Cleanup
            if provider is not None:
                del D_left, D_tgt
            if Lambda_gw is not None and fgw_alpha > 0:
                del Lambda_gw
            del Lambda, Lambda_aug, T_aug, T_new
            if n_iter % 50 == 0:
                maybe_gc(do_cuda=True)

    T_out = T_real.detach()
    if log:
        return T_out, {"err_list": err_list, "n_iter": n_iter, "gw_cost": gw_cost_val}
    return T_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_solver.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_solver.py tests/test_solver.py
git commit -m "refactor: provider dispatch, tensor I/O, remove graph params"
```

---

## Task 5: Add Fused GW tests

**Files:**
- Modify: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests for FGW**

Append to `tests/test_solver.py`:

```python
def test_fused_gw(two_datasets):
    """Fused GW with fgw_alpha between 0 and 1."""
    X_src, X_tgt = two_datasets
    # Simple squared Euclidean cost as linear term (same dim here)
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        X_src, X_tgt,
        fgw_alpha=0.5,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_pure_wasserstein(two_datasets):
    """fgw_alpha=1.0 should work without structural distances."""
    X_src, X_tgt = two_datasets
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        fgw_alpha=1.0,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_fgw_missing_c_linear(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="C_linear"):
        sampled_gw(X_src, X_tgt, fgw_alpha=0.5, M=30, max_iter=10)
```

- [ ] **Step 2: Run tests to verify they pass**

These should already pass if Task 4 was implemented correctly.

Run: `pytest tests/test_solver.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_solver.py
git commit -m "test: add Fused GW tests"
```

---

## Task 6: Update integration tests and embedding tests

Fix `test_spiral_swissroll.py` and `test_embedding.py` for the new API (tensor output, no `graph_source`/`graph_target` params).

**Files:**
- Modify: `tests/test_spiral_swissroll.py`
- Modify: `tests/test_embedding.py`

- [ ] **Step 1: Update `test_spiral_swissroll.py`**

Replace `tests/test_spiral_swissroll.py` with:

```python
"""Integration test: spiral -> Swiss roll alignment.

Validates that SGW produces a high-quality transport plan on the
canonical spiral-to-Swiss-roll benchmark.
"""
import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from sgw import sampled_gw


def _sample_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float32), angles


def _sample_swiss_roll(n, seed=1):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = 0.1 * rng.uniform(size=n) * 10
    return np.stack((x, z, y), axis=1).astype(np.float32), angles


class TestSpiralToSwissRoll:
    """Integration tests on 400 vs 500 spiral -> Swiss roll."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.spiral, self.a_src = _sample_spiral(400, seed=0)
        self.swiss_roll, self.a_tgt = _sample_swiss_roll(500, seed=1)

        self.T, self.log_dict = sampled_gw(
            self.spiral, self.swiss_roll,
            s_shared=400, M=80, alpha=0.8,
            max_iter=200, epsilon=0.005, k=5,
            log=True,
        )

    def test_transport_plan_shape(self):
        assert self.T.shape == (400, 500)

    def test_transport_plan_nonnegative(self):
        assert torch.all(self.T >= 0)

    def test_transport_plan_is_tensor(self):
        assert isinstance(self.T, torch.Tensor)

    def test_spearman_correlation(self):
        T_np = self.T.numpy()
        matched_angles = self.a_tgt[T_np.argmax(axis=1)]
        rho, _ = spearmanr(self.a_src, matched_angles)
        assert rho >= 0.95, f"Spearman rho = {rho:.4f}, expected >= 0.95"

    def test_monotone_matching(self):
        T_np = self.T.numpy()
        row_argmax = T_np.argmax(axis=1)
        rho, _ = spearmanr(np.arange(400), row_argmax)
        assert rho >= 0.95, f"Monotonicity rho = {rho:.4f}, expected >= 0.95"

    def test_gw_cost_returned(self):
        assert "gw_cost" in self.log_dict
        assert np.isfinite(self.log_dict["gw_cost"])
        assert self.log_dict["gw_cost"] > 0

    def test_convergence_info(self):
        assert "err_list" in self.log_dict
        assert "n_iter" in self.log_dict
        assert len(self.log_dict["err_list"]) > 0
        assert self.log_dict["n_iter"] > 0
```

- [ ] **Step 2: Update `test_embedding.py`**

Replace `tests/test_embedding.py` with:

```python
import numpy as np
import pytest
from sgw._graph import build_knn_graph
from sgw._solver import sampled_gw
from sgw._embedding import joint_embedding


@pytest.fixture
def aligned_data(two_datasets):
    """Pre-compute graphs and transport plan for embedding tests."""
    X_src, X_tgt = two_datasets
    g_src = build_knn_graph(X_src, k=10)
    g_tgt = build_knn_graph(X_tgt, k=10)
    np.random.seed(0)
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=30, epsilon=0.01)
    # joint_embedding expects numpy arrays and graphs
    T_np = T.numpy()
    return X_src, X_tgt, g_src, g_tgt, T_np


def test_joint_embedding_shapes(aligned_data):
    X_src, X_tgt, g_src, g_tgt, T = aligned_data
    result = joint_embedding(
        anchor_name="tgt",
        data_by_name={"src": X_src, "tgt": X_tgt},
        graphs_by_name={"src": g_src, "tgt": g_tgt},
        transport_plans={("src", "tgt"): T},
        out_dim=5,
    )
    assert "src" in result and "tgt" in result
    assert result["src"].shape == (150, 5)
    assert result["tgt"].shape == (150, 5)


def test_joint_embedding_finite(aligned_data):
    X_src, X_tgt, g_src, g_tgt, T = aligned_data
    result = joint_embedding(
        anchor_name="tgt",
        data_by_name={"src": X_src, "tgt": X_tgt},
        graphs_by_name={"src": g_src, "tgt": g_tgt},
        transport_plans={("src", "tgt"): T},
        out_dim=5,
    )
    for name, emb in result.items():
        assert np.all(np.isfinite(emb)), f"{name} has non-finite values"
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_spiral_swissroll.py tests/test_embedding.py
git commit -m "test: update integration and embedding tests for v0.2.0 API"
```

---

## Task 7: Update version and run full test suite

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update version**

In `pyproject.toml`, change:

```
version = "0.1.0"
```

to:

```
version = "0.2.0"
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
```
