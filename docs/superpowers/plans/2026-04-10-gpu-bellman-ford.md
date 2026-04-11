# GPU Bellman-Ford SSSP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GpuBFProvider`, a GPU-accelerated multi-source SSSP provider using vectorized Bellman-Ford, and benchmark it against `scipy.sparse.csgraph.dijkstra`.

**Architecture:** The KNN graph (CSR) is converted to a dense `(N, k)` neighbor/weight tensor on GPU. Each BF iteration gathers neighbor distances and takes element-wise minimum across all M sources simultaneously. `GpuBFProvider` wraps this with the same LRU cache and `DistanceProvider` interface as the existing `DijkstraProvider`.

**Tech Stack:** PyTorch (CUDA), scipy.sparse (CSR input), numpy (conversion), existing `build_knn_graph` from `torchgw._graph`.

**Working directory for all commands:** `/scratch/users/chensj16/projects/sgw/.worktrees/gpu-dij`

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `torchgw/_gpu_bf.py` | **create** | `build_dense_knn`, `gpu_bf_batch`, `GpuBFProvider` |
| `tests/test_gpu_bf.py` | **create** | Unit tests for all three above |
| `examples/benchmark_gpu_dij.py` | **create** | Standalone benchmark script |

No existing files are modified — this is an isolated addition.

---

## Task 1: `build_dense_knn` — CSR to dense GPU tensors

**Files:**
- Create: `torchgw/_gpu_bf.py`
- Create: `tests/test_gpu_bf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_bf.py`:

```python
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from torchgw._gpu_bf import build_dense_knn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path_graph_4():
    """0-1-2-3 path with unit weights (undirected)."""
    row = [0, 1, 1, 2, 2, 3]
    col = [1, 0, 2, 1, 3, 2]
    data = [1.0] * 6
    return csr_matrix((data, (row, col)), shape=(4, 4), dtype=np.float32)


def test_build_dense_knn_shapes():
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=3, device=DEVICE)
    assert nbrs.shape == (4, 3), f"expected (4,3), got {nbrs.shape}"
    assert wts.shape == (4, 3), f"expected (4,3), got {wts.shape}"
    assert nbrs.dtype == torch.int64
    assert wts.dtype == torch.float32
    assert nbrs.device.type == DEVICE.type


def test_build_dense_knn_self_loop_padding():
    """Nodes with fewer than k neighbors should be padded with self-loops (weight 0)."""
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=3, device=DEVICE)
    # Node 0 has 1 neighbor (node 1); slots 1 and 2 should be self-loop (0, weight 0.0)
    assert nbrs[0, 0].item() == 1
    assert wts[0, 0].item() == pytest.approx(1.0)
    assert nbrs[0, 1].item() == 0  # self-loop
    assert wts[0, 1].item() == pytest.approx(0.0)


def test_build_dense_knn_k_clipped():
    """k larger than max degree should not crash; actual k equals max degree."""
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=100, device=DEVICE)
    # Max degree in path graph is 2; result should have k=min(100, max_degree)=2
    assert nbrs.shape[1] == 2
    assert wts.shape[1] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/users/chensj16/projects/sgw/.worktrees/gpu-dij
PYTHONPATH=. pytest tests/test_gpu_bf.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'build_dense_knn' from 'torchgw._gpu_bf'` (file doesn't exist yet).

- [ ] **Step 3: Implement `build_dense_knn` in `torchgw/_gpu_bf.py`**

Create `torchgw/_gpu_bf.py`:

```python
"""GPU-accelerated multi-source SSSP via vectorized Bellman-Ford."""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch
from scipy.sparse import csr_matrix


def build_dense_knn(
    graph: csr_matrix,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a CSR sparse graph to dense (N, k) neighbor/weight GPU tensors.

    Nodes with fewer than k neighbors are padded with self-loops (weight 0.0).
    If k exceeds the maximum degree, the output width is clamped to max degree.

    Parameters
    ----------
    graph : csr_matrix, shape (N, N)
        Symmetric sparse distance graph (float32).
    k : int
        Number of neighbor slots.
    device : torch.device
        Target device for the output tensors.

    Returns
    -------
    nbrs : LongTensor, shape (N, k_actual)
        Neighbor node indices. Padding slots contain the node's own index.
    wts : FloatTensor, shape (N, k_actual)
        Edge weights. Padding slots contain 0.0.
    """
    N = graph.shape[0]
    row_lengths = np.diff(graph.indptr)
    k_actual = int(min(k, row_lengths.max()))

    # Default: self-loops with zero weight
    nbrs_np = np.tile(np.arange(N, dtype=np.int64)[:, None], (1, k_actual))
    wts_np = np.zeros((N, k_actual), dtype=np.float32)

    indptr = graph.indptr
    indices = graph.indices
    data = graph.data.astype(np.float32)

    for i in range(N):
        s, e = indptr[i], indptr[i + 1]
        cnt = min(e - s, k_actual)
        if cnt > 0:
            nbrs_np[i, :cnt] = indices[s : s + cnt]
            wts_np[i, :cnt] = data[s : s + cnt]

    return (
        torch.from_numpy(nbrs_np).to(device),
        torch.from_numpy(wts_np).to(device),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. pytest tests/test_gpu_bf.py::test_build_dense_knn_shapes \
    tests/test_gpu_bf.py::test_build_dense_knn_self_loop_padding \
    tests/test_gpu_bf.py::test_build_dense_knn_k_clipped -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add torchgw/_gpu_bf.py tests/test_gpu_bf.py
git commit -m "feat: add build_dense_knn — CSR to dense GPU neighbor tensors"
```

---

## Task 2: `gpu_bf_batch` — vectorized multi-source Bellman-Ford

**Files:**
- Modify: `torchgw/_gpu_bf.py` (append)
- Modify: `tests/test_gpu_bf.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_bf.py`:

```python
from torchgw._gpu_bf import build_dense_knn, gpu_bf_batch


def test_gpu_bf_batch_path_graph_single_source():
    """Source 0 on 0-1-2-3 path: distances must be [0, 1, 2, 3]."""
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=3, device=DEVICE)
    D = gpu_bf_batch(nbrs, wts, sources=np.array([0]), n_iters=10, device=DEVICE)
    assert D.shape == (1, 4)
    expected = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=DEVICE)
    assert torch.allclose(D, expected, atol=1e-4), f"got {D}"


def test_gpu_bf_batch_multi_source():
    """Two sources on path graph: verify both distance rows are correct."""
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=3, device=DEVICE)
    D = gpu_bf_batch(nbrs, wts, sources=np.array([0, 3]), n_iters=10, device=DEVICE)
    assert D.shape == (2, 4)
    expected = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
    ], device=DEVICE)
    assert torch.allclose(D, expected, atol=1e-4), f"got {D}"


def test_gpu_bf_batch_source_distance_zero():
    """Source node must always have distance 0."""
    g = _path_graph_4()
    nbrs, wts = build_dense_knn(g, k=3, device=DEVICE)
    sources = np.array([1, 2])
    D = gpu_bf_batch(nbrs, wts, sources=sources, n_iters=10, device=DEVICE)
    for i, s in enumerate(sources):
        assert D[i, s].item() == pytest.approx(0.0), f"D[{i},{s}] = {D[i,s]}"


def test_gpu_bf_batch_nonnegative():
    """All returned distances must be non-negative."""
    from torchgw._graph import build_knn_graph
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    g = build_knn_graph(X, k=5)
    nbrs, wts = build_dense_knn(g, k=5, device=DEVICE)
    sources = np.arange(10)
    D = gpu_bf_batch(nbrs, wts, sources=sources, n_iters=30, device=DEVICE)
    finite_mask = torch.isfinite(D)
    assert torch.all(D[finite_mask] >= 0), "negative distances found"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. pytest tests/test_gpu_bf.py -k "gpu_bf_batch" -v 2>&1 | head -15
```

Expected: `ImportError` for `gpu_bf_batch`.

- [ ] **Step 3: Implement `gpu_bf_batch` — append to `torchgw/_gpu_bf.py`**

Add after `build_dense_knn`:

```python
def gpu_bf_batch(
    nbrs: torch.Tensor,
    wts: torch.Tensor,
    sources: np.ndarray,
    n_iters: int,
    device: torch.device,
) -> torch.Tensor:
    """Run Bellman-Ford SSSP from multiple sources simultaneously on GPU.

    Each BF iteration performs a vectorized "pull" relaxation:
        D[:, v] = min(D[:, v], D[:, nbrs[v, j]] + wts[v, j])  for all j, v
    After ``n_iters`` rounds this converges to exact shortest paths when
    n_iters >= diameter. With fewer iters the result is an approximation
    (underestimates on un-reached paths left as inf).

    Parameters
    ----------
    nbrs : LongTensor, shape (N, k)
        Neighbor index array from ``build_dense_knn``.
    wts : FloatTensor, shape (N, k)
        Edge weight array from ``build_dense_knn``.
    sources : ndarray of int, shape (M,)
        Source node indices.
    n_iters : int
        Number of BF iterations. Use ``default_n_iters(N)`` if unsure.
    device : torch.device
        Device for computation. nbrs/wts must already be on this device.

    Returns
    -------
    D : FloatTensor, shape (M, N)
        D[i, v] = shortest-path distance from sources[i] to node v.
    """
    N = nbrs.shape[0]
    k = nbrs.shape[1]
    M = len(sources)

    D = torch.full((M, N), float("inf"), dtype=torch.float32, device=device)
    for i, s in enumerate(sources):
        D[i, int(s)] = 0.0

    nbrs = nbrs.to(device)
    wts = wts.to(device)

    for _ in range(n_iters):
        for j in range(k):
            u_idx = nbrs[:, j]          # (N,): for each node v, its j-th neighbor u
            # cand[:, v] = D[:, u] + w(v, u)  — pull from neighbor
            cand = D[:, u_idx] + wts[:, j].unsqueeze(0)   # (M, N)
            D = torch.minimum(D, cand)

    return D


def default_n_iters(N: int) -> int:
    """Conservative iteration count: 2 * ceil(log2(N+1)).

    Empirically sufficient for KNN graphs with k >= 5.
    """
    return 2 * math.ceil(math.log2(N + 1))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. pytest tests/test_gpu_bf.py -k "gpu_bf_batch" -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add torchgw/_gpu_bf.py tests/test_gpu_bf.py
git commit -m "feat: add gpu_bf_batch — vectorized multi-source Bellman-Ford on GPU"
```

---

## Task 3: `GpuBFProvider` — DistanceProvider protocol implementation

**Files:**
- Modify: `torchgw/_gpu_bf.py` (append)
- Modify: `tests/test_gpu_bf.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_bf.py`:

```python
from torchgw._gpu_bf import build_dense_knn, gpu_bf_batch, GpuBFProvider
from torchgw._graph import build_knn_graph


def _make_provider(N=80, k=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 3)).astype(np.float32)
    Y = rng.normal(size=(N + 20, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=k)
    g_y = build_knn_graph(Y, k=k)
    return GpuBFProvider(g_x, g_y), g_x.shape[0], g_y.shape[0]


def test_gpu_bf_provider_output_shapes():
    provider, N, K = _make_provider()
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3, 7, 9])
    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, DEVICE)
    assert D_X.shape == (N, 3), f"D_X shape {D_X.shape}"
    assert D_Y.shape == (K, 4), f"D_Y shape {D_Y.shape}"


def test_gpu_bf_provider_dtype_and_device():
    provider, _, _ = _make_provider()
    D_X, D_Y = provider.get_distances(np.array([0, 1]), np.array([0, 1]), DEVICE)
    assert D_X.dtype == torch.float32
    assert D_Y.dtype == torch.float32
    assert D_X.device.type == DEVICE.type
    assert D_Y.device.type == DEVICE.type


def test_gpu_bf_provider_nonnegative():
    provider, _, _ = _make_provider()
    D_X, D_Y = provider.get_distances(np.arange(5), np.arange(5), DEVICE)
    for D in (D_X, D_Y):
        finite = torch.isfinite(D)
        assert torch.all(D[finite] >= 0)


def test_gpu_bf_provider_source_zero():
    """Column j of D_X must have D_X[src_idx[j], j] == 0."""
    provider, _, _ = _make_provider()
    src_idx = np.array([2, 7, 15])
    D_X, _ = provider.get_distances(src_idx, np.array([0]), DEVICE)
    for col, s in enumerate(src_idx):
        assert D_X[s, col].item() == pytest.approx(0.0, abs=1e-4), \
            f"D_X[{s},{col}] = {D_X[s,col]}"


def test_gpu_bf_provider_cache_hit():
    """Calling get_distances twice with same indices should return identical results."""
    provider, _, _ = _make_provider()
    idx = np.array([0, 3, 6])
    D_X1, D_Y1 = provider.get_distances(idx, idx, DEVICE)
    D_X2, D_Y2 = provider.get_distances(idx, idx, DEVICE)
    assert torch.allclose(D_X1, D_X2)
    assert torch.allclose(D_Y1, D_Y2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. pytest tests/test_gpu_bf.py -k "provider" -v 2>&1 | head -15
```

Expected: `ImportError` for `GpuBFProvider`.

- [ ] **Step 3: Implement `GpuBFProvider` — append to `torchgw/_gpu_bf.py`**

Add after `default_n_iters`:

```python
class GpuBFProvider:
    """Multi-source SSSP via GPU Bellman-Ford on kNN graphs.

    Drop-in replacement for ``DijkstraProvider``. Converts the CSR graph to
    dense (N, k) neighbor/weight tensors once at init, then serves distance
    rows on demand with an LRU cache shared across GW iterations.
    """

    _MAX_CACHE_ROWS = 2000

    def __init__(
        self,
        graph_source: csr_matrix,
        graph_target: csr_matrix,
        k: int = 30,
        n_iters: int | None = None,
        device: torch.device | None = None,
    ):
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nbrs_src, self._wts_src = build_dense_knn(graph_source, k, self._device)
        self._nbrs_tgt, self._wts_tgt = build_dense_knn(graph_target, k, self._device)
        N_max = max(graph_source.shape[0], graph_target.shape[0])
        self._n_iters = n_iters if n_iters is not None else default_n_iters(N_max)
        self._cache_src: dict[int, np.ndarray] = {}
        self._cache_tgt: dict[int, np.ndarray] = {}

    def _get_rows(
        self,
        nbrs: torch.Tensor,
        wts: torch.Tensor,
        indices: np.ndarray,
        cache: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Return (len(indices), N) float32 distance array, using cache for hits."""
        unique = np.unique(indices)
        uncached = np.array([s for s in unique if s not in cache], dtype=np.intp)

        if len(uncached) > 0:
            D_new = gpu_bf_batch(nbrs, wts, uncached, self._n_iters, self._device)
            D_new_np = D_new.cpu().numpy()
            needed = set(int(s) for s in unique)
            for i, s in enumerate(uncached):
                if len(cache) >= self._MAX_CACHE_ROWS:
                    for key in list(cache):
                        if key not in needed:
                            cache.pop(key)
                            break
                cache[int(s)] = D_new_np[i].astype(np.float32)

        N = nbrs.shape[0]
        result = np.empty((len(indices), N), dtype=np.float32)
        for i, s in enumerate(indices):
            result[i] = cache[int(s)]
        return result

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (D_X, D_Y) as float32 tensors on ``device``.

        D_X : shape (N_src, len(src_indices))
        D_Y : shape (N_tgt, len(tgt_indices))
        """
        D_src = self._get_rows(self._nbrs_src, self._wts_src, src_indices, self._cache_src)
        D_tgt = self._get_rows(self._nbrs_tgt, self._wts_tgt, tgt_indices, self._cache_tgt)
        return (
            torch.from_numpy(D_src).to(device).T,
            torch.from_numpy(D_tgt).to(device).T,
        )
```

- [ ] **Step 4: Run all tests including existing suite**

```bash
PYTHONPATH=. pytest tests/test_gpu_bf.py -v
```

Expected: all `test_gpu_bf.py` tests pass.

```bash
PYTHONPATH=. pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: `111 passed` (existing suite unchanged).

- [ ] **Step 5: Commit**

```bash
git add torchgw/_gpu_bf.py tests/test_gpu_bf.py
git commit -m "feat: add GpuBFProvider — GPU BF drop-in for DijkstraProvider"
```

---

## Task 4: Benchmark script

**Files:**
- Create: `examples/benchmark_gpu_dij.py`

No TDD for this script — verify by running it end-to-end.

- [ ] **Step 1: Create `examples/benchmark_gpu_dij.py`**

```python
"""Benchmark GPU Bellman-Ford vs scipy Dijkstra for M-source SSSP.

Usage:
    PYTHONPATH=. python examples/benchmark_gpu_dij.py

Measures wall-clock time for M=50 random sources on KNN graphs of
varying size N, reports speedup and approximation error vs scipy.
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra

from torchgw._graph import build_knn_graph
from torchgw._gpu_bf import build_dense_knn, gpu_bf_batch, default_n_iters

N_VALUES = [1_000, 5_000, 10_000, 30_000]
M = 50       # sources per call
K = 5        # kNN graph degree
N_REPEATS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_graph(N: int, seed: int = 42) -> object:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3)).astype(np.float32)
    return build_knn_graph(X, k=K)


def bench_scipy(graph, sources: np.ndarray) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    D = dijkstra(csgraph=graph, directed=False, indices=sources)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000, D.astype(np.float32)


def bench_gpu(graph, sources: np.ndarray) -> tuple[float, np.ndarray]:
    nbrs, wts = build_dense_knn(graph, k=K, device=DEVICE)
    n_iters = default_n_iters(graph.shape[0])
    # Warm-up
    _ = gpu_bf_batch(nbrs, wts, sources, n_iters, DEVICE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    D = gpu_bf_batch(nbrs, wts, sources, n_iters, DEVICE)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed * 1000, D.cpu().numpy()


def compute_errors(D_ref: np.ndarray, D_gpu: np.ndarray) -> tuple[float, float]:
    """Mean and max absolute error on finite pairs."""
    finite = np.isfinite(D_ref) & np.isfinite(D_gpu)
    if not finite.any():
        return float("nan"), float("nan")
    err = np.abs(D_ref[finite] - D_gpu[finite])
    return float(err.mean()), float(err.max())


print(f"Device: {DEVICE}  |  M={M} sources  |  k={K}  |  {N_REPEATS} repeats\n")
header = f"{'N':>8}  {'scipy_ms':>10}  {'gpu_ms':>10}  {'speedup':>8}  {'mean_err':>10}  {'max_err':>10}"
print(header)
print("-" * len(header))

rows = []
for N in N_VALUES:
    graph = make_graph(N)
    rng = np.random.default_rng(7)
    sources = rng.choice(N, size=M, replace=False).astype(np.intp)

    scipy_times, gpu_times = [], []
    D_ref = D_gpu_last = None

    for rep in range(N_REPEATS):
        t_sp, D_ref = bench_scipy(graph, sources)
        t_gpu, D_gpu_last = bench_gpu(graph, sources)
        scipy_times.append(t_sp)
        gpu_times.append(t_gpu)

    scipy_ms = float(np.median(scipy_times))
    gpu_ms = float(np.median(gpu_times))
    speedup = scipy_ms / gpu_ms
    mean_err, max_err = compute_errors(D_ref, D_gpu_last)

    row = dict(N=N, scipy_ms=scipy_ms, gpu_ms=gpu_ms,
               speedup=speedup, mean_err=mean_err, max_err=max_err)
    rows.append(row)
    print(f"{N:>8}  {scipy_ms:>10.1f}  {gpu_ms:>10.1f}  {speedup:>8.2f}x"
          f"  {mean_err:>10.4f}  {max_err:>10.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

Ns = [r["N"] for r in rows]
ax1.plot(Ns, [r["scipy_ms"] for r in rows], "o-", label="scipy dijkstra (CPU)")
ax1.plot(Ns, [r["gpu_ms"] for r in rows], "s-", label=f"GPU BF ({DEVICE.type})")
ax1.set_xlabel("N (graph nodes)")
ax1.set_ylabel("Time (ms, median over 5 runs)")
ax1.set_title(f"M={M} sources SSSP: scipy vs GPU BF")
ax1.legend()
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)

ax2.plot(Ns, [r["mean_err"] for r in rows], "o-", label="mean |err|")
ax2.plot(Ns, [r["max_err"] for r in rows], "s--", label="max |err|")
ax2.set_xlabel("N (graph nodes)")
ax2.set_ylabel("Absolute error vs scipy")
ax2.set_title("GPU BF approximation error")
ax2.legend()
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "examples/benchmark_gpu_dij.png"
fig.savefig(out_path, dpi=120)
print(f"\nPlot saved → {out_path}")
plt.close(fig)
```

- [ ] **Step 2: Run the benchmark to verify it executes end-to-end**

```bash
cd /scratch/users/chensj16/projects/sgw/.worktrees/gpu-dij
PYTHONPATH=. python examples/benchmark_gpu_dij.py
```

Expected: table printed with 4 rows (N=1k/5k/10k/30k), `benchmark_gpu_dij.png` created, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark_gpu_dij.py examples/benchmark_gpu_dij.png
git commit -m "feat: add GPU BF vs scipy benchmark script and results"
```

---

## Self-Review

**Spec coverage:**
- ✅ `build_dense_knn` — Task 1
- ✅ `gpu_bf_batch` — Task 2
- ✅ `GpuBFProvider` with `DistanceProvider` protocol and LRU cache — Task 3
- ✅ Benchmark: N ∈ {1k,5k,10k,30k}, M=50, 5 repeats, time + error table + plot — Task 4
- ✅ `default_n_iters` helper defined in Task 2 and used in Tasks 3 & 4

**Type consistency:**
- `build_dense_knn` returns `(LongTensor, FloatTensor)` — used as `nbrs, wts` in Tasks 2, 3, 4 ✅
- `gpu_bf_batch` signature: `(nbrs, wts, sources: np.ndarray, n_iters: int, device)` → returns `(M,N) FloatTensor` — matches all call sites ✅
- `GpuBFProvider.get_distances` returns `(D_X, D_Y)` where `D_X.shape=(N,M), D_Y.shape=(K,M)` ✅

**No placeholders found.**
