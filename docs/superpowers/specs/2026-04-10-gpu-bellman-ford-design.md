# GPU Bellman-Ford SSSP — Design Spec

**Date:** 2026-04-10  
**Branch:** `gpu-dij`  
**Scope:** Add a GPU-accelerated multi-source SSSP provider (`GpuBFProvider`) and benchmark it against scipy's Dijkstra in the `dijkstra` distance mode (M=50 sources per call).

---

## Context

`torchgw` computes Gromov-Wasserstein distances between graphs. The critical inner loop calls `DijkstraProvider.get_distances()`, which runs M≈50 single-source shortest paths per GW iteration using `scipy.sparse.csgraph.dijkstra` (CPU, C implementation). For N=5k–50k nodes this is the dominant cost. The goal is a GPU-accelerated drop-in replacement and a benchmark showing where it wins.

---

## Algorithm: Vectorized Bellman-Ford

### Why not GPU Dijkstra?

Standard Dijkstra requires a global-minimum priority queue, which is inherently sequential and maps poorly to SIMD GPU execution. For sparse KNN graphs (k=5 edges/node), each SSSP has very limited per-step parallelism.

### Why Bellman-Ford works here

- Each BF iteration relaxes **all edges simultaneously** — fully data-parallel.
- With M sources batched together, we exploit GPU width across both sources and nodes.
- KNN graphs have small diameter (O(log N) empirically), so BF converges in ~2×diameter iterations.
- Approximate (early-stop) is acceptable per requirements.

### Core iteration

```
D: (M, N) float32 on GPU   # M source rows, N nodes
Initialize: D[i, sources[i]] = 0.0, all others = inf

for _ in range(n_iters):          # n_iters ≈ 2 × estimated_diameter
    for j in range(k):            # k = number of neighbors per node
        cand = D[:, nbrs[:, j]] + wts[:, j]   # (M, N) gather + broadcast
        D = torch.minimum(D, cand)             # in-place min
```

Memory: M=50, N=50k, float32 → 10 MB per call. Trivial for H100 80GB.

---

## Implementation

### New file: `torchgw/_gpu_bf.py`

**`build_dense_knn(graph: csr_matrix, k: int) -> tuple[Tensor, Tensor]`**  
Converts a CSR sparse graph to two `(N, k)` float32/int64 GPU tensors: `nbrs` (neighbor indices) and `wts` (edge weights). Pads with self-loops (weight=0) for nodes with fewer than k neighbors.

**`gpu_bf_batch(nbrs, wts, sources, n_iters, device) -> Tensor`**  
Core BF loop. Returns `(M, N)` float32 distance matrix. `n_iters` defaults to `2 * k * int(log2(N+1))` as a conservative diameter estimate.

**`GpuBFProvider`**  
Implements the `DistanceProvider` protocol. Drop-in replacement for `DijkstraProvider`.

- `__init__(graph_source, graph_target, n_iters=None, device=None)`: converts both graphs to dense-knn format, stores on GPU.
- `_get_rows(nbrs, wts, indices, cache)`: cache-hit lookup + `gpu_bf_batch` for misses. Same LRU eviction strategy as `DijkstraProvider`.
- `get_distances(src_indices, tgt_indices, device)`: returns `(D_X, D_Y)` as float32 tensors, matching existing interface exactly.

Cache: reuses `DijkstraProvider._MAX_CACHE_ROWS = 2000` logic to avoid redundant BF calls across GW iterations.

### New file: `examples/benchmark_gpu_dij.py`

Standalone script, no dependency on the full GW solver.

**Test matrix:** N ∈ {1000, 5000, 10000, 30000}, M=50 sources (random), k=5, 5 repeated runs.

**Per (N, M) cell:**
1. Build KNN graph (same `build_knn_graph` as production code)
2. Run scipy `dijkstra` (CPU) → reference distances
3. Run `gpu_bf_batch` (GPU) → approximate distances
4. Record: `scipy_ms`, `gpu_ms`, `speedup`, `mean_abs_err`, `max_abs_err` (excluding inf pairs)

**Output:** printed table + `examples/benchmark_gpu_dij.png` (time and error subplots).

---

## Interface contract

`GpuBFProvider` must satisfy the `DistanceProvider` protocol:

```python
def get_distances(
    self,
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

It does **not** need to be wired into `sampled_gw()` for this branch — that is a follow-up decision pending benchmark results.

---

## Success criteria

- `gpu_ms < scipy_ms` for N ≥ 10k, M=50
- `mean_abs_err < 0.05` (normalized distances, relative to max finite distance)
- `max_abs_err < 0.5` for ≥ 95% of node pairs
- Script runs end-to-end with `PYTHONPATH=. python examples/benchmark_gpu_dij.py`

---

## Out of scope

- Wiring `GpuBFProvider` into `sampled_gw()` (pending benchmark)
- Custom CUDA kernels (Option B)
- Delta-stepping
- Changes to existing providers or solver
