# GPU Dijkstra (Bellman-Ford) Design

**Date:** 2026-04-10  
**Branch:** `gpu-dij`  
**Goal:** Implement a GPU-accelerated approximate SSSP via vectorized Bellman-Ford, benchmark against `scipy.sparse.csgraph.dijkstra` in the `dijkstra` distance mode (M=50 sources per call).

---

## Context

`torchgw/_distances.py` contains three `DistanceProvider` implementations. The `DijkstraProvider` calls `scipy.sparse.csgraph.dijkstra` on a KNN graph (k=5) for M source nodes per GW iteration, with LRU caching. For N=5k–50k, this is the bottleneck.

scipy's Dijkstra is CPU-only and sequentially bounded by its priority queue. For M simultaneous sources on a GPU, a vectorized Bellman-Ford can process all sources in parallel.

---

## Algorithm: Vectorized Bellman-Ford

### Key insight

The KNN graph is stored as a sparse CSR matrix but can also be viewed as a dense `(N, k)` neighbor/weight table. Each BF iteration is:

```
D[:, v] = min(D[:, v], D[:, nbrs[v, j]] + wts[v, j])   for j in 0..k-1
```

where `D` is `(M, N)` float32 — all M source distances at once.

### Steps

1. **Build neighbor table**: Convert CSR to `(N, k)` dense arrays `nbrs` and `wts` on GPU (done once per graph).
2. **Initialize**: `D[s, v] = inf` everywhere; `D[s, source_node[s]] = 0` for each of M sources.
3. **Iterate** for `n_iters` steps (default = 2× estimated graph diameter):
   - For each of the k neighbor slots `j`:
     - `candidate = D[:, nbrs[:, j]] + wts[:, j]`  — shape `(M, N)`
     - `D = torch.minimum(D, candidate)`
4. **Return** rows of `D` indexed by requested source nodes.

Approximate: stopping at a fixed iteration count instead of convergence. For KNN graphs with k=5, diameter ≈ O(log N), so 20–40 iterations is sufficient.

### Memory

| N     | M  | D matrix    |
|-------|----|-------------|
| 5 000 | 50 | ~1 MB       |
| 50 000| 50 | ~10 MB      |

H100 (80 GB HBM3) — trivial.

---

## Implementation Plan

### New file: `torchgw/_gpu_distances.py`

```
GpuBFProvider
  __init__(graph_source, graph_target, n_iters=30)
    - Converts CSR → dense (N, k) nbrs/wts tensors, puts on GPU
  get_distances(src_indices, tgt_indices, device)
    - Runs _batch_bf(graph, sources, n_iters) for src and tgt
    - Returns (D_X, D_Y) same interface as DijkstraProvider

_build_neighbor_table(csr: csr_matrix) -> (nbrs: Tensor, wts: Tensor)
  - Returns (N, max_degree) int64 and (N, max_degree) float32
  - max_degree = actual max degree in graph (≤ 2k after symmetrization + stitching edges)
  - Pads short rows with (self_node, inf) so minimum ops are no-ops

_batch_bf(nbrs, wts, sources, n_iters) -> Tensor (M, N)
  - Core vectorized BF on GPU
```

### Integration in `torchgw/_distances.py`

Add `"gpu_bf"` as a new valid `distance_mode`. Wire up `GpuBFProvider` in `_solver.py` alongside the existing three modes.

### Benchmark script: `examples/benchmark_gpu_dij.py`

- N ∈ {1k, 5k, 10k, 30k}, M=50 sources, k=5 KNN graph
- Repeat 5 runs per config, report median ± std
- Compare: scipy Dijkstra vs GPU BF (n_iters=30)
- Output table: time ratio + mean/max absolute error vs scipy

---

## Success Criteria

1. GPU BF produces distances with mean absolute error < 5% vs scipy on spiral KNN graph.
2. GPU BF is faster than scipy for N ≥ 10k, M=50 on H100.
3. `GpuBFProvider` passes the same interface contract as `DijkstraProvider` (`get_distances` returns `(D_X, D_Y)` float32 tensors).
4. Benchmark script runs end-to-end and prints a clean table.

---

## Out of Scope

- CuPy custom CUDA kernels (added complexity, marginal gain over PyTorch BF)
- Replacing `DijkstraProvider` as default (this is an experimental branch)
- Support for directed graphs
- `precomputed` or `landmark` mode changes
