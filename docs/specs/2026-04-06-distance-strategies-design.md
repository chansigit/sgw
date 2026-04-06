# Design: Three Distance Strategies + Fused GW (v0.2.0)

Date: 2026-04-06

## Summary

Refactor `sampled_gw` to support three distance computation strategies (precomputed,
dijkstra, spectral) and Fused Gromov-Wasserstein, with torch tensor as the primary
input/output type.

This is a **breaking change** from v0.1.0.

---

## API Changes

### New signature

```python
def sampled_gw(
    X_source: torch.Tensor | np.ndarray | None = None,
    X_target: torch.Tensor | np.ndarray | None = None,
    p: torch.Tensor | np.ndarray | None = None,
    q: torch.Tensor | np.ndarray | None = None,
    *,
    # Distance strategy
    distance_mode: str = "dijkstra",        # "precomputed" | "dijkstra" | "spectral"
    dist_source: torch.Tensor | np.ndarray | None = None,   # (N, N)
    dist_target: torch.Tensor | np.ndarray | None = None,   # (K, K)
    spectral_dim: int = 50,                 # only used when distance_mode="spectral"

    # Fused GW
    fgw_alpha: float = 0.0,                 # 0.0 = pure GW, 1.0 = pure Wasserstein
    C_linear: torch.Tensor | np.ndarray | None = None,      # (N, K) feature cost

    # Existing parameters (unchanged)
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
```

### Removed parameters (breaking)

- `graph_source` — removed. kNN graph is an internal detail, not a user-facing concept.
- `graph_target` — removed. Same reason.

### Input/output type change (breaking)

- **Input**: accepts both `torch.Tensor` and `np.ndarray`. Internally converted to tensor
  via `torch.as_tensor` (zero-copy when already a tensor).
- **Output**: returns `torch.Tensor` (v0.1.0 returned `np.ndarray`).

---

## Parameter Validation

### Which parameters are required for each mode

| Parameter | precomputed (with dist) | precomputed (without dist) | dijkstra | spectral |
|-----------|:-----------------------:|:--------------------------:|:--------:|:--------:|
| X_source, X_target | not needed | **required** | **required** | **required** |
| dist_source, dist_target | **required** | not needed | ignored | ignored |
| k | ignored | used (build kNN) | used (build kNN) | used (build kNN) |
| spectral_dim | ignored | ignored | ignored | used |

### Fused GW validation

| fgw_alpha | C_linear | Behavior |
|-----------|----------|----------|
| 0.0 (default) | ignored | Pure GW, C_linear not used |
| 0.0 < alpha < 1.0 | **required** | Fused GW |
| 1.0 | **required** | Pure Wasserstein (no structural term) |

When `fgw_alpha=1.0`, no distance strategy is needed (no structural cost).
X_source, X_target, dist_source, dist_target can all be None.

### N and K inference

Inferred from the first available source (in priority order):
1. dist_source.shape[0], dist_target.shape[0]
2. X_source.shape[0], X_target.shape[0]
3. C_linear.shape (N, K)

Raise ValueError if shapes are inconsistent.

---

## Internal Architecture

### Distance provider pattern

New file: `sgw/_distances.py`

Three provider classes, all implementing the same interface:

```python
class DistanceProvider:
    def get_distances(self, src_indices, tgt_indices, device) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        D_X : (N, M) distances from all source points to sampled source anchors
        D_Y : (K, M) distances from all target points to sampled target anchors
        """
        ...
```

#### PrecomputedProvider

- Stores full distance matrices C_X (N, N) and C_Y (K, K) as tensors on device.
- `get_distances`: index columns `C_X[:, src_indices]`, `C_Y[:, tgt_indices]`.
- If user passed dist_source/dist_target, use directly.
- If user passed X_source/X_target (no dist), compute all-pairs Dijkstra once at init.

#### DijkstraProvider

- Stores kNN graph references (built at init from X_source, X_target).
- `get_distances`: runs `batch_dijkstra` each call (current v0.1.0 behavior).

#### SpectralProvider

- At init: builds kNN graphs, computes spectral embedding Z_X (N, d) and Z_Y (K, d),
  stores as GPU tensors.
- `get_distances`: computes Euclidean distances on GPU.
  `D_X[:, m] = || Z_X - Z_X[src_m] ||` for each anchor m.

### Inf handling and normalization

Stays in `_solver.py` main loop, applied to the output of any provider.
Provider returns raw distances; main loop clamps inf and normalizes.

### Fused GW cost assembly

In the main loop, after computing Lambda_gw (structural cost):

```python
if fgw_alpha > 0:
    Lambda = (1 - fgw_alpha) * Lambda_gw + fgw_alpha * C_linear
else:
    Lambda = Lambda_gw
```

C_linear is precomputed once (does not change across iterations), stored as a tensor
on device.

---

## File Changes

| File | Change |
|------|--------|
| `sgw/_distances.py` | **New**. Three provider classes. |
| `sgw/_solver.py` | Add new parameters. Replace inline Dijkstra with provider call. Add FGW blending. Change input/output to tensor. Remove graph_source/graph_target. |
| `sgw/_graph.py` | No change (still used internally by providers). |
| `sgw/_sampling.py` | No change. |
| `sgw/_embedding.py` | No change. |
| `sgw/__init__.py` | No change to exports. |
| `tests/test_solver.py` | Update for tensor output. Add tests for each distance mode and FGW. |
| `tests/test_spiral_swissroll.py` | Update for tensor output. |

---

## Migration from v0.1.0

```python
# v0.1.0
T = sampled_gw(X_np, Y_np)              # returns np.ndarray
T = sampled_gw(X_np, Y_np, graph_source=g1, graph_target=g2)

# v0.2.0
T = sampled_gw(X_np, Y_np)              # returns torch.Tensor
T = sampled_gw(X_np, Y_np).numpy()      # if numpy needed

# graph_source/graph_target removed — just pass X, graphs are built internally

# New features
T = sampled_gw(dist_source=D1, dist_target=D2, distance_mode="precomputed")
T = sampled_gw(X, Y, distance_mode="spectral", spectral_dim=80)
T = sampled_gw(X, Y, fgw_alpha=0.5, C_linear=C_feat)
```
