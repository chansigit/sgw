# Design: Multi-Scale Warm Start + Low-Rank Sinkhorn (v0.3.0)

Date: 2026-04-06

## Summary

Add two scaling features to `sampled_gw`:
1. **Multi-scale warm start** (`multiscale=True`) — downsample, solve coarse GW, upsample to warm-start full solve
2. **Low-rank Sinkhorn** (`rank=50`) — factorize the transport plan to reduce memory from O(NK) to O((N+K)r)

Both are orthogonal to each other and to existing features (distance_mode, fgw_alpha, semi_relaxed).

---

## API Changes

### New parameters

```python
sampled_gw(
    ...,
    multiscale: bool = False,     # two-stage coarse-to-fine solve
    n_coarse: int | None = None,  # coarse problem size (auto if None)
    rank: int | None = None,      # low-rank Sinkhorn rank (None = standard)
)
```

### Parameter interactions

| multiscale | rank | Behavior |
|:---:|:---:|------|
| False | None | Current behavior (standard Sinkhorn) |
| True | None | Two-stage: coarse solve + warm-start fine solve |
| False | r | Low-rank Sinkhorn, memory O((N+K)r) |
| True | r | Two-stage, both stages use low-rank Sinkhorn |

All combinations work with any distance_mode and fgw_alpha.

---

## Feature 1: Multi-Scale Warm Start

### Algorithm

**Stage 1 — Coarse solve:**
1. FPS downsample source: select n_coarse representative points from X_source.
   Record assignment `assign_src[i] -> i_coarse` (each original point maps to
   its nearest representative).
2. FPS downsample target: same for X_target.
3. Default n_coarse = min(500, N // 4, K // 4).
4. Build coarse problem inputs (subsample X, dist matrices, or C_linear as needed).
5. Call `sampled_gw` recursively on the coarse problem to get T_coarse.

**Stage 2 — Fine solve with warm start:**
1. Upsample: T_init[i, k] = T_coarse[assign_src[i], assign_tgt[k]].
2. Normalize T_init so row sums match p and column sums match q (Sinkhorn-like scaling).
3. Use T_init as the initial coupling (replacing the default p (x) q outer product).
4. Run the full-scale SGW main loop.

### FPS Downsample

Farthest-point sampling in feature space (Euclidean):
1. Start from point 0.
2. Compute distances from all points to the selected set.
3. Pick the point farthest from the selected set.
4. Repeat until n_coarse points selected.

Returns: indices of selected points, assignment array.

Uses `torch.cdist` for distance computation (GPU-friendly).

### Upsample Plan

```python
def upsample_plan(T_coarse, assign_src, assign_tgt, p, q):
    """Expand (n_coarse, m_coarse) plan to (N, K) using nearest-representative assignment."""
    T_init = T_coarse[assign_src][:, assign_tgt]  # (N, K)
    # Normalize to match marginals (iterative row/col scaling, 3-5 iterations)
    for _ in range(5):
        T_init *= (p / T_init.sum(dim=1).clamp(min=1e-30)).unsqueeze(1)
        T_init *= (q / T_init.sum(dim=0).clamp(min=1e-30)).unsqueeze(0)
    return T_init
```

### File: `sgw/_multiscale.py`

Contains: `fps_downsample(X, n)`, `upsample_plan(T_coarse, assign_src, assign_tgt, p, q)`.

---

## Feature 2: Low-Rank Sinkhorn

### Algorithm (Scetbon & Peyre 2022)

Standard Sinkhorn produces T of shape (N, K), memory O(NK).
Low-rank Sinkhorn factorizes:

```
T = diag(g1) @ Q @ diag(g3) @ R^T @ diag(g2)
```

Where:
- Q: (N, r) non-negative factor
- R: (K, r) non-negative factor
- g1: (N,) scaling vector
- g2: (K,) scaling vector
- g3: (r,) coupling scaling vector

Memory: O((N+K)r) instead of O(NK).

### Initialization

Q and R are initialized via softmax of random noise projected through the
cost matrix:

```
Q = softmax(-C[:, :r] / reg)    # (N, r)
R = softmax(-C[:r, :].T / reg)  # (K, r)
```

g1, g2, g3 initialized to ones.

### Dykstra Iteration

Each iteration alternates three projections to satisfy the marginal constraints:

```
# Project onto source marginal constraint: T @ 1 = p
g1 = p / (Q @ diag(g3) @ R^T @ diag(g2) @ 1)

# Project onto target marginal constraint: T^T @ 1 = q
g2 = q / (R @ diag(g3) @ Q^T @ diag(g1) @ 1)

# Update Q, R via Dykstra splitting (KL projections)
# ... (see implementation details below)
```

All operations are O(Nr + Kr), never materializing the full (N, K) matrix.

### Reconstructing T for the SGW main loop

The SGW main loop needs T for:
1. Sampling anchor pairs (needs T as dense matrix or row marginals)
2. Computing GW cost: (Lambda * T).sum()
3. Momentum update: T <- (1-alpha) * T_prev + alpha * T_new

For (1): compute row marginals from factors: `p_row = g1 * (Q @ (g3 * (R^T @ g2)))`.
Sample rows from p_row, then for each row reconstruct that row of T to sample columns.

For (2) and (3): reconstruct T = diag(g1) @ Q @ diag(g3) @ R^T @ diag(g2).
At the target scale this reconstruction IS O(NK), so low-rank only helps if we
can avoid full reconstruction. For GW cost, use trace trick:
```
gw_cost = trace(Lambda^T @ diag(g1) @ Q @ diag(g3) @ R^T @ diag(g2))
        = (g1 * (Lambda @ (g2 * R))) . (Q @ g3)  ... O(NKr) still
```

**Practical approach**: reconstruct T explicitly only at moderate scales.
At very large scale (N, K > 50k), the cost matrix Lambda itself is O(NK)
and is already computed each iteration, so reconstructing T adds no
asymptotic overhead. The memory saving comes from not *storing* T across
iterations — reconstruct, use, discard.

For momentum: store the factors (Q, R, g1, g2, g3) and do momentum on them:
```
Q <- (1 - alpha) * Q_prev + alpha * Q_new
R <- (1 - alpha) * R_prev + alpha * R_new
```

### File: `sgw/_lowrank.py`

Contains: `sinkhorn_lowrank(p, q, C, rank, reg, max_iter, tol, semi_relaxed, rho)`.

Returns: T as a dense (N+1, K+1) tensor (same interface as `_sinkhorn_torch`),
reconstructed from the low-rank factors. This keeps the solver code simple —
the only change in `_solver.py` is which Sinkhorn function to call.

Future optimization: return factors directly and avoid reconstruction. But this
would require rewriting the sampling and GW cost logic, so defer.

### Convergence

Check marginal error on the reconstructed marginals, same as standard Sinkhorn.
Low-rank Sinkhorn typically needs more iterations to converge (default max_iter
can be higher, e.g., 200 inner iterations vs 100 for standard).

---

## File Changes

| File | Change |
|------|--------|
| `sgw/_multiscale.py` | **New**. `fps_downsample`, `upsample_plan` |
| `sgw/_lowrank.py` | **New**. `sinkhorn_lowrank` with Dykstra projection |
| `sgw/_solver.py` | Add `multiscale`, `n_coarse`, `rank` params; dispatch logic |
| `sgw/__init__.py` | No change (internal modules) |
| `tests/test_multiscale.py` | **New**. FPS, upsample, end-to-end multiscale test |
| `tests/test_lowrank.py` | **New**. Low-rank Sinkhorn correctness, rank parameter |
| `tests/test_solver.py` | Add tests for multiscale + rank combinations |

---

## Migration

Fully backward compatible. All new parameters have defaults that preserve
existing behavior (multiscale=False, rank=None).

```python
# v0.2.1 (unchanged)
T = sampled_gw(X, Y)

# v0.3.0 new features
T = sampled_gw(X, Y, multiscale=True)                    # warm start
T = sampled_gw(X, Y, rank=50)                            # low-rank
T = sampled_gw(X, Y, multiscale=True, rank=50)           # both
T = sampled_gw(X, Y, multiscale=True, distance_mode="landmark")  # with landmark
```
