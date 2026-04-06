# Multi-Scale + Low-Rank Sinkhorn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-scale warm start (`multiscale=True`) and low-rank Sinkhorn (`rank=r`) to `sampled_gw`.

**Architecture:** Two new internal modules (`_multiscale.py`, `_lowrank.py`) with clean interfaces, wired into `_solver.py` via parameter dispatch. Both features are orthogonal and composable.

**Tech Stack:** PyTorch, scipy (Dijkstra for landmark), numpy

**Spec:** `docs/specs/2026-04-06-multiscale-lowrank-design.md`

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `sgw/_multiscale.py` | FPS downsample + upsample plan | **New** |
| `sgw/_lowrank.py` | Low-rank Sinkhorn via Dykstra projection | **New** |
| `sgw/_solver.py` | Main solver | **Modify** — add params, dispatch logic |
| `tests/test_multiscale.py` | Multi-scale unit tests | **New** |
| `tests/test_lowrank.py` | Low-rank Sinkhorn unit tests | **New** |
| `tests/test_solver.py` | Solver integration tests | **Modify** — add combo tests |

---

## Task 1: Create `_multiscale.py` with `fps_downsample`

**Files:**
- Create: `sgw/_multiscale.py`
- Create: `tests/test_multiscale.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_multiscale.py
import numpy as np
import torch
import pytest
from sgw._multiscale import fps_downsample


def test_fps_downsample_shapes():
    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.normal(size=(100, 5)).astype(np.float32))
    indices, assignments = fps_downsample(X, n=20)
    assert indices.shape == (20,)
    assert assignments.shape == (100,)
    # Every assignment should point to a valid index
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multiscale.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `fps_downsample`**

```python
# sgw/_multiscale.py
from __future__ import annotations

import torch


def fps_downsample(
    X: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Farthest-point sampling in feature space.

    Parameters
    ----------
    X : Tensor of shape (N, D)
    n : number of points to select

    Returns
    -------
    indices : LongTensor of shape (n,) — indices of selected points
    assignments : LongTensor of shape (N,) — maps each point to its
        nearest selected point (index into ``indices``, not into X)
    """
    N = X.shape[0]
    n = min(n, N)
    device = X.device

    selected = []
    min_dists = torch.full((N,), float("inf"), device=device)
    next_idx = 0

    for _ in range(n):
        selected.append(next_idx)
        dists = torch.cdist(X, X[next_idx : next_idx + 1]).squeeze(1)  # (N,)
        min_dists = torch.minimum(min_dists, dists)
        next_idx = int(torch.argmax(min_dists).item())

    indices = torch.tensor(selected, dtype=torch.long, device=device)

    # Assign each point to its nearest selected point
    # D_sel shape: (N, n)
    D_sel = torch.cdist(X, X[indices])
    assignments = torch.argmin(D_sel, dim=1)  # (N,), index into `indices`

    return indices, assignments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multiscale.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_multiscale.py tests/test_multiscale.py
git commit -m "feat: add fps_downsample in _multiscale.py"
```

---

## Task 2: Add `upsample_plan` to `_multiscale.py`

**Files:**
- Modify: `sgw/_multiscale.py`
- Modify: `tests/test_multiscale.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_multiscale.py`:

```python
from sgw._multiscale import upsample_plan


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multiscale.py::test_upsample_plan_shapes -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `upsample_plan`**

Add to `sgw/_multiscale.py`:

```python
def upsample_plan(
    T_coarse: torch.Tensor,
    assign_src: torch.Tensor,
    assign_tgt: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    n_iter: int = 10,
) -> torch.Tensor:
    """Expand coarse transport plan to full size via nearest-representative assignment.

    Parameters
    ----------
    T_coarse : (n_coarse, m_coarse) coarse transport plan
    assign_src : (N,) LongTensor, maps each source point to coarse index
    assign_tgt : (K,) LongTensor, maps each target point to coarse index
    p : (N,) source marginal
    q : (K,) target marginal
    n_iter : Sinkhorn-like scaling iterations to match marginals

    Returns
    -------
    T_fine : (N, K) upsampled transport plan with marginals matching p, q
    """
    T_fine = T_coarse[assign_src][:, assign_tgt]  # (N, K)
    T_fine = T_fine.to(dtype=p.dtype)
    T_fine = T_fine.clamp(min=1e-30)

    # Iterative row/column scaling to match marginals
    for _ in range(n_iter):
        row_sum = T_fine.sum(dim=1).clamp(min=1e-30)
        T_fine *= (p / row_sum).unsqueeze(1)
        col_sum = T_fine.sum(dim=0).clamp(min=1e-30)
        T_fine *= (q / col_sum).unsqueeze(0)

    return T_fine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multiscale.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_multiscale.py tests/test_multiscale.py
git commit -m "feat: add upsample_plan in _multiscale.py"
```

---

## Task 3: Create `_lowrank.py` with `sinkhorn_lowrank`

**Files:**
- Create: `sgw/_lowrank.py`
- Create: `tests/test_lowrank.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lowrank.py
import torch
import pytest
from sgw._lowrank import sinkhorn_lowrank


def test_lowrank_returns_correct_shape():
    N, K, r = 30, 40, 5
    p = torch.ones(N, dtype=torch.float64) / N
    q = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)

    T = sinkhorn_lowrank(p, q, C, rank=r, reg=0.1)
    assert T.shape == (N, K)
    assert T.dtype == torch.float64


def test_lowrank_nonnegative():
    N, K, r = 30, 40, 5
    p = torch.ones(N, dtype=torch.float64) / N
    q = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)

    T = sinkhorn_lowrank(p, q, C, rank=r, reg=0.1)
    assert torch.all(T >= 0)


def test_lowrank_marginals():
    """Row and column sums should approximately match p and q."""
    N, K, r = 50, 60, 10
    p = torch.ones(N, dtype=torch.float64) / N
    q = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)

    T = sinkhorn_lowrank(p, q, C, rank=r, reg=0.05, max_iter=500)
    assert torch.allclose(T.sum(dim=1), p, atol=1e-3), \
        f"Row sum error: {(T.sum(dim=1) - p).abs().max():.4e}"
    assert torch.allclose(T.sum(dim=0), q, atol=1e-3), \
        f"Col sum error: {(T.sum(dim=0) - q).abs().max():.4e}"


def test_lowrank_vs_standard_close():
    """Low-rank with large rank should approximate standard Sinkhorn."""
    from sgw._solver import _sinkhorn_torch
    N, K = 20, 25
    p = torch.ones(N, dtype=torch.float64) / N
    q = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)
    reg = 0.1

    T_std = _sinkhorn_torch(p, q, C, reg, max_iter=200)
    T_lr = sinkhorn_lowrank(p, q, C, rank=min(N, K), reg=reg, max_iter=500)

    # Should be reasonably close when rank = min(N,K)
    rel_err = torch.linalg.norm(T_lr - T_std) / torch.linalg.norm(T_std)
    assert rel_err < 0.3, f"Relative error {rel_err:.4f} too large"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lowrank.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `sinkhorn_lowrank`**

```python
# sgw/_lowrank.py
from __future__ import annotations

import torch


def sinkhorn_lowrank(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    rank: int,
    reg: float,
    max_iter: int = 300,
    tol: float = 1e-4,
    check_every: int = 10,
    semi_relaxed: bool = False,
    rho: float = 1.0,
) -> torch.Tensor:
    """Low-rank Sinkhorn via Dykstra's algorithm.

    Approximates the optimal transport plan T as a low-rank matrix
    T = diag(u) @ Q @ diag(v), where Q is (N, K) of rank <= r,
    computed without ever materializing a full (N, K) plan during
    the iteration.

    Based on Scetbon, Cuturi & Peyre (2021): "Low-Rank Sinkhorn Factorization".

    Parameters
    ----------
    a : (N,) source marginal
    b : (K,) target marginal
    C : (N, K) cost matrix
    rank : int, target rank r
    reg : float, entropic regularization
    max_iter : int
    tol : float, convergence threshold on marginal error
    check_every : int
    semi_relaxed : bool
    rho : float, KL penalty for semi-relaxed mode

    Returns
    -------
    T : (N, K) transport plan (dense, reconstructed from factors)
    """
    N, K = C.shape
    r = min(rank, N, K)
    dtype = C.dtype
    device = C.device

    tau = rho / (rho + reg) if semi_relaxed else 1.0

    # ── Initialize low-rank factors via cost-informed softmax ──
    # Select r column/row indices spread across the cost matrix
    col_idx = torch.linspace(0, K - 1, r, device=device).long()
    row_idx = torch.linspace(0, N - 1, r, device=device).long()

    # Q: (N, r), R: (K, r) — initialized via softmax of cost submatrix
    Q = torch.softmax(-C[:, col_idx] / reg, dim=0)  # (N, r)
    R = torch.softmax(-C[row_idx, :].T / reg, dim=0)  # (K, r)

    # Scaling vectors
    g = torch.ones(r, dtype=dtype, device=device) / r  # (r,)
    u = torch.ones(N, dtype=dtype, device=device)  # (N,)
    v = torch.ones(K, dtype=dtype, device=device)  # (K,)

    # Dykstra dual variables (for the three projection sets)
    y1 = torch.zeros(N, dtype=dtype, device=device)
    y2 = torch.zeros(K, dtype=dtype, device=device)

    for it in range(max_iter):
        # ── Project onto source marginal: T @ 1 = a ──
        # T @ 1 = u * (Q @ (g * (R^T @ v)))
        Rv = R.T @ v  # (r,)
        Qg = Q @ (g * Rv)  # (N,)
        u_new = a / Qg.clamp(min=1e-30)
        # Dykstra correction
        u = u_new * torch.exp(y1)
        y1 = y1 + torch.log(u_new.clamp(min=1e-30)) - torch.log(u.clamp(min=1e-30))

        # ── Project onto target marginal: T^T @ 1 = b ──
        # T^T @ 1 = v * (R @ (g * (Q^T @ u)))
        Qu = Q.T @ u  # (r,)
        Rg = R @ (g * Qu)  # (K,)
        v_raw = b / Rg.clamp(min=1e-30)
        # Apply tau for semi-relaxed
        v_new = v_raw ** tau
        # Dykstra correction
        v = v_new * torch.exp(y2)
        y2 = y2 + torch.log(v_new.clamp(min=1e-30)) - torch.log(v.clamp(min=1e-30))

        # ── Update coupling scaling g ──
        Qu2 = Q.T @ u  # (r,)
        Rv2 = R.T @ v  # (r,)
        g = 1.0 / (Qu2 * Rv2).clamp(min=1e-30)
        g = g / g.sum()  # normalize

        # ── Convergence check ──
        if tol > 0 and (it + 1) % check_every == 0:
            # Reconstruct marginals
            row_marginal = u * (Q @ (g * (R.T @ v)))
            err = torch.abs(row_marginal - a).max().item()
            if err < tol:
                break

    # ── Reconstruct T ──
    # T = diag(u) @ Q @ diag(g) @ R^T @ diag(v)
    # = (u.unsqueeze(1) * Q) @ diag(g) @ (v.unsqueeze(1) * R)^T
    UQ = u.unsqueeze(1) * Q  # (N, r)
    VR = v.unsqueeze(1) * R  # (K, r)
    T = UQ @ torch.diag(g) @ VR.T  # (N, K)

    return T
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lowrank.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_lowrank.py tests/test_lowrank.py
git commit -m "feat: add sinkhorn_lowrank in _lowrank.py"
```

---

## Task 4: Wire multi-scale into `_solver.py`

**Files:**
- Modify: `sgw/_solver.py`
- Modify: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_solver.py`:

```python
def test_multiscale_basic(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, multiscale=True, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_multiscale_with_precomputed(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        multiscale=True,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_multiscale_custom_n_coarse(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, multiscale=True, n_coarse=30,
                   s_shared=50, M=30, max_iter=10)
    assert T.shape == (150, 150)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_solver.py::test_multiscale_basic -v`
Expected: FAIL (parameter not accepted)

- [ ] **Step 3: Implement multi-scale in `_solver.py`**

Add `multiscale: bool = False` and `n_coarse: int | None = None` to the `sampled_gw` signature (keyword-only, after existing params).

Add the following block **after input coercion and N/K inference, before provider construction** (around line 275):

```python
    # ── Multi-scale warm start ──────────────────────────────────────
    if multiscale and X_source is not None and X_target is not None:
        from sgw._multiscale import fps_downsample, upsample_plan

        _n_coarse = n_coarse if n_coarse is not None else min(500, N // 4, K // 4)
        _n_coarse = max(_n_coarse, 10)  # floor

        if _n_coarse < N and _n_coarse < K:
            # Downsample
            idx_src, assign_src = fps_downsample(X_source, _n_coarse)
            idx_tgt, assign_tgt = fps_downsample(X_target, _n_coarse)

            X_src_coarse = X_source[idx_src]
            X_tgt_coarse = X_target[idx_tgt]

            # Coarse C_linear if FGW
            C_lin_coarse = None
            if C_linear_t is not None and fgw_alpha > 0:
                C_lin_coarse = C_linear_t[idx_src][:, idx_tgt]

            # Coarse dist matrices if precomputed
            dist_src_coarse = None
            dist_tgt_coarse = None
            if dist_source is not None and dist_target is not None:
                dist_src_coarse = dist_source[idx_src][:, idx_src]
                dist_tgt_coarse = dist_target[idx_tgt][:, idx_tgt]

            # Solve coarse problem (recursive call, no multiscale)
            T_coarse = sampled_gw(
                X_src_coarse, X_tgt_coarse,
                distance_mode=distance_mode,
                dist_source=dist_src_coarse, dist_target=dist_tgt_coarse,
                n_landmarks=n_landmarks,
                fgw_alpha=fgw_alpha, C_linear=C_lin_coarse,
                s_shared=min(_n_coarse, s_shared) if s_shared else None,
                M=min(M, _n_coarse // 2),
                alpha=alpha, max_iter=max_iter, tol=tol, epsilon=epsilon,
                k=min(k, _n_coarse - 1),
                device=device, rank=rank,
                semi_relaxed=semi_relaxed, rho=rho,
            )
```

Then, **replace the line** `T_real = torch.outer(p_real, q_real)` (line ~340) with:

```python
    # Initial coupling
    if multiscale and X_source is not None and X_target is not None and _n_coarse < N and _n_coarse < K:
        T_real = upsample_plan(T_coarse, assign_src, assign_tgt, p_real, q_real)
    else:
        T_real = torch.outer(p_real, q_real)
```

Also update the docstring to document the new parameters.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_solver.py -v`
Expected: all PASS (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add sgw/_solver.py tests/test_solver.py
git commit -m "feat: wire multi-scale warm start into sampled_gw"
```

---

## Task 5: Wire low-rank into `_solver.py`

**Files:**
- Modify: `sgw/_solver.py`
- Modify: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_solver.py`:

```python
def test_rank_basic(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, rank=10, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_rank_with_precomputed(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        rank=10,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_multiscale_and_rank(two_datasets):
    """Both features combined."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        multiscale=True,
        rank=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_solver.py::test_rank_basic -v`
Expected: FAIL (parameter not accepted)

- [ ] **Step 3: Implement rank dispatch in `_solver.py`**

Add `rank: int | None = None` to the `sampled_gw` signature.

Change the Sinkhorn function selection (around line 357):

```python
    # Select Sinkhorn implementation
    if rank is not None:
        from sgw._lowrank import sinkhorn_lowrank
        def _sinkhorn_fn(a, b, C, reg, **kw):
            kw.pop('check_every', None)
            return sinkhorn_lowrank(a, b, C, rank=rank, reg=reg, **kw)
    elif differentiable:
        _sinkhorn_fn = _sinkhorn_differentiable
    else:
        _sinkhorn_fn = _sinkhorn_torch
```

Note: `sinkhorn_lowrank` returns (N, K) not (N+1, K+1). So when `rank is not None`, we need to call it on the **non-augmented** cost matrix and marginals. Modify the Sinkhorn call block:

```python
            # Sinkhorn
            if rank is not None:
                T_new = _sinkhorn_fn(p_real, q_real, Lambda.to(torch.float64),
                                     current_reg,
                                     semi_relaxed=semi_relaxed, rho=rho)
            else:
                T_aug = _sinkhorn_fn(p_aug, q_aug, Lambda_aug, current_reg,
                                     semi_relaxed=semi_relaxed, rho=rho)
                T_new = T_aug[:-1, :-1]
```

When using low-rank, we skip the slack augmentation (low-rank Sinkhorn doesn't use slack variables — partial transport is handled differently). This is acceptable for v0.3.0; partial transport support with low-rank can be added later.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_solver.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add sgw/_solver.py tests/test_solver.py
git commit -m "feat: wire low-rank Sinkhorn into sampled_gw"
```

---

## Task 6: Bump version and full test suite

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

Change `version = "0.2.1"` to `version = "0.3.0"` in `pyproject.toml`.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.3.0"
```
