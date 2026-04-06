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
        dists = torch.cdist(X, X[next_idx : next_idx + 1]).squeeze(1)
        min_dists = torch.minimum(min_dists, dists)
        next_idx = int(torch.argmax(min_dists).item())

    indices = torch.tensor(selected, dtype=torch.long, device=device)

    # Assign each point to its nearest selected point
    D_sel = torch.cdist(X, X[indices])
    assignments = torch.argmin(D_sel, dim=1)

    return indices, assignments


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
