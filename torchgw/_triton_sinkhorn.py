"""Triton-fused Sinkhorn kernels for log-domain row/column updates.

Fuses the logsumexp reduction (amax + sub + exp + sum + log + sub) into a
single kernel pass per row/column, eliminating intermediate N×K matrices
and reducing kernel launch overhead from ~6 to 1 per Sinkhorn half-step.

Also provides fused kernels for:
  - Transport plan materialization: T[n,k] = exp(log_u[n] + log_K[n,k] + log_v[k])
  - Marginal error check: max|exp(log_u + lse) - a| without materializing T

Falls back gracefully: if Triton is unavailable, import will fail and
the caller should use the pure-PyTorch path instead.
"""

import torch
import triton
import triton.language as tl


# ── Row update kernel ─────────────────────────────────────────────────

@triton.jit
def _sinkhorn_row_kernel(
    log_K_ptr, log_v_ptr, log_a_ptr, log_u_ptr,
    N, K, stride_k_n, stride_k_k,
    BLOCK_K: tl.constexpr, USE_FP64: tl.constexpr,
):
    """log_u[n] = log_a[n] - logsumexp_k(log_K[n,k] + log_v[k])."""
    n = tl.program_id(0)
    if n >= N:
        return

    log_a_n = tl.load(log_a_ptr + n)
    if USE_FP64:
        m = tl.full([], -float("inf"), dtype=tl.float64)
        s = tl.full([], 0.0, dtype=tl.float64)
    else:
        m = tl.full([], -float("inf"), dtype=tl.float32)
        s = tl.full([], 0.0, dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(log_K_ptr + n * stride_k_n + k_offs * stride_k_k, mask=mask, other=-float("inf")) \
            + tl.load(log_v_ptr + k_offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    tl.store(log_u_ptr + n, log_a_n - (m + tl.log(s)))


# ── Column update kernel ──────────────────────────────────────────────

@triton.jit
def _sinkhorn_col_kernel(
    log_K_ptr, log_u_ptr, log_b_ptr, log_v_ptr,
    N, K, stride_k_n, stride_k_k,
    BLOCK_N: tl.constexpr, USE_FP64: tl.constexpr,
):
    """log_v_raw[k] = log_b[k] - logsumexp_n(log_K[n,k] + log_u[n])."""
    k = tl.program_id(0)
    if k >= K:
        return

    log_b_k = tl.load(log_b_ptr + k)
    if USE_FP64:
        m = tl.full([], -float("inf"), dtype=tl.float64)
        s = tl.full([], 0.0, dtype=tl.float64)
    else:
        m = tl.full([], -float("inf"), dtype=tl.float32)
        s = tl.full([], 0.0, dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask = n_offs < N
        x = tl.load(log_K_ptr + n_offs * stride_k_n + k * stride_k_k, mask=mask, other=-float("inf")) \
            + tl.load(log_u_ptr + n_offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    tl.store(log_v_ptr + k, log_b_k - (m + tl.log(s)))


# ── Fused marginal error kernel ───────────────────────────────────────

@triton.jit
def _marginal_err_kernel(
    log_K_ptr, log_u_ptr, log_v_ptr, a_ptr, err_ptr,
    N, K, stride_k_n, stride_k_k,
    BLOCK_K: tl.constexpr, USE_FP64: tl.constexpr,
):
    """Compute marginal[n] = exp(log_u[n] + lse_n), then atomically max |marginal - a|.

    Avoids materializing the N×K transport plan.
    """
    n = tl.program_id(0)
    if n >= N:
        return

    log_u_n = tl.load(log_u_ptr + n)
    a_n = tl.load(a_ptr + n)

    if USE_FP64:
        m = tl.full([], -float("inf"), dtype=tl.float64)
        s = tl.full([], 0.0, dtype=tl.float64)
    else:
        m = tl.full([], -float("inf"), dtype=tl.float32)
        s = tl.full([], 0.0, dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(log_K_ptr + n * stride_k_n + k_offs * stride_k_k, mask=mask, other=-float("inf")) \
            + tl.load(log_v_ptr + k_offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    marginal_n = tl.exp(log_u_n + m + tl.log(s))
    abs_err = tl.abs(marginal_n - a_n)
    tl.atomic_max(err_ptr, abs_err)


# ── Fused T materialization kernel ────────────────────────────────────

@triton.jit
def _materialize_T_kernel(
    log_u_ptr, log_K_ptr, log_v_ptr, T_ptr,
    N, K, stride_k_n, stride_k_k, stride_t_n, stride_t_k,
    BLOCK_K: tl.constexpr,
):
    """T[n,k] = exp(log_u[n] + log_K[n,k] + log_v[k]) — no intermediate N×K."""
    n = tl.program_id(0)
    if n >= N:
        return

    log_u_n = tl.load(log_u_ptr + n)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        log_k = tl.load(log_K_ptr + n * stride_k_n + k_offs * stride_k_k, mask=mask, other=-float("inf"))
        log_v = tl.load(log_v_ptr + k_offs, mask=mask, other=-float("inf"))
        t_val = tl.exp(log_u_n + log_k + log_v)
        tl.store(T_ptr + n * stride_t_n + k_offs * stride_t_k, t_val, mask=mask)


# ── Main entry point ──────────────────────────────────────────────────

def triton_sinkhorn_loop(
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    tau: float,
    max_iter: int,
    tol: float,
    check_every: int,
    a: torch.Tensor,
    verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated log-domain Sinkhorn. Drop-in replacement for _sinkhorn_loop."""
    N, K = log_K.shape
    log_u = log_u_init.clone() if log_u_init is not None else torch.zeros_like(log_a)
    log_v = log_v_init.clone() if log_v_init is not None else torch.zeros_like(log_b)
    is_balanced = (tau == 1.0)
    use_fp64 = (log_K.dtype == torch.float64)

    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    stride_k_n = log_K.stride(0)
    stride_k_k = log_K.stride(1)

    log_v_raw = torch.empty_like(log_v) if not is_balanced else log_v

    done = 0
    while done < max_iter:
        _sinkhorn_row_kernel[(N,)](
            log_K, log_v, log_a, log_u,
            N, K, stride_k_n, stride_k_k,
            BLOCK_K=BLOCK_K, USE_FP64=use_fp64,
        )

        if is_balanced:
            _sinkhorn_col_kernel[(K,)](
                log_K, log_u, log_b, log_v,
                N, K, stride_k_n, stride_k_k,
                BLOCK_N=BLOCK_N, USE_FP64=use_fp64,
            )
        else:
            _sinkhorn_col_kernel[(K,)](
                log_K, log_u, log_b, log_v_raw,
                N, K, stride_k_n, stride_k_k,
                BLOCK_N=BLOCK_N, USE_FP64=use_fp64,
            )
            log_v = tau * log_v_raw + (1 - tau) * log_v

        done += 1

        # Fused convergence check — no N×K materialization
        if tol > 0 and done % check_every == 0:
            err_buf = torch.zeros(1, device=log_K.device, dtype=log_K.dtype)
            _marginal_err_kernel[(N,)](
                log_K, log_u, log_v, a, err_buf,
                N, K, stride_k_n, stride_k_k,
                BLOCK_K=BLOCK_K, USE_FP64=use_fp64,
            )
            marginal_err = err_buf.item()
            if verbose:
                print(f"    sinkhorn {done:>4}/{max_iter} | marginal_err: {marginal_err:.4e}")
            if marginal_err < tol:
                if verbose:
                    print(f"    sinkhorn converged at {done} (err={marginal_err:.4e})")
                break

    return log_u, log_v


def triton_materialize_T(
    log_u: torch.Tensor,
    log_K: torch.Tensor,
    log_v: torch.Tensor,
) -> torch.Tensor:
    """Fused T = exp(log_u[:,None] + log_K + log_v[None,:]) without intermediate."""
    N, K = log_K.shape
    T = torch.empty(N, K, device=log_K.device, dtype=log_K.dtype)
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _materialize_T_kernel[(N,)](
        log_u, log_K, log_v, T,
        N, K,
        log_K.stride(0), log_K.stride(1),
        T.stride(0), T.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return T
