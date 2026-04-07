"""Triton-fused Sinkhorn kernels for log-domain row/column updates.

Fuses the logsumexp reduction (amax + sub + exp + sum + log + sub) into a
single kernel pass per row/column, eliminating intermediate N×K matrices
and reducing kernel launch overhead from ~6 to 1 per Sinkhorn half-step.

Falls back gracefully: if Triton is unavailable, import will fail and
the caller should use the pure-PyTorch path instead.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_row_kernel(
    log_K_ptr,   # (N, K) log-kernel matrix
    log_v_ptr,   # (K,)   column dual
    log_a_ptr,   # (N,)   log source marginal
    log_u_ptr,   # (N,)   output: row dual
    N, K,
    stride_k_n,  # log_K row stride
    stride_k_k,  # log_K col stride (usually 1)
    BLOCK_K: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Compute log_u[n] = log_a[n] - logsumexp_k(log_K[n,k] + log_v[k])."""
    n = tl.program_id(0)
    if n >= N:
        return

    log_a_n = tl.load(log_a_ptr + n)

    # Online logsumexp: single pass, numerically stable
    # Initialize m and s in the correct dtype to satisfy Triton's type checker
    if USE_FP64:
        m = tl.full([], -float("inf"), dtype=tl.float64)
        s = tl.full([], 0.0, dtype=tl.float64)
    else:
        m = tl.full([], -float("inf"), dtype=tl.float32)
        s = tl.full([], 0.0, dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        log_k_val = tl.load(log_K_ptr + n * stride_k_n + k_offs * stride_k_k, mask=mask, other=-float("inf"))
        log_v_val = tl.load(log_v_ptr + k_offs, mask=mask, other=-float("inf"))
        x = log_k_val + log_v_val

        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    lse = m + tl.log(s)
    tl.store(log_u_ptr + n, log_a_n - lse)


@triton.jit
def _sinkhorn_col_kernel(
    log_K_ptr,   # (N, K)
    log_u_ptr,   # (N,)   row dual
    log_b_ptr,   # (K,)   log target marginal
    log_v_ptr,   # (K,)   output: column dual (raw, before tau blend)
    N, K,
    stride_k_n,
    stride_k_k,
    BLOCK_N: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Compute log_v_raw[k] = log_b[k] - logsumexp_n(log_K[n,k] + log_u[n])."""
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
        log_k_val = tl.load(log_K_ptr + n_offs * stride_k_n + k * stride_k_k, mask=mask, other=-float("inf"))
        log_u_val = tl.load(log_u_ptr + n_offs, mask=mask, other=-float("inf"))
        x = log_k_val + log_u_val

        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    lse = m + tl.log(s)
    tl.store(log_v_ptr + k, log_b_k - lse)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated log-domain Sinkhorn. Drop-in replacement for _sinkhorn_loop."""
    N, K = log_K.shape
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
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

        if tol > 0 and done % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            marginal_err = torch.abs(torch.exp(log_marginal) - a).max().item()
            if verbose:
                print(f"    sinkhorn {done:>4}/{max_iter} | marginal_err: {marginal_err:.4e}")
            if marginal_err < tol:
                if verbose:
                    print(f"    sinkhorn converged at {done} (err={marginal_err:.4e})")
                break

    return log_u, log_v
