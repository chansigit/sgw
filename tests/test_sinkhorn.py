"""Unit tests for Sinkhorn internals: warm-start, PyTorch fallback, custom marginals."""
import numpy as np
import torch
import pytest

from torchgw._solver import _sinkhorn_torch, _sinkhorn_loop_pytorch


# ── Warm-start attributes ────────────────────────────────────────────

def test_sinkhorn_torch_sets_warm_start_attributes():
    """_sinkhorn_torch should stash _log_u and _log_v on the returned T."""
    N, K = 20, 25
    a = torch.ones(N, dtype=torch.float64) / N
    b = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)

    T = _sinkhorn_torch(a, b, C, reg=0.1, max_iter=50)

    assert hasattr(T, '_log_u'), "T should have _log_u attribute for warm-start"
    assert hasattr(T, '_log_v'), "T should have _log_v attribute for warm-start"
    assert T._log_u.shape == (N,)
    assert T._log_v.shape == (K,)
    assert torch.all(torch.isfinite(T._log_u))
    assert torch.all(torch.isfinite(T._log_v))


def test_warm_start_converges_faster():
    """Warm-started Sinkhorn should converge in fewer iterations than cold start."""
    N, K = 40, 50
    a = torch.ones(N, dtype=torch.float64) / N
    b = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)
    reg = 0.05

    # Cold start: count iterations to convergence
    log_K = -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))

    cold_iters = 0
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for i in range(200):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        if (i + 1) % 5 == 0:
            marginal = torch.exp(log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1))
            err = torch.abs(marginal - a).max().item()
            if err < 1e-6:
                cold_iters = i + 1
                break
    else:
        cold_iters = 200

    # Warm start: use converged potentials as init
    warm_iters = 0
    log_u_w = log_u.clone()
    log_v_w = log_v.clone()
    # Perturb cost slightly (simulating next GW iteration)
    C2 = C + 0.01 * torch.rand_like(C)
    log_K2 = -C2 / reg
    for i in range(200):
        log_u_w = log_a - torch.logsumexp(log_K2 + log_v_w.unsqueeze(0), dim=1)
        log_v_w = log_b - torch.logsumexp(log_K2 + log_u_w.unsqueeze(1), dim=0)
        if (i + 1) % 5 == 0:
            marginal = torch.exp(log_u_w + torch.logsumexp(log_K2 + log_v_w.unsqueeze(0), dim=1))
            err = torch.abs(marginal - a).max().item()
            if err < 1e-6:
                warm_iters = i + 1
                break
    else:
        warm_iters = 200

    assert warm_iters < cold_iters, (
        f"Warm start ({warm_iters} iters) should be faster than cold ({cold_iters} iters)"
    )


def test_sinkhorn_warm_start_integration():
    """_sinkhorn_torch with warm-start init should produce valid T."""
    N, K = 20, 25
    a = torch.ones(N, dtype=torch.float64) / N
    b = torch.ones(K, dtype=torch.float64) / K
    C = torch.rand(N, K, dtype=torch.float64)

    T1 = _sinkhorn_torch(a, b, C, reg=0.1, max_iter=50)
    # Use potentials from T1 as warm-start for a similar problem
    C2 = C + 0.01 * torch.rand_like(C)
    T2 = _sinkhorn_torch(a, b, C2, reg=0.1, max_iter=50,
                         log_u_init=T1._log_u, log_v_init=T1._log_v)

    assert T2.shape == (N, K)
    assert torch.all(T2 >= 0)
    assert torch.all(torch.isfinite(T2))


# ── PyTorch fallback explicit test ───────────────────────────────────

def test_sinkhorn_pytorch_fallback_balanced():
    """_sinkhorn_loop_pytorch should converge for balanced Sinkhorn."""
    N, K = 30, 40
    a = torch.ones(N, dtype=torch.float64) / N
    log_K = -torch.rand(N, K, dtype=torch.float64) / 0.1
    log_a = torch.log(a)
    log_b = torch.log(torch.ones(K, dtype=torch.float64) / K)

    log_u, log_v = _sinkhorn_loop_pytorch(
        log_K, log_a, log_b, tau_a=1.0, tau_b=1.0, max_iter=200, tol=1e-6,
        check_every=10, a=a,
    )
    # Verify marginal constraint
    marginal = torch.exp(log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1))
    assert torch.allclose(marginal, a, atol=1e-5)


def test_sinkhorn_pytorch_fallback_unbalanced():
    """_sinkhorn_loop_pytorch should work with tau < 1 (semi-relaxed)."""
    N, K = 20, 25
    a = torch.ones(N, dtype=torch.float64) / N
    log_K = -torch.rand(N, K, dtype=torch.float64) / 0.1
    log_a = torch.log(a)
    log_b = torch.log(torch.ones(K, dtype=torch.float64) / K)

    log_u, log_v = _sinkhorn_loop_pytorch(
        log_K, log_a, log_b, tau_a=1.0, tau_b=0.5, max_iter=200, tol=1e-6,
        check_every=10, a=a,
    )
    assert torch.all(torch.isfinite(log_u))
    assert torch.all(torch.isfinite(log_v))


# ── Custom marginals ────────────────────────────────────────────────

def test_custom_marginals():
    """sampled_gw should respect non-uniform marginals p and q."""
    from torchgw._solver import sampled_gw

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    Y = rng.normal(size=(50, 3)).astype(np.float32)

    p = np.abs(rng.normal(size=40)).astype(np.float64)
    p /= p.sum()
    q = np.abs(rng.normal(size=50)).astype(np.float64)
    q /= q.sum()

    T = sampled_gw(X, Y, p=p, q=q, s_shared=30, M=20, max_iter=10)
    assert T.shape == (40, 50)
    assert torch.all(T >= 0)
    assert T.sum().item() <= 1.0 + 1e-6


# ── Tiny problem edge case ──────────────────────────────────────────

def test_tiny_problem():
    """Solver should handle very small problems (N=5, K=7)."""
    from torchgw._solver import sampled_gw

    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 2)).astype(np.float32)
    Y = rng.normal(size=(7, 2)).astype(np.float32)

    T = sampled_gw(X, Y, s_shared=5, M=5, k=3, max_iter=20)
    assert T.shape == (5, 7)
    assert torch.all(T >= 0)
    assert torch.all(torch.isfinite(T))


# ── Sinkhorn with float32 input ─────────────────────────────────────

def test_sinkhorn_torch_float32():
    """_sinkhorn_torch should work with float32 inputs (mixed_precision path)."""
    N, K = 20, 25
    a = (torch.ones(N, dtype=torch.float32) / N)
    b = (torch.ones(K, dtype=torch.float32) / K)
    C = torch.rand(N, K, dtype=torch.float32)

    T = _sinkhorn_torch(a, b, C, reg=0.1, max_iter=50)
    assert T.dtype == torch.float32
    assert T.shape == (N, K)
    assert torch.all(T >= 0)
    assert torch.all(torch.isfinite(T))


def test_sinkhorn_unbalanced_two_sided_marginals():
    """With small rho_a == rho_b, both source and target marginals relax."""
    import torch
    from torchgw._solver import _sinkhorn_torch
    n, m = 30, 40
    a = torch.full((n,), 1.0 / n, dtype=torch.float64)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64)
    rng = torch.Generator().manual_seed(0)
    C = torch.rand(n, m, generator=rng, dtype=torch.float64)
    reg = 0.05
    # Balanced reference
    T_balanced = _sinkhorn_torch(a, b, C.clone(), reg, max_iter=500, tol=0,
                                 semi_relaxed=False, rho_a=1.0, rho_b=1.0)
    # Fully unbalanced (small rho relaxes both sides)
    T_unbal = _sinkhorn_torch(a, b, C.clone(), reg, max_iter=500, tol=0,
                              semi_relaxed=True, rho_a=0.1, rho_b=0.1)
    src_err_balanced = (T_balanced.sum(dim=1) - a).abs().max().item()
    src_err_unbal = (T_unbal.sum(dim=1) - a).abs().max().item()
    assert src_err_balanced < 1e-3, f"balanced source err {src_err_balanced}"
    assert src_err_unbal > 5e-3, f"unbalanced source should drift, got {src_err_unbal}"
    tgt_err_unbal = (T_unbal.sum(dim=0) - b).abs().max().item()
    assert tgt_err_unbal > 5e-3, f"unbalanced target should drift, got {tgt_err_unbal}"
