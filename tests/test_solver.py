import numpy as np
import torch
import pytest
from sgw._solver import sampled_gw


def test_sampled_gw_returns_tensor(two_datasets):
    """Output should be a torch.Tensor in v0.2.0."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_sampled_gw_accepts_tensor_input(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        torch.from_numpy(X_src), torch.from_numpy(X_tgt),
        s_shared=50, M=30, max_iter=10,
    )
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)


def test_sampled_gw_mass_bounded(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert T.sum().item() <= 1.0 + 1e-6


def test_distance_mode_dijkstra(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, distance_mode="dijkstra", s_shared=50, M=30, max_iter=10)
    assert T.shape == (150, 150)


def test_distance_mode_precomputed_with_matrices(two_datasets):
    from sgw._graph import build_knn_graph
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    X_src, X_tgt = two_datasets
    g_src = build_knn_graph(X_src, k=10)
    g_tgt = build_knn_graph(X_tgt, k=10)
    C_X = sp_dijkstra(csgraph=g_src, directed=False).astype(np.float32)
    C_Y = sp_dijkstra(csgraph=g_tgt, directed=False).astype(np.float32)

    T = sampled_gw(
        dist_source=C_X, dist_target=C_Y,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_distance_mode_precomputed_without_matrices(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_distance_mode_landmark(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="landmark",
        n_landmarks=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_invalid_distance_mode(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="distance_mode"):
        sampled_gw(X_src, X_tgt, distance_mode="invalid", M=30, max_iter=10)


def test_precomputed_missing_params():
    with pytest.raises(ValueError):
        sampled_gw(distance_mode="precomputed", M=30, max_iter=10)


def test_sampled_gw_identity_alignment():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5)).astype(np.float32)
    np.random.seed(2)
    T = sampled_gw(X, X.copy(), s_shared=30, M=20, max_iter=50, k=5,
                   alpha=0.5, epsilon=0.01)
    row_argmax = T.argmax(dim=1).cpu().numpy()
    diagonal_fraction = np.mean(row_argmax == np.arange(30))
    assert diagonal_fraction > 0.5, f"Only {diagonal_fraction:.0%} on diagonal"


def test_sampled_gw_log_returns_tuple(two_datasets):
    X_src, X_tgt = two_datasets
    result = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10, log=True)
    assert isinstance(result, tuple)
    T, log_dict = result
    assert isinstance(T, torch.Tensor)
    assert "err_list" in log_dict
    assert "n_iter" in log_dict
    assert "gw_cost" in log_dict


def test_public_import():
    from sgw import sampled_gw, build_knn_graph, joint_embedding
    assert callable(sampled_gw)
    assert callable(build_knn_graph)
    assert callable(joint_embedding)


def test_fused_gw(two_datasets):
    """Fused GW with fgw_alpha between 0 and 1."""
    X_src, X_tgt = two_datasets
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        X_src, X_tgt,
        fgw_alpha=0.5,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_pure_wasserstein(two_datasets):
    """fgw_alpha=1.0 should work without structural distances."""
    X_src, X_tgt = two_datasets
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        fgw_alpha=1.0,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_fgw_missing_c_linear(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="C_linear"):
        sampled_gw(X_src, X_tgt, fgw_alpha=0.5, M=30, max_iter=10)


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


def test_lowrank_basic(two_datasets):
    from sgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(X_src, X_tgt, rank=10, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_lowrank_with_precomputed(two_datasets):
    from sgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(
        X_src, X_tgt,
        rank=10,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_lowrank_and_multiscale(two_datasets):
    """Both features combined."""
    from sgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(
        X_src, X_tgt,
        multiscale=True,
        rank=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_public_import_lowrank():
    from sgw import sampled_lowrank_gw
    assert callable(sampled_lowrank_gw)


def test_differentiable_gradient_flows():
    """differentiable=True should allow gradients to flow through T."""
    X = torch.randn(20, 3)
    Y = torch.randn(25, 3)
    C_feat = torch.cdist(X, Y).requires_grad_(True)

    # Pure Wasserstein so Lambda = C_linear (differentiable end-to-end)
    T = sampled_gw(
        fgw_alpha=1.0, C_linear=C_feat,
        differentiable=True, M=10, max_iter=3, epsilon=0.1,
        device=torch.device("cpu"),
    )

    assert T.requires_grad, "T should require grad when differentiable=True"
    loss = (C_feat.detach() * T).sum()
    loss.backward()
    assert C_feat.grad is not None, "Gradient should flow to C_linear"
    assert C_feat.grad.shape == C_feat.shape


def test_semi_relaxed(two_datasets):
    """Semi-relaxed mode should produce a valid plan with relaxed target marginal."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        semi_relaxed=True, rho=1.0,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)
    # Source marginal should still be approximately enforced
    p = torch.ones(150, dtype=T.dtype, device=T.device) / 150
    row_sums = T.sum(dim=1)
    assert torch.allclose(row_sums, p, atol=0.01), \
        f"Source marginal error: {(row_sums - p).abs().max():.4e}"
