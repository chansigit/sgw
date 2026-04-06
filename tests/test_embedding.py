import numpy as np
import pytest
from torchgw._graph import build_knn_graph
from torchgw._solver import sampled_gw
from torchgw._embedding import joint_embedding


@pytest.fixture
def aligned_data(two_datasets):
    """Pre-compute graphs and transport plan for embedding tests."""
    X_src, X_tgt = two_datasets
    g_src = build_knn_graph(X_src, k=10)
    g_tgt = build_knn_graph(X_tgt, k=10)
    np.random.seed(0)
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=30, epsilon=0.01)
    # joint_embedding expects numpy arrays and graphs
    T_np = T.cpu().numpy()
    return X_src, X_tgt, g_src, g_tgt, T_np


def test_joint_embedding_shapes(aligned_data):
    X_src, X_tgt, g_src, g_tgt, T = aligned_data
    result = joint_embedding(
        anchor_name="tgt",
        data_by_name={"src": X_src, "tgt": X_tgt},
        graphs_by_name={"src": g_src, "tgt": g_tgt},
        transport_plans={("src", "tgt"): T},
        out_dim=5,
    )
    assert "src" in result and "tgt" in result
    assert result["src"].shape == (150, 5)
    assert result["tgt"].shape == (150, 5)


def test_joint_embedding_finite(aligned_data):
    X_src, X_tgt, g_src, g_tgt, T = aligned_data
    result = joint_embedding(
        anchor_name="tgt",
        data_by_name={"src": X_src, "tgt": X_tgt},
        graphs_by_name={"src": g_src, "tgt": g_tgt},
        transport_plans={("src", "tgt"): T},
        out_dim=5,
    )
    for name, emb in result.items():
        assert np.all(np.isfinite(emb)), f"{name} has non-finite values"
