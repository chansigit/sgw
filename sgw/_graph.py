import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(X: np.ndarray, k: int = 30) -> csr_matrix:
    """Build a k-NN graph, stitching disconnected components if needed.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : number of nearest neighbors

    Returns
    -------
    graph : csr_matrix of shape (n_samples, n_samples)
        Symmetric sparse distance graph, guaranteed connected.
    """
    graph = kneighbors_graph(X, k, mode="distance", n_jobs=-1)
    n_components, labels = connected_components(graph, directed=False)
    if n_components > 1:
        graph = _stitch_components(graph, X, labels, n_components)
    return graph.astype(np.float32)


def _stitch_components(
    graph: csr_matrix,
    X: np.ndarray,
    labels: np.ndarray,
    n_components: int,
) -> csr_matrix:
    """Add bridge edges between disconnected components."""
    graph = graph.tolil()
    while n_components > 1:
        comp0 = np.where(labels == 0)[0]
        others = np.where(labels != 0)[0]
        tree = KDTree(X[others])
        dist, ind = tree.query(X[comp0], k=1)
        dist = np.ravel(dist)
        ind = np.ravel(ind)
        best = int(np.argmin(dist))
        pt_from, pt_to = comp0[best], others[ind[best]]
        edge_dist = float(dist[best])
        graph[pt_from, pt_to] = edge_dist
        graph[pt_to, pt_from] = edge_dist
        n_components, labels = connected_components(graph.tocsr(), directed=False)
    return graph.tocsr()
