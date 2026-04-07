"""Shared utilities for TU benchmark."""
import time
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_tu(name, root="data/TU", max_graphs=50):
    all_graphs = list(np.load(f"{root}/{name}_graphs.npz", allow_pickle=True)["graphs"])
    if len(all_graphs) > max_graphs:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(all_graphs), max_graphs, replace=False))
        graphs = [all_graphs[i] for i in idx]
    else:
        graphs = all_graphs
    return graphs


def sp_dist(g):
    n = g["num_nodes"]
    r, c = g["edge_index"][0], g["edge_index"][1]
    adj = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    D = shortest_path(adj, directed=False).astype(np.float32)
    inf = np.isinf(D)
    if np.any(inf):
        D[inf] = D[~inf].max() * 1.5 if np.any(~inf) else 1.0
    return D


def feat_cost(f1, f2):
    return cdist(f1, f2, "sqeuclidean").astype(np.float32)


def pairwise_distances(sp_dists, dist_fn, desc=""):
    N = len(sp_dists)
    D = np.zeros((N, N), dtype=np.float64)
    done = 0
    total = N * (N - 1) // 2
    t0 = time.time()
    for i in range(N):
        for j in range(i + 1, N):
            t_pair = time.time()
            d = dist_fn(i, j)
            dt = time.time() - t_pair
            D[i, j] = d
            D[j, i] = d
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate
            if done <= 3 or done % 100 == 0 or done == total:
                print(f"    [{desc}] {done}/{total} | "
                      f"pair ({i},{j}) cost={d:.6f} {dt:.2f}s/pair | "
                      f"elapsed {elapsed:.0f}s ETA {eta:.0f}s", flush=True)
    print(f"    [{desc}] DONE {total} pairs in {time.time()-t0:.1f}s", flush=True)
    return D


def classify(D, labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    D_norm = D / (D.max() + 1e-10)
    best_acc, best_std = 0, 0
    for gamma in (0.01, 0.1, 1.0, 10.0):
        K = np.exp(-gamma * D_norm)
        np.fill_diagonal(K, 1.0)
        svm = SVC(kernel="precomputed", C=10.0)
        cv = StratifiedKFold(n_splits=min(10, min(np.bincount(y))), shuffle=True, random_state=42)
        scores = cross_val_score(svm, K, y, cv=cv, scoring="accuracy")
        if scores.mean() > best_acc:
            best_acc, best_std = scores.mean(), scores.std()
    return best_acc, best_std
