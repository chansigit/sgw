"""Benchmark TorchGW vs CNT-GW on spiral → Swiss roll.

CNT-GW uses squared Euclidean cost directly (no graph distances).
TorchGW uses kNN graph + Dijkstra shortest-path distances.
This is a fair comparison: both solve GW alignment on the same data,
each using its natural cost function.

Usage:
    PYTHONPATH=. python examples/benchmark_vs_cntgw.py
"""
import sys
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Add egw-solvers to path
sys.path.insert(0, "/scratch/users/chensj16/projects/egw-solvers")

from torchgw import sampled_gw


# ── Data generators ──────────────────────────────────────────────────────

def sample_spiral(n, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float32), angles


def sample_swiss_roll(n, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = 0.1 * rng.uniform(size=n) * 10
    return np.stack((x, z, y), axis=1).astype(np.float32), angles


# ── CNT-GW solver wrapper ───────────────────────────────────────────────

def run_cntgw(X, Y, eps=0.1, n_outer=50, n_sinkhorn=100, approx_dims=10, device="cuda"):
    """Run CNT-GW with squared Euclidean cost (native CNT)."""
    from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW

    # Build solver on CPU (KeOps handles GPU internally via LazyTensor)
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    solver = QuadraticGW(
        X=X_t, Y=Y_t,
        eps=eps,
        numItermax=n_outer,
        SINK_ARGS={"numItermax": n_sinkhorn},
    )
    solver.solve()
    T = solver.transport_plan(lazy=False)
    return T


def run_cntgw_with_kpca(X, Y, eps=0.1, n_outer=50, n_sinkhorn=100, approx_dims=10, device="cuda"):
    """Run CNT-GW approach: graph distances → Kernel PCA → QuadraticGW on embeddings."""
    from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
    from utils.implementation.kernels import kernel_from_costmatrix
    from torchgw import build_knn_graph
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    # Build graph distances (same preprocessing as TorchGW)
    g_src = build_knn_graph(X, k=5)
    g_tgt = build_knn_graph(Y, k=5)
    C1 = sp_dijkstra(g_src, directed=False).astype(np.float32)
    C2 = sp_dijkstra(g_tgt, directed=False).astype(np.float32)
    for C in [C1, C2]:
        inf_mask = np.isinf(C)
        if np.any(inf_mask):
            C[inf_mask] = C[~inf_mask].max() * 1.5
    C1 /= C1.max(); C2 /= C2.max()

    # Kernel PCA: cost → kernel → eigendecomposition → embedding
    N, K = len(X), len(Y)
    C1_t = torch.from_numpy(C1).float()
    C2_t = torch.from_numpy(C2).float()
    a = torch.ones(N) / N
    b = torch.ones(K) / K

    # Convert cost to kernel, center, PCA
    K1 = kernel_from_costmatrix(C1_t, a=a, center=True)
    K2 = kernel_from_costmatrix(C2_t, a=b, center=True)

    # Eigendecomposition
    eigvals1, eigvecs1 = torch.linalg.eigh(K1)
    eigvals2, eigvecs2 = torch.linalg.eigh(K2)
    # Take top-d components
    d = approx_dims
    X_emb = eigvecs1[:, -d:] * eigvals1[-d:].clamp(min=0).sqrt()
    Y_emb = eigvecs2[:, -d:] * eigvals2[-d:].clamp(min=0).sqrt()

    # Run QuadraticGW on embeddings (squared Euclidean is now a proxy for graph distance)
    solver = QuadraticGW(
        X=X_emb, Y=Y_emb,
        eps=eps, numItermax=n_outer,
        SINK_ARGS={"numItermax": n_sinkhorn},
    )
    solver.solve()
    T = solver.transport_plan(lazy=False)
    return T


# ── TorchGW wrapper ─────────────────────────────────────────────────────

def run_torchgw(X, Y, M=50, max_iter=300, eps=0.001, mode="dijkstra", beta=None):
    """Run TorchGW with graph distances."""
    np.random.seed(42)
    T = sampled_gw(
        X, Y,
        distance_mode=mode,
        s_shared=min(len(X), len(Y)),
        M=M, alpha=0.9, max_iter=max_iter,
        epsilon=eps, k=5, verbose=False,
        lambda_ema_beta=beta,
    )
    return T.cpu()


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(T, a_src, a_tgt, label):
    T_np = T.numpy() if isinstance(T, torch.Tensor) else T
    matching = T_np.argmax(axis=1)
    rho, _ = spearmanr(a_src, a_tgt[matching])
    print(f"  {label:<35s}  ρ={rho:+.4f}  T_sum={T_np.sum():.4f}")
    return rho


# ── Main ─────────────────────────────────────────────────────────────────

def run_scale(n_src, n_tgt):
    print(f"\n{'='*65}")
    print(f"  Scale: spiral({n_src}) → swiss_roll({n_tgt})")
    print(f"{'='*65}")

    spiral, a_src = sample_spiral(n_src)
    swiss_roll, a_tgt = sample_swiss_roll(n_tgt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    # --- CNT-GW (squared Euclidean, native) ---
    try:
        t0 = time.time()
        T_cnt = run_cntgw(spiral, swiss_roll, eps=0.1, n_outer=50,
                          n_sinkhorn=100, device=device)
        t_cnt = time.time() - t0
        rho_cnt = evaluate(T_cnt, a_src, a_tgt, f"CNT-GW (Euclidean, {t_cnt:.2f}s)")
        results["CNT-GW (Euclidean)"] = {"rho": rho_cnt, "time": t_cnt, "T": T_cnt}
    except Exception as e:
        print(f"  CNT-GW (Euclidean) FAILED: {e}")

    # --- CNT-GW with KPCA of graph distances ---
    try:
        t0 = time.time()
        T_cnt_kpca = run_cntgw_with_kpca(spiral, swiss_roll, eps=0.1,
                                          n_outer=50, n_sinkhorn=100,
                                          approx_dims=10, device=device)
        t_cnt_kpca = time.time() - t0
        rho_cnt_kpca = evaluate(T_cnt_kpca, a_src, a_tgt, f"CNT-GW (KPCA graph, {t_cnt_kpca:.2f}s)")
        results["CNT-GW (KPCA)"] = {"rho": rho_cnt_kpca, "time": t_cnt_kpca, "T": T_cnt_kpca}
    except Exception as e:
        print(f"  CNT-GW (KPCA graph) FAILED: {e}")

    # --- TorchGW (Dijkstra) ---
    t0 = time.time()
    T_sgw = run_torchgw(spiral, swiss_roll, M=50, max_iter=300, eps=0.001)
    t_sgw = time.time() - t0
    rho_sgw = evaluate(T_sgw, a_src, a_tgt, f"TorchGW dijkstra ({t_sgw:.2f}s)")
    results["TorchGW (dijkstra)"] = {"rho": rho_sgw, "time": t_sgw, "T": T_sgw}

    # --- TorchGW (precomputed) ---
    t0 = time.time()
    T_pre = run_torchgw(spiral, swiss_roll, M=50, max_iter=300, eps=0.001, mode="precomputed")
    t_pre = time.time() - t0
    rho_pre = evaluate(T_pre, a_src, a_tgt, f"TorchGW precomputed ({t_pre:.2f}s)")
    results["TorchGW (precomputed)"] = {"rho": rho_pre, "time": t_pre, "T": T_pre}

    # --- TorchGW (landmark) ---
    t0 = time.time()
    T_lm = run_torchgw(spiral, swiss_roll, M=50, max_iter=300, eps=0.001, mode="landmark")
    t_lm = time.time() - t0
    rho_lm = evaluate(T_lm, a_src, a_tgt, f"TorchGW landmark ({t_lm:.2f}s)")
    results["TorchGW (landmark)"] = {"rho": rho_lm, "time": t_lm, "T": T_lm}

    # --- TorchGW + EMA ---
    t0 = time.time()
    T_ema = run_torchgw(spiral, swiss_roll, M=50, max_iter=300, eps=0.001, beta=0.5)
    t_ema = time.time() - t0
    rho_ema = evaluate(T_ema, a_src, a_tgt, f"TorchGW dijkstra+EMA ({t_ema:.2f}s)")
    results["TorchGW (dijkstra+EMA)"] = {"rho": rho_ema, "time": t_ema, "T": T_ema}

    return results, a_src, a_tgt, spiral


def plot_results(all_results, filename):
    """Bar chart: rho and time for each method at each scale."""
    scales = list(all_results.keys())
    fig, axes = plt.subplots(len(scales), 2, figsize=(14, 5 * len(scales)))
    if len(scales) == 1:
        axes = axes[np.newaxis, :]

    for row, scale in enumerate(scales):
        results = all_results[scale]
        methods = list(results.keys())
        rhos = [abs(results[m]["rho"]) for m in methods]
        times = [results[m]["time"] for m in methods]

        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        ax = axes[row, 0]
        bars = ax.bar(range(len(methods)), rhos, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("|Spearman ρ|")
        ax.set_title(f"{scale} — Matching Quality")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, rhos):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)

        ax = axes[row, 1]
        bars = ax.bar(range(len(methods)), times, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{scale} — Runtime")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.1f}s", ha="center", fontsize=8)

    fig.suptitle("TorchGW vs CNT-GW — Spiral → Swiss Roll", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=150)
    print(f"\nPlot saved → {filename}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = {}

    for n_src, n_tgt in [(400, 500), (2000, 2500)]:
        results, _, _, _ = run_scale(n_src, n_tgt)
        all_results[f"{n_src}v{n_tgt}"] = results

    plot_results(all_results, "examples/benchmark_vs_cntgw.png")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Scale':<12} {'Method':<30} {'|ρ|':>6} {'Time':>8}")
    print(f"{'-'*80}")
    for scale, results in all_results.items():
        for method, r in results.items():
            print(f"{scale:<12} {method:<30} {abs(r['rho']):>6.4f} {r['time']:>7.2f}s")
    print(f"{'='*80}")
