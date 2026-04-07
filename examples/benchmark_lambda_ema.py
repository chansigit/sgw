"""Benchmark Lambda EMA variance reduction on spiral → Swiss roll.

Compares baseline (no EMA) vs several beta values.
Measures: convergence curve, final GW cost, Spearman rho, wall time.

Usage:
    PYTHONPATH=. python examples/benchmark_lambda_ema.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

from torchgw import sampled_gw, build_knn_graph


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


def gw_cost_from_dists(C1, C2, T):
    """Compute GW cost from precomputed distance matrices."""
    # GW = sum_{i,k} Lambda_{ik} T_{ik}
    # where Lambda_{ik} = sum_j p_j C1_{ij}^2 + sum_l q_l C2_{kl}^2 - 2 (C1 T C2^T)_{ik}
    p = T.sum(axis=1)
    q = T.sum(axis=0)
    term_A = (C1 ** 2) @ p
    term_C = (C2 ** 2) @ q
    term_B = -2 * C1 @ T @ C2.T
    Lambda = term_A[:, None] + term_C[None, :] + np.diag(term_B).reshape(-1, 1)
    # Simpler: use full 4D formula via factored form
    return float(np.sum(term_A[:, None] * T) + np.sum(term_C[None, :] * T)
                 - 2 * np.trace(C1 @ T @ C2.T @ T.T))


# ── Experiment ───────────────────────────────────────────────────────────

def run_benchmark(n_src, n_tgt, k, M, max_iter, betas, n_repeats=5):
    """Run sampled_gw with different EMA betas, repeated for variance estimate."""
    spiral, a_src = sample_spiral(n_src)
    swiss_roll, a_tgt = sample_swiss_roll(n_tgt)

    g_src = build_knn_graph(spiral, k=k)
    g_tgt = build_knn_graph(swiss_roll, k=k)

    # Precompute full distances for GW cost evaluation
    C1 = dijkstra(g_src, directed=False).astype(np.float64)
    C2 = dijkstra(g_tgt, directed=False).astype(np.float64)
    for C in [C1, C2]:
        inf_mask = np.isinf(C)
        if np.any(inf_mask):
            C[inf_mask] = C[~inf_mask].max() * 1.5
    C1 /= C1.max()
    C2 /= C2.max()

    configs = [("baseline", None)] + [(f"beta={b}", b) for b in betas]
    results = {}

    for label, beta in configs:
        print(f"\n  {label} ({'no EMA' if beta is None else f'lambda_ema_beta={beta}'})")
        all_err = []
        all_rho = []
        all_gw = []
        all_time = []
        all_niter = []

        for r in range(n_repeats):
            np.random.seed(r * 1000)
            t0 = time.time()
            T, log = sampled_gw(
                spiral, swiss_roll,
                distance_mode="precomputed",
                dist_source=C1, dist_target=C2,
                s_shared=min(n_src, n_tgt),
                M=M, alpha=0.9, max_iter=max_iter,
                epsilon=0.005, k=k,
                verbose=False, log=True,
                lambda_ema_beta=beta,
            )
            elapsed = time.time() - t0
            T_np = T.cpu().numpy()
            rho, _ = spearmanr(a_src, a_tgt[T_np.argmax(axis=1)])

            all_err.append(log["err_list"])
            all_rho.append(rho)
            all_gw.append(log["gw_cost"])
            all_time.append(elapsed)
            all_niter.append(log["n_iter"])

        results[label] = {
            "err_curves": all_err,
            "rho": all_rho,
            "gw_cost": all_gw,
            "time": all_time,
            "n_iter": all_niter,
        }
        print(f"    rho:  {np.mean(all_rho):.4f} ± {np.std(all_rho):.4f}")
        print(f"    GW:   {np.mean(all_gw):.4e} ± {np.std(all_gw):.4e}")
        print(f"    iter: {np.mean(all_niter):.0f} ± {np.std(all_niter):.0f}")
        print(f"    time: {np.mean(all_time):.2f}s ± {np.std(all_time):.2f}s")

    return results, configs


def plot_results(results, configs, n_src, n_tgt, filename):
    """Plot convergence curves, GW cost distribution, and rho distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Panel 1: Convergence curves (median + IQR)
    ax = axes[0]
    for idx, (label, _) in enumerate(configs):
        errs = results[label]["err_curves"]
        max_len = max(len(e) for e in errs)
        # Pad shorter curves with their last value
        padded = np.array([e + [e[-1]] * (max_len - len(e)) for e in errs])
        median = np.median(padded, axis=0)
        q25 = np.percentile(padded, 25, axis=0)
        q75 = np.percentile(padded, 75, axis=0)
        iters = np.arange(1, max_len + 1)
        ax.plot(iters, median, label=label, color=colors[idx], linewidth=1.5)
        ax.fill_between(iters, q25, q75, alpha=0.15, color=colors[idx])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||T - T_prev||_F")
    ax.set_title("Convergence (median ± IQR)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Final GW cost (box plot)
    ax = axes[1]
    data = [results[label]["gw_cost"] for label, _ in configs]
    labels = [label for label, _ in configs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("GW Cost")
    ax.set_title("Final GW Cost Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Spearman rho (box plot)
    ax = axes[2]
    data = [results[label]["rho"] for label, _ in configs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Matching Quality Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Lambda EMA Benchmark — Spiral ({n_src}) → Swiss Roll ({n_tgt})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=150)
    print(f"\n  Plot saved → {filename}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_SRC, N_TGT = 400, 500
    K = 5
    M = 50
    MAX_ITER = 300
    BETAS = [0.3, 0.5, 0.7]
    N_REPEATS = 10

    print(f"Benchmark: spiral({N_SRC}) → swiss_roll({N_TGT}), M={M}, k={K}")
    print(f"  EMA betas: {BETAS}")
    print(f"  Repeats per config: {N_REPEATS}")

    results, configs = run_benchmark(
        N_SRC, N_TGT, K, M, MAX_ITER, BETAS, n_repeats=N_REPEATS,
    )

    plot_results(results, configs, N_SRC, N_TGT,
                 "examples/benchmark_lambda_ema.png")

    # Summary table
    print(f"\n{'='*78}")
    print(f"{'Config':<14} {'ρ mean':>8} {'ρ std':>8} {'GW mean':>12} {'GW std':>12} {'iters':>6} {'time':>8}")
    print(f"{'-'*78}")
    for label, _ in configs:
        r = results[label]
        print(f"{label:<14} {np.mean(r['rho']):>8.4f} {np.std(r['rho']):>8.4f} "
              f"{np.mean(r['gw_cost']):>12.4e} {np.std(r['gw_cost']):>12.4e} "
              f"{np.mean(r['n_iter']):>6.0f} {np.mean(r['time']):>7.2f}s")
    print(f"{'='*78}")
