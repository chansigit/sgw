"""Compare TorchGW vs POT entropic Gromov-Wasserstein on spiral → Swiss roll.

Both methods use identical preprocessing (same kNN graphs, same Dijkstra
shortest-path distances). Runs at two scales: 400/500 and 4000/5000.

Usage:
    PYTHONPATH=. python examples/demo_spiral_to_swissroll.py
"""
import sys

# Block TensorFlow auto-import by POT — prevents LLVM segfault
# on systems with conflicting LLVM versions.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ot
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

from torchgw import sampled_gw, build_knn_graph


# ── Helpers ───────────────────────────────────────────────────────────────

def sample_spiral(n, min_radius=0.3, max_radius=1.0,
                  min_angle=0, max_angle=9, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(min_radius, max_radius, n)
    angles = np.linspace(min_angle, max_angle, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float32), angles


def sample_swiss_roll(n, min_radius=0.3, max_radius=1.0, length=10,
                      min_angle=0, max_angle=9, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    radius = np.linspace(min_radius, max_radius, n)
    angles = np.linspace(min_angle, max_angle, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = 0.1 * rng.uniform(size=n) * length
    return np.stack((x, z, y), axis=1).astype(np.float32), angles


def gw_cost(C1, C2, T):
    """Compute GW distance: sum_{i,j,k,l} (C1_{ik} - C2_{jl})^2 T_{ij} T_{kl}."""
    p, q = T.sum(axis=1), T.sum(axis=0)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")
    return ot.gromov.gwloss(constC, hC1, hC2, T)


def run_experiment(n_src, n_tgt, k, run_pot=True):
    """Run both solvers and return results dict."""
    spiral, a_src = sample_spiral(n_src, seed=0)
    swiss_roll, a_tgt = sample_swiss_roll(n_tgt, seed=1)
    print(f"\n{'='*70}")
    print(f"  Scale: {n_src} vs {n_tgt}  (k={k})")
    print(f"{'='*70}")

    g_src = build_knn_graph(spiral, k=k)
    g_tgt = build_knn_graph(swiss_roll, k=k)

    # Full distance matrices (needed for POT and for gw_cost evaluation)
    t0 = time.time()
    C1 = dijkstra(g_src, directed=False)
    C2 = dijkstra(g_tgt, directed=False)
    for C in [C1, C2]:
        inf_mask = np.isinf(C)
        if np.any(inf_mask):
            C[inf_mask] = C[~inf_mask].max() * 1.5
    C1 /= C1.max()
    C2 /= C2.max()
    t_prep = time.time() - t0
    print(f"  Dijkstra prep: {t_prep:.2f}s")

    res = dict(spiral=spiral, swiss_roll=swiss_roll,
               a_src=a_src, a_tgt=a_tgt, n_src=n_src, n_tgt=n_tgt, k=k)

    # POT
    if run_pot:
        t0 = time.time()
        T_pot = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, loss_fun="square_loss", epsilon=0.005,
            max_iter=500, tol=1e-9, verbose=False, log=False,
        )
        t_pot = time.time() - t0
        gw_pot = gw_cost(C1, C2, T_pot)
        rho_pot, _ = spearmanr(a_src, a_tgt[T_pot.argmax(axis=1)])
        res.update(T_pot=T_pot, t_pot=t_pot, gw_pot=gw_pot, rho_pot=rho_pot)
        print(f"  POT:  {t_pot:>8.2f}s  |  GW={gw_pot:.4e}  |  ρ={rho_pot:.4f}")
    else:
        res.update(T_pot=None, t_pot=np.nan, gw_pot=np.nan, rho_pot=np.nan)
        print(f"  POT:  skipped (would need {n_src}x{n_src} + {n_tgt}x{n_tgt} matrices)")

    # TorchGW
    np.random.seed(42)
    t0 = time.time()
    T_sgw = sampled_gw(
        spiral, swiss_roll,
        distance_mode="precomputed",
        dist_source=C1, dist_target=C2,
        s_shared=min(n_src, n_tgt), M=80, alpha=0.8,
        max_iter=300, epsilon=0.005, k=k,
        verbose=True, verbose_every=100,
    )
    t_sgw = time.time() - t0
    T_sgw_np = T_sgw.cpu().numpy()
    gw_sgw = gw_cost(C1, C2, T_sgw_np)
    rho_sgw, _ = spearmanr(a_src, a_tgt[T_sgw_np.argmax(axis=1)])
    res.update(T_sgw=T_sgw, t_sgw=t_sgw, gw_sgw=gw_sgw, rho_sgw=rho_sgw,
               C1=C1, C2=C2)
    print(f"  TorchGW:  {t_sgw:>8.2f}s  |  GW={gw_sgw:.4e}  |  ρ={rho_sgw:.4f}")

    return res


def plot_comparison(res, filename):
    """Generate a 3-row × 2-col comparison figure."""
    spiral = res["spiral"]; swiss_roll = res["swiss_roll"]
    a_src = res["a_src"]; a_tgt = res["a_tgt"]
    n_src = res["n_src"]; n_tgt = res["n_tgt"]; k = res["k"]
    T_pot = res["T_pot"]; T_sgw = res["T_sgw"]

    cmap = "Spectral"
    angle_norm = plt.Normalize(
        min(a_src.min(), a_tgt.min()),
        max(a_src.max(), a_tgt.max()),
    )

    has_pot = T_pot is not None
    ncols = 2 if has_pot else 1
    fig, axes = plt.subplots(3, ncols, figsize=(7 * ncols, 14), dpi=100,
                             gridspec_kw={"height_ratios": [1, 0.85, 1]})
    if ncols == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing

    # ── Row 1: input data ────────────────────────────────────────────────
    ax = axes[0, 0]
    sc = ax.scatter(spiral[:, 0], spiral[:, 1],
                    c=a_src, cmap=cmap, norm=angle_norm, s=10, edgecolors="none")
    ax.set_title(f"Source: 2D Spiral  (n={n_src})", fontsize=13, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="angle (rad)", shrink=0.8)

    ax = axes[0, ncols - 1]
    # For 3D we replace the 2D axes with a 3D one at the same position
    pos = ax.get_position()
    ax.remove()
    ax3d = fig.add_axes(pos, projection="3d")
    ax3d.scatter(swiss_roll[:, 0], swiss_roll[:, 1], swiss_roll[:, 2],
                 c=a_tgt, cmap=cmap, norm=angle_norm, s=10, edgecolors="none")
    ax3d.view_init(7, -80)
    ax3d.set_title(f"Target: 3D Swiss Roll  (n={n_tgt})", fontsize=13, fontweight="bold")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("z"); ax3d.set_zlabel("y")

    # ── Row 2: transport plans (log-scale colorbar) ──────────────────────
    plans = []
    if has_pot:
        plans.append((T_pot, "POT Entropic GW", res["gw_pot"], res["t_pot"]))
    plans.append((T_sgw, "TorchGW (Sampled GW)", res["gw_sgw"], res["t_sgw"]))

    for col, (T, label, gw, t) in enumerate(plans):
        ax = axes[1, col]
        T_clip = np.clip(T, 1e-9, None)
        im = ax.imshow(T_clip, aspect="auto", cmap="magma",
                       norm=LogNorm(vmin=1e-9, vmax=T.max()), interpolation="nearest")
        ax.set_xlabel("Swiss roll index"); ax.set_ylabel("Spiral index")
        ax.set_title(f"{label}\nGW = {gw:.4e}  |  {t:.2f}s", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="coupling weight (log)", shrink=0.8)

    # ── Row 3: matching quality ──────────────────────────────────────────
    match_data = []
    if has_pot:
        match_data.append((a_tgt[T_pot.argmax(axis=1)], "POT", res["rho_pot"]))
    match_data.append((a_tgt[T_sgw.argmax(axis=1)], "TorchGW", res["rho_sgw"]))

    for col, (matched, label, rho) in enumerate(match_data):
        ax = axes[2, col]
        sc = ax.scatter(spiral[:, 0], spiral[:, 1],
                        c=matched, cmap=cmap, norm=angle_norm, s=10, edgecolors="none")
        ax.set_title(f"{label}: Spiral colored by matched angle\n"
                     f"Spearman $\\rho$ = {rho:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="matched angle (rad)", shrink=0.8)

    fig.suptitle(
        f"TorchGW vs POT  —  Spiral ({n_src}) → Swiss Roll ({n_tgt})"
        f"  |  kNN k={k} + Dijkstra",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(filename, dpi=100)
    print(f"  Plot saved → {filename}")
    plt.close(fig)


# ── Run experiments ───────────────────────────────────────────────────────

# Small scale: 400 vs 500
res_small = run_experiment(400, 500, k=5, run_pot=True)
plot_comparison(res_small, "examples/demo_spiral_to_swissroll_400v500.png")

# Large scale: 4000 vs 5000
res_large = run_experiment(4000, 5000, k=5, run_pot=True)
plot_comparison(res_large, "examples/demo_spiral_to_swissroll_4000v5000.png")

# ── Final summary table ──────────────────────────────────────────────────

print("\n" + "=" * 78)
print(f"{'Scale':<16} {'Method':<10} {'Time (s)':>10} {'GW dist':>14} {'Spearman ρ':>12}")
print("-" * 78)
for label, r in [("400 vs 500", res_small), ("4000 vs 5000", res_large)]:
    if r["T_pot"] is not None:
        print(f"{label:<16} {'POT':<10} {r['t_pot']:>10.2f} {r['gw_pot']:>14.4e} {r['rho_pot']:>12.4f}")
    print(f"{label:<16} {'TorchGW':<10} {r['t_sgw']:>10.2f} {r['gw_sgw']:>14.4e} {r['rho_sgw']:>12.4f}")
print("=" * 78)
