"""Benchmark three distance modes on the spiral → Swiss roll task.

Generates comparison plots and prints a summary table.

Usage:
    PYTHONPATH=. python examples/benchmark_distance_modes.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from torchgw import sampled_gw


# ── Data generators ──────────────────────────────────────────────────────

def sample_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float32), angles


def sample_swiss_roll(n, seed=1):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = 0.1 * rng.uniform(size=n) * 10
    return np.stack((x, z, y), axis=1).astype(np.float32), angles


# ── Run benchmark ────────────────────────────────────────────────────────

N_SRC, N_TGT = 400, 500
COMMON_KWARGS = dict(s_shared=400, M=80, alpha=0.8, max_iter=200, epsilon=0.005, k=5)

MODES = [
    ("dijkstra", {}),
    ("precomputed", {}),
    ("landmark", {"n_landmarks": 20}),
]

spiral, a_src = sample_spiral(N_SRC, seed=0)
swiss_roll, a_tgt = sample_swiss_roll(N_TGT, seed=1)

results = {}

for mode, extra_kwargs in MODES:
    np.random.seed(42)
    print(f"Running {mode}...", end=" ", flush=True)

    t0 = time.perf_counter()
    T, log_dict = sampled_gw(
        spiral, swiss_roll,
        distance_mode=mode,
        **extra_kwargs,
        **COMMON_KWARGS,
        log=True,
    )
    elapsed = time.perf_counter() - t0

    T_np = T.cpu().numpy()
    matched_angles = a_tgt[T_np.argmax(axis=1)]
    spearman_angle, _ = spearmanr(a_src, matched_angles)
    spearman_mono, _ = spearmanr(np.arange(N_SRC), T_np.argmax(axis=1))

    results[mode] = {
        "T": T_np,
        "time": elapsed,
        "n_iter": log_dict["n_iter"],
        "gw_cost": log_dict["gw_cost"],
        "spearman_angle": spearman_angle,
        "spearman_mono": spearman_mono,
    }
    print(f"{elapsed:.2f}s | rho={spearman_angle:.4f}")


# ── Print table ──────────────────────────────────────────────────────────

print(f"\n{'Mode':<14} {'Time (s)':>10} {'Iters':>7} {'GW Cost':>12} {'spearman_angle':>11} {'spearman_mono':>10}")
print("-" * 68)
for mode, r in results.items():
    print(f"{mode:<14} {r['time']:>10.2f} {r['n_iter']:>7d} {r['gw_cost']:>12.4e} "
          f"{r['spearman_angle']:>11.4f} {r['spearman_mono']:>10.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────

cmap = "Spectral"
angle_norm = plt.Normalize(min(a_src.min(), a_tgt.min()), max(a_src.max(), a_tgt.max()))

fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=100)

for col, (mode, _) in enumerate(MODES):
    r = results[mode]
    T_np = r["T"]
    matched = a_tgt[T_np.argmax(axis=1)]

    # Row 1: transport plan
    ax = axes[0, col]
    T_clip = np.clip(T_np, 1e-9, None)
    im = ax.imshow(T_clip, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-9, vmax=T_np.max()), interpolation="nearest")
    ax.set_xlabel("Swiss roll index")
    ax.set_ylabel("Spiral index")
    ax.set_title(f"{mode}\nGW={r['gw_cost']:.4e} | {r['time']:.2f}s",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: spiral colored by matched angle
    ax = axes[1, col]
    sc = ax.scatter(spiral[:, 0], spiral[:, 1],
                    c=matched, cmap=cmap, norm=angle_norm, s=12, edgecolors="none")
    ax.set_title(f"Matched angles\nSpearman rho = {r['spearman_angle']:.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.8)

    # Row 3: convergence
    ax = axes[2, col]
    # We don't have per-iteration errors stored in results, so show a bar chart
    bars = [r["time"], r["gw_cost"] * 1000, abs(r["spearman_angle"])]
    labels = [f"Time\n{r['time']:.2f}s", f"GW Cost\n{r['gw_cost']:.4e}", f"|rho|\n{abs(r['spearman_angle']):.4f}"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(labels, bars, color=colors, width=0.6)
    ax.set_title(f"{mode} summary", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value")

fig.suptitle(
    f"Distance Mode Comparison — Spiral ({N_SRC}) → Swiss Roll ({N_TGT})",
    fontsize=15, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig("examples/benchmark_distance_modes.png", dpi=100, bbox_inches="tight")
print(f"\nPlot saved → examples/benchmark_distance_modes.png")
plt.close(fig)
