#!/usr/bin/env python3
"""Large-scale spiral-to-Swiss-roll benchmark for TorchGW.

Measures alignment quality (Spearman rho) and wall-clock time across
scales from 4k x 5k to 100k x 120k.

Usage:
    python examples/benchmark_scale.py              # default scales
    python examples/benchmark_scale.py --quick      # skip 100k scale
    python examples/benchmark_scale.py --csv        # CSV output

Requires: torch (with CUDA), torchgw, scipy, numpy
"""
import argparse
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from torchgw import sampled_gw


# ── Data generation ────────────────────────────────────────────────────

def sample_spiral(n, seed=0):
    """2D Archimedean spiral with Gaussian noise."""
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float64), angles


def sample_swiss_roll(n, seed=1):
    """3D Swiss roll (spiral + uniform height) with Gaussian noise."""
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = rng.uniform(size=n)
    return np.stack((x, z, y), axis=1).astype(np.float64), angles


# ── Benchmark runner ───────────────────────────────────────────────────

def run_benchmark(N, K, distance_mode, repeats=2, **solver_kw):
    """Run spiral-to-Swiss-roll alignment and measure quality + time."""
    X, angles_src = sample_spiral(N, seed=0)
    Y, angles_tgt = sample_swiss_roll(K, seed=1)

    kw = dict(
        epsilon=0.005, M=80, max_iter=300, mixed_precision=True,
        distance_mode=distance_mode, log=True, verbose=False,
    )
    kw.update(solver_kw)

    # Warmup (small problem, same code path)
    n_warm = min(500, N)
    _ = sampled_gw(X[:n_warm], Y[:n_warm], distance_mode=distance_mode,
                   max_iter=5, M=20, mixed_precision=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        T, log = sampled_gw(X, Y, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best_time = min(times)
    n_iter = log["n_iter"]

    # Quality: Spearman correlation of matched angles
    T_np = T.cpu().numpy()
    matched_angles = angles_tgt[T_np.argmax(axis=1)]
    rho, _ = spearmanr(angles_src, matched_angles)

    return dict(
        N=N, K=K, mode=distance_mode, time=best_time,
        n_iter=n_iter, spearman=abs(rho), mass=T.sum().item(),
    )


# ── Main ───────────────────────────────────────────────────────────────

SCALES = [
    # (N, K, distance_mode, extra_kwargs)
    (4_000,   5_000,   "landmark",    dict(k=5)),
    (4_000,   5_000,   "dijkstra",    dict(k=5)),
    (10_000,  12_000,  "landmark",    dict(k=5)),
    (15_000,  18_000,  "landmark",    dict(k=5)),
    (20_000,  25_000,  "landmark",    dict(k=5)),
    (25_000,  30_000,  "landmark",    dict(k=5, M=100)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="Skip the 100k scale")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV instead of table")
    args = parser.parse_args()

    scales = SCALES[:-1] if args.quick else SCALES

    # Device info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.0f} GB)")
    else:
        print("Running on CPU")
    print()

    results = []
    header = f"{'Scale':<20} {'Mode':<12} {'Time':>8} {'Iters':>6} {'Spearman':>9} {'Mass':>6}"

    if args.csv:
        print("N,K,mode,time_s,iters,spearman,mass")
    else:
        print(header)
        print("-" * len(header))

    for N, K, mode, extra_kw in scales:
        try:
            r = run_benchmark(N, K, mode, **extra_kw)
            results.append(r)
            if args.csv:
                print(f"{r['N']},{r['K']},{r['mode']},{r['time']:.2f},"
                      f"{r['n_iter']},{r['spearman']:.4f},{r['mass']:.3f}")
            else:
                print(f"{r['N']:>7} x {r['K']:<7}  {r['mode']:<12} "
                      f"{r['time']:>7.1f}s {r['n_iter']:>6} "
                      f"{r['spearman']:>9.4f} {r['mass']:>6.3f}")
        except Exception as e:
            print(f"{N:>7} x {K:<7}  {mode:<12} FAILED: {e}")

    if not args.csv:
        print()
        print("Spiral (2D) -> Swiss roll (3D), mixed_precision=True, epsilon=0.005")


if __name__ == "__main__":
    main()
