"""Benchmark TorchGW vs CNT-GW on MUSE cross-lingual word embedding alignment.

Task: Align 2000 English words to 2000 French words using GW on cosine distance.
Evaluation: Precision@k — what fraction of ground-truth translation pairs are
            recovered by the transport plan's top-k matching.

Usage:
    PYTHONPATH=. python examples/benchmark_muse.py
"""
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, "/scratch/users/chensj16/projects/egw-solvers")

from torchgw import sampled_gw


# ── Data loading ─────────────────────────────────────────────────────────

def load_muse(data_dir="muse_gw_data"):
    en_emb = np.load(f"{data_dir}/en_emb.npy")
    fr_emb = np.load(f"{data_dir}/fr_emb.npy")
    C_en = np.load(f"{data_dir}/C_en.npy")
    C_fr = np.load(f"{data_dir}/C_fr.npy")

    with open(f"{data_dir}/en_words.txt") as f:
        en_words = [l.strip() for l in f]
    with open(f"{data_dir}/fr_words.txt") as f:
        fr_words = [l.strip() for l in f]
    with open(f"{data_dir}/ground_truth_dict.txt") as f:
        gt_pairs = [l.strip().split() for l in f if l.strip()]

    # Build ground-truth index mapping
    en_idx = {w: i for i, w in enumerate(en_words)}
    fr_idx = {w: i for i, w in enumerate(fr_words)}

    gt_indices = []
    for en_w, fr_w in gt_pairs:
        if en_w in en_idx and fr_w in fr_idx:
            gt_indices.append((en_idx[en_w], fr_idx[fr_w]))

    return en_emb, fr_emb, C_en, C_fr, en_words, fr_words, gt_indices


# ── Evaluation ───────────────────────────────────────────────────────────

def precision_at_k(T, gt_indices, ks=(1, 5, 10)):
    """Compute Precision@k from transport plan T and ground-truth pairs."""
    T_np = T.numpy() if isinstance(T, torch.Tensor) else T
    results = {}
    for k in ks:
        # For each source word, get top-k target indices by transport mass
        top_k = np.argsort(-T_np, axis=1)[:, :k]
        hits = 0
        total = len(gt_indices)
        for src_i, tgt_i in gt_indices:
            if tgt_i in top_k[src_i]:
                hits += 1
        results[f"P@{k}"] = hits / total
    return results


# ── Solvers ──────────────────────────────────────────────────────────────

def run_torchgw_precomputed(C_en, C_fr, **kwargs):
    """TorchGW with precomputed cosine distance matrices."""
    np.random.seed(42)
    T = sampled_gw(
        distance_mode="precomputed",
        dist_source=C_en, dist_target=C_fr,
        s_shared=2000, M=kwargs.get("M", 80),
        alpha=kwargs.get("alpha", 0.9),
        max_iter=kwargs.get("max_iter", 500),
        epsilon=kwargs.get("epsilon", 0.05),
        verbose=False,
        lambda_ema_beta=kwargs.get("beta", None),
    )
    return T.cpu()


def run_cntgw_euclidean(en_emb, fr_emb, **kwargs):
    """CNT-GW QuadraticGW on raw embeddings (squared Euclidean cost)."""
    from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW

    X = torch.from_numpy(en_emb).float()
    Y = torch.from_numpy(fr_emb).float()

    solver = QuadraticGW(
        X=X, Y=Y,
        eps=kwargs.get("eps", 0.1),
        numItermax=kwargs.get("n_outer", 50),
        SINK_ARGS={"numItermax": kwargs.get("n_sinkhorn", 200)},
    )
    solver.solve()
    T = solver.transport_plan(lazy=False)
    return T


def run_cntgw_cosine_kpca(C_en, C_fr, **kwargs):
    """CNT-GW with Kernel PCA on precomputed cosine distance matrices."""
    from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
    from utils.implementation.kernels import kernel_from_costmatrix

    approx_dims = kwargs.get("approx_dims", 20)
    N, K = C_en.shape[0], C_fr.shape[0]
    a = torch.ones(N) / N
    b = torch.ones(K) / K

    C1_t = torch.from_numpy(C_en).float()
    C2_t = torch.from_numpy(C_fr).float()

    K1 = kernel_from_costmatrix(C1_t, a=a, center=True)
    K2 = kernel_from_costmatrix(C2_t, a=b, center=True)

    eigvals1, eigvecs1 = torch.linalg.eigh(K1)
    eigvals2, eigvecs2 = torch.linalg.eigh(K2)

    d = approx_dims
    X_emb = eigvecs1[:, -d:] * eigvals1[-d:].clamp(min=0).sqrt()
    Y_emb = eigvecs2[:, -d:] * eigvals2[-d:].clamp(min=0).sqrt()

    solver = QuadraticGW(
        X=X_emb, Y=Y_emb,
        eps=kwargs.get("eps", 0.1),
        numItermax=kwargs.get("n_outer", 50),
        SINK_ARGS={"numItermax": kwargs.get("n_sinkhorn", 200)},
    )
    solver.solve()
    T = solver.transport_plan(lazy=False)
    return T


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MUSE en-fr data...")
    en_emb, fr_emb, C_en, C_fr, en_words, fr_words, gt_indices = load_muse()
    print(f"  {len(en_words)} EN words, {len(fr_words)} FR words, {len(gt_indices)} GT pairs")

    configs = [
        ("TorchGW (precomputed)",
         lambda: run_torchgw_precomputed(C_en, C_fr, M=80, max_iter=500, epsilon=0.05)),
        ("TorchGW (precomputed+EMA)",
         lambda: run_torchgw_precomputed(C_en, C_fr, M=80, max_iter=500, epsilon=0.05, beta=0.5)),
        ("CNT-GW (Euclidean)",
         lambda: run_cntgw_euclidean(en_emb, fr_emb, eps=0.1, n_outer=50, n_sinkhorn=200)),
        ("CNT-GW (cosine KPCA d=20)",
         lambda: run_cntgw_cosine_kpca(C_en, C_fr, eps=0.1, n_outer=50, n_sinkhorn=200, approx_dims=20)),
        ("CNT-GW (cosine KPCA d=50)",
         lambda: run_cntgw_cosine_kpca(C_en, C_fr, eps=0.1, n_outer=50, n_sinkhorn=200, approx_dims=50)),
    ]

    results = {}
    for name, fn in configs:
        print(f"\n  Running {name}...")
        t0 = time.time()
        try:
            T = fn()
            elapsed = time.time() - t0
            prec = precision_at_k(T, gt_indices, ks=(1, 5, 10))
            results[name] = {"time": elapsed, **prec, "T": T}
            print(f"    {elapsed:.2f}s | P@1={prec['P@1']:.4f} P@5={prec['P@5']:.4f} P@10={prec['P@10']:.4f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED ({elapsed:.1f}s): {e}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'P@1':>6} {'P@5':>6} {'P@10':>6} {'Time':>8}")
    print(f"{'-'*80}")
    for name, r in results.items():
        print(f"{name:<30} {r['P@1']:>6.4f} {r['P@5']:>6.4f} {r['P@10']:>6.4f} {r['time']:>7.2f}s")
    print(f"{'='*80}")

    # Plot
    if results:
        methods = list(results.keys())
        p1 = [results[m]["P@1"] for m in methods]
        p5 = [results[m]["P@5"] for m in methods]
        p10 = [results[m]["P@10"] for m in methods]
        times = [results[m]["time"] for m in methods]
        colors = ['#3498db' if 'TorchGW' in m else '#e74c3c' for m in methods]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # P@k grouped bar chart
        ax = axes[0]
        x = np.arange(len(methods))
        w = 0.25
        ax.bar(x - w, p1, w, label='P@1', color='#2c3e50')
        ax.bar(x, p5, w, label='P@5', color='#2980b9')
        ax.bar(x + w, p10, w, label='P@10', color='#7fb3d8')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=7)
        ax.set_ylabel('Precision')
        ax.set_title('Translation Precision@k', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Runtime
        ax = axes[1]
        ax.bar(range(len(methods)), times, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=7)
        ax.set_ylabel('Time (s)')
        ax.set_title('Runtime', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(times):
            ax.text(i, v + 0.3, f'{v:.1f}s', ha='center', fontsize=8)

        # Show some example translations from best method
        ax = axes[2]
        best = max(results, key=lambda m: results[m]['P@1'])
        T_best = results[best]['T']
        T_np = T_best.numpy() if isinstance(T_best, torch.Tensor) else T_best
        ax.axis('off')
        ax.set_title(f'Sample translations ({best})', fontweight='bold')

        lines = []
        shown = 0
        for src_i, tgt_i in gt_indices[:50]:
            pred_i = T_np[src_i].argmax()
            correct = pred_i == tgt_i
            mark = "OK" if correct else "X "
            lines.append(f"  {mark}  {en_words[src_i]:<12} -> {fr_words[pred_i]:<12} (GT: {fr_words[tgt_i]})")
            shown += 1
            if shown >= 20:
                break
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace')

        fig.suptitle('MUSE EN-FR Word Alignment (2000 x 2000, cosine distance)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig('examples/benchmark_muse.png', dpi=130)
        print(f"\nPlot saved -> examples/benchmark_muse.png")
