# SGW Improvement Roadmap

Updated: v0.3.0 (2026-04-06)

---

## Completed

### A1. Vectorized Sampling — DONE (v0.1.0)

Replaced Python `for` loop in `sample_pairs_from_plan` with Gumbel-max trick.
Eliminated the 38% CPU sampling bottleneck.

### A2. Distance Strategies — DONE (v0.2.0 + v0.2.1)

Originally proposed as "spectral embedding". After benchmarking, spectral embedding
(commute-time distance) performed poorly on manifold alignment tasks (Spearman = -0.21
on spiral-to-Swiss-roll). Replaced with three strategies:

- `distance_mode="precomputed"` — full pairwise, small scale (v0.2.0)
- `distance_mode="dijkstra"` — on-the-fly, medium scale (v0.1.0, default)
- `distance_mode="landmark"` — landmark Dijkstra via farthest-point sampling, large scale (v0.2.1)

Landmark Dijkstra achieves Spearman = 0.999 on the benchmark, matching exact Dijkstra quality.
See `examples/benchmark_distance_modes.md` for results.

### A3. Return GW Distance — DONE (v0.2.0)

`log_dict["gw_cost"]` is returned when `log=True`.

### A4. Fused Gromov-Wasserstein — DONE (v0.2.0)

`fgw_alpha` and `C_linear` parameters added. Blends structural GW cost with
feature-space linear cost:

```
Lambda = (1 - fgw_alpha) * Lambda_gw + fgw_alpha * C_linear
```

### A5. Multi-Scale / Coarse-to-Fine — DONE (v0.3.0)

`multiscale=True` parameter on `sampled_gw` and `sampled_lowrank_gw`.
Uses farthest-point sampling (FPS) to downsample, solves coarse GW,
then upsamples via nearest-representative assignment + Sinkhorn scaling
to warm-start the full problem.

**Known limitation**: GW has multiple equivalent local optima (e.g., forward
and reverse matching on symmetric manifolds). The coarse solve may converge
to a different local optimum than the fine solve would, locking the warm start
into that solution. On the spiral-to-Swiss-roll benchmark, this manifests as
a perfectly reversed matching (Spearman = -0.999). This is not a bug but an
inherent property of GW — multiscale works best on data without such symmetries
(e.g., single-cell data with distinct cell type compositions).

### A6. Low-Rank Sinkhorn — DONE (v0.3.0)

Implemented as a separate function `sampled_lowrank_gw` (not a parameter on
`sampled_gw`) because the behavior differs substantially:

- Uses the Scetbon, Cuturi & Peyre (2021) algorithm: mirror descent + Dykstra
  projection on factored transport plan T = Q @ diag(1/g) @ R^T
- **Memory optimization**, not speed optimization. At 400x500 scale, ~10x slower
  than standard Sinkhorn but uses O((N+K)*r) memory instead of O(NK)
- Matching quality is good (Spearman >= 0.995 on Swiss roll) despite higher
  transport cost (inherent to the low-rank constraint + different entropy objective)
- Default parameters: `rank=20, lr_max_iter=5, lr_dykstra_max_iter=50`
  (tuned for use as an inner step in the SGW loop)

Only use when N*K is too large for standard Sinkhorn memory (e.g., N, K > 50k).

### Tensor I/O — DONE (v0.2.0)

Input accepts both `torch.Tensor` and `np.ndarray`. Output is `torch.Tensor`.
`graph_source`/`graph_target` parameters removed (internal detail).

### B4. Custom Autograd — DONE (v0.1.0)

`_SinkhornAutograd` (envelope theorem: dL/dC = -T * grad / reg) already implemented.

### Internal Refactoring — DONE (v0.3.0)

Extracted shared SGW loop into `_sgw_loop`, `_prepare_inputs`, `_maybe_multiscale`
so that `sampled_gw` and `sampled_lowrank_gw` share preprocessing and iteration
logic, differing only in the Sinkhorn step.

---

## Remaining: Path B (Differentiable / GW as Loss)

Goal: `gw_loss` with full computation graph, backprop to input features or embeddings.
Use case: end-to-end learning where GW distance is a loss term.

### Current differentiability status

- `_SinkhornAutograd` works: gradient flows from GW cost back to cost matrix Lambda
- Dijkstra and landmark embedding are **not differentiable** — gradient does not
  flow back to input features X
- If Lambda is constructed from learnable parameters (e.g., a neural network output),
  the existing envelope theorem gradient is sufficient

### B1. Differentiable Distance Embedding

The original plan proposed spectral embedding for this purpose. Since spectral was
replaced with landmark Dijkstra (also non-differentiable), a different approach is
needed for end-to-end differentiability.

Options to evaluate:
1. **Learnable embedding** — train a small network to predict graph distances from
   features. The network parameters receive gradients through the GW loss.
2. **Soft kNN + differentiable shortest path** — replace hard kNN with a soft
   attention-weighted graph, then use differentiable shortest path algorithms.
   High complexity, unclear if practical.
3. **Direct feature distance** — when source and target share a feature space,
   use differentiable feature distances (L2, cosine) instead of graph distances.
   This is essentially Fused GW with fgw_alpha=1.0, already supported.

Recommendation: option 3 (already available) covers most practical use cases.
Option 1 is worth exploring if graph-distance gradients are truly needed.

### B2. Sampling Does NOT Need Gumbel-Softmax

Still valid. Anchor selection is discrete but does not block gradients.
Gradient flows through the distance computation on selected anchors, not through
the selection itself. Same principle as mini-batch SGD.

### B3. GW Loss with Entropy

Return both values:

```python
gw_cost = (Lambda * T).sum()                        # for reporting
gw_loss = gw_cost - reg * (T * (T.log() - 1)).sum() # for backprop
```

By envelope theorem, gradients of gw_cost and gw_loss w.r.t. upstream parameters
are identical. Difference is only in the loss value (not gradient):
- `gw_cost`: pure GW distance, comparable across runs
- `gw_loss`: what Sinkhorn actually minimizes

- **Status**: not yet implemented
- **Effort**: 30 minutes

---

## Execution Plan (updated)

### Done

- [x] A1 — Vectorized sampling (v0.1.0)
- [x] A3 — Return GW distance (v0.2.0)
- [x] A2 — Distance strategies: precomputed + landmark (v0.2.0, v0.2.1)
- [x] A4 — Fused GW (v0.2.0)
- [x] B4 — Custom autograd (v0.1.0)
- [x] Tensor I/O (v0.2.0)
- [x] A5 — Multi-scale warm start (v0.3.0)
- [x] A6 — Low-rank Sinkhorn as `sampled_lowrank_gw` (v0.3.0)

### Next

- [ ] B3 — Return `gw_loss` with retained computation graph (quick win, 30 min)
- [ ] B1 — Evaluate differentiable distance options
