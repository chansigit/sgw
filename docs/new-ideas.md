# New Research Ideas: Variance Reduction for Sampled GW

## Background

Standard variance reduction techniques (e.g., control variates with gradient tables)
cannot be directly applied to SGW because:
1. Sampling distribution (T) is non-stationary
2. A gradient table would require O(NK) entries with O(N² log N) Dijkstra cost
3. Algorithm is stochastic proximal (Sinkhorn), not stochastic gradient

However, three modified approaches CAN achieve variance reduction.

---

## Idea 1: Lambda EMA (Exponential Moving Average of Cost Matrix)

**Complexity**: ~5 lines of code. Lowest risk.

Maintain a running average of the cost matrix:

```
Lambda_ema = (1 - beta) * Lambda_ema + beta * Lambda_sample
```

Use `Lambda_ema` instead of `Lambda_sample` for Sinkhorn.

- **Biased**, but bias is O(α² · err²) — vanishes at convergence
- **Variance** reduced from O(1/M) to O(β/M)
- Near convergence: MSE(Lambda_ema) = Var + Bias² = O(β/M) + O(α² err²)
- Related to Polyak-Ruppert averaging, but applied to cost matrix of a proximal step

**Paper angle**: Biased-but-low-variance cost matrix estimation for stochastic proximal GW, with convergence analysis showing bias vanishes.

---

## Idea 2: Spectral Decomposition of GW Cost Matrix

**Complexity**: Medium. Most publishable.

The GW cost matrix decomposes as:

```
Λ_{ij} = [Σ_a p_a · d_s(i,a)²] + [Σ_b q_b · d_t(j,b)²] - 2·(D_s T D_t^T)_{ij}
          ├── diagonal terms ──┤                              ├── cross term ──┤
```

**Key insight**: The diagonal terms depend ONLY on the fixed marginals p, q (not on T's internal structure). They can be computed exactly using landmark Dijkstra + Nyström approximation. Only the cross term needs sampling.

- Diagonal terms: 0 variance (exact computation)
- Cross term variance: O(σ²/r) where r = effective rank of T
- Overall variance reduction factor: ~r/(N·K), substantial for entropic OT

**Why novel**: The observation that GW cost matrix variance is dominated by diagonal terms that happen to be exactly computable from fixed marginals has not appeared in the literature.

**Implementation**:
1. Select L landmarks via farthest-point sampling
2. Pre-compute Dijkstra from all landmarks: O(L · N log N), one-time
3. Nyström approximate full distance matrix from landmarks
4. Compute diagonal terms exactly each iteration
5. Only sample anchors for cross term

---

## Idea 3: Landmark-Cached Variance Reduction

**Complexity**: High. The most principled variance reduction approach for SGW.

Select L landmark nodes, pre-compute ALL Dijkstra distances from landmarks (one-time O(L·N log N)). Then:

- Sample anchor pairs only from the L×L landmark set
- Since distances are cached, maintain a gradient correction table of size L²
- Table update cost: O(M) per iteration
- Full average maintained incrementally

**Variance guarantee**: Under slow-T-change assumption (ensured by momentum), the correction gives O(1/t) variance decay per pass through landmark set, vs O(1/M) for plain Monte Carlo.

**Tradeoff**: Accuracy limited by landmark density. L must be large enough to capture graph geometry.

---

## References

- Kerdoncuff et al. (2021) — Sampled Gromov Wasserstein (original SGW framework)
- Scetbon & Peyré (2022) — Linear-Time GW via Low-Rank Couplings
- Fatras et al. (2021) — Minibatch OT Distances
- Asi & Duchi (2019) — Variance Reduction for Stochastic Proximal Point
- Defazio et al. (2014) — Incremental Gradient Methods with Memory
