# TorchGW v0.2.1 — Algorithm Description

## Overview

Sampled Gromov-Wasserstein (TorchGW) computes an optimal transport plan between two point clouds
that may live in different dimensional spaces. Instead of computing the full N x K pairwise
cost matrix each iteration, TorchGW **samples** M anchor pairs and approximates the GW cost
using only the distances from those anchors, reducing the per-iteration cost from
O(NK(N+K)) to O(NKM).

v0.2.0 introduced three distance strategies, Fused GW, and torch tensor I/O.
v0.2.1 replaced the spectral embedding strategy with landmark Dijkstra.

---

## 1. Input

| Symbol | Shape | Description |
|--------|-------|-------------|
| X | (N, D) | Source point cloud (Tensor or ndarray, optional if dist provided) |
| Y | (K, D') | Target point cloud (dimension may differ) |
| **p** | (N,) | Source marginal (default: uniform) |
| **q** | (K,) | Target marginal (default: uniform) |
| epsilon | scalar | Entropic regularization |
| M | int | Number of anchor pairs per iteration |
| alpha | scalar | Momentum for transport plan update |
| distance_mode | str | `"precomputed"`, `"dijkstra"` (default), or `"landmark"` |
| fgw_alpha | float | Fused GW blending: 0 = pure GW, 1 = pure Wasserstein |
| C_linear | (N, K) | Feature cost matrix for Fused GW (required if fgw_alpha > 0) |

---

## 2. Preprocessing: kNN Graph Construction

For each point cloud, build a k-nearest-neighbor graph (default k = 30):

1. Compute k-NN distances via `sklearn.neighbors.kneighbors_graph`.
2. Check connectivity with `scipy.sparse.csgraph.connected_components`.
3. If disconnected, iteratively **stitch** components:
   - Find the closest pair of points between component 0 and all other components (via KDTree).
   - Add a bidirectional edge with the Euclidean distance as weight.
   - Repeat until the graph is connected.

Output: sparse CSR distance matrices G_X and G_Y, guaranteed connected.

Not needed when `distance_mode="precomputed"` with user-provided distance matrices,
or when `fgw_alpha=1.0` (pure Wasserstein).

---

## 3. Distance Computation Strategies

TorchGW's main loop (Section 5) requires distance vectors D_X(:, m) and D_Y(:, m) for each
sampled anchor. There are three strategies, suited to different problem scales.

### 3.1 Precomputed Full Pairwise Distances — small scale

Precompute all-pairs shortest paths once before the main loop:

```
C_X = all_pairs_dijkstra(G_X)    # shape (N, N)
C_Y = all_pairs_dijkstra(G_Y)    # shape (K, K)
```

In the main loop, simply index into the precomputed matrices:

```
D_X = C_X[:, sampled_source_indices]    # shape (N, M), table lookup
D_Y = C_Y[:, sampled_target_indices]    # shape (K, M), table lookup
```

Users can also pass their own distance matrices directly (e.g., Euclidean, diffusion
distances), bypassing graph construction entirely.

- **Precomputation**: O(N^2 log N + K^2 log K), one-time
- **Per-iteration**: O(NM + KM), just indexing
- **Memory**: O(N^2 + K^2)
- **Suitable for**: N, K < ~5k

### 3.2 On-the-fly Dijkstra — medium scale (default)

No precomputation. Each iteration runs Dijkstra from the M sampled anchors:

```
D_X = batch_dijkstra(G_X, sampled_source_indices)    # shape (N, M)
D_Y = batch_dijkstra(G_Y, sampled_target_indices)    # shape (K, M)
```

Post-processing:
- Infinite distances (disconnected nodes) are clamped to 1.5 * max(finite values).
- Each distance matrix is normalized by its maximum.
- Dijkstra runs in parallel (via joblib) when there are >= 64 unique sources.

- **Precomputation**: none
- **Per-iteration**: O(M * N log N + M * K log K), CPU-bound
- **Memory**: O(NM + KM)
- **Suitable for**: N, K ~ 5k to 50k

### 3.3 Landmark Dijkstra — large scale

Select d well-spread landmark nodes via **farthest-point sampling** (FPS), then
precompute shortest-path distances from every node to each landmark. This gives
each node a d-dimensional coordinate vector. At query time, Euclidean distance
in this landmark-distance space approximates geodesic distance on the graph.

**Farthest-point sampling**:
1. Start from node 0.
2. Run Dijkstra to all other nodes.
3. Pick the node farthest from all existing landmarks.
4. Repeat until d landmarks are selected.

This produces well-spread landmarks and reuses the Dijkstra results, so the
total precomputation is exactly d Dijkstra runs.

```
L_X = landmark_embed(G_X, d)    # shape (N, d), distances to d landmarks
L_Y = landmark_embed(G_Y, d)    # shape (K, d)
```

In the main loop, compute Euclidean distances on GPU:

```
D_X[i, m] = || L_X[i] - L_X[j_m] ||    # GPU torch.cdist
D_Y[k, m] = || L_Y[k] - L_Y[l_m] ||    # GPU torch.cdist
```

- **Precomputation**: O(d * N log N + d * K log K), one-time
- **Per-iteration**: O((NM + KM) * d), fully GPU-resident
- **Memory**: O(Nd + Kd)
- **Suitable for**: N, K > 50k
- **Accuracy**: uses real shortest-path distances (not an approximation like spectral
  embedding), so quality is close to exact Dijkstra. On the spiral-to-Swiss-roll
  benchmark with d=20, achieves rho = 0.999.

### Summary

| Scale | Strategy | Precompute | Per-iteration | Memory | Status |
|-------|----------|------------|---------------|--------|--------|
| Small (< 5k) | Full pairwise | O(N^2 log N) | O(NM) lookup | O(N^2) | v0.2.0 |
| Medium (5k-50k) | Dijkstra on-the-fly | None | O(MN log N) | O(NM) | v0.1.0 (default) |
| Large (> 50k) | Landmark Dijkstra | O(dN log N) | O(NMd) GPU | O(Nd) | v0.2.1 |

---

## 4. Initialization

1. **Marginals** — If not provided, set p = 1_N / N and q = 1_K / K.

2. **Slack variables** — For partial transport:

   ```
   m_frac  = s_eff / max(N, K)
   slack_p = max(||q||_1 - m_frac, 1e-10)
   slack_q = max(||p||_1 - m_frac, 1e-10)
   ```

3. **Augmented marginals** — Append slack mass to each marginal:

   ```
   p_aug = [p; slack_p]    in R^{N+1}
   q_aug = [q; slack_q]    in R^{K+1}
   ```

4. **Initial coupling** — T = p (x) q (outer product).

5. **Regularization schedule** — Exponential decay from epsilon_0 to epsilon_f = min(5e-4, epsilon_0):

   ```
   epsilon_t = epsilon_0 * gamma^t
   gamma     = (epsilon_f / epsilon_0) ^ (1 / T_max)
   ```

---

## 5. Main Loop (per iteration t)

### 5.1 Sample Anchor Pairs

Sample M pairs (j_m, l_m) from the current plan T using the **Gumbel-max trick**:

1. Sample source indices j_1, ..., j_M from row marginals p_row = T * 1.
2. For each sampled row j_m, sample column l_m via:

   ```
   l_m = argmax_k [ log T[j_m, k] + g_k ],    g_k ~ Gumbel(0, 1)
   ```

This vectorizes the categorical sampling over all M pairs in a single numpy operation.

### 5.2 Obtain Anchor Distances

Obtain D_X (N x M) and D_Y (K x M) using the distance strategy described in Section 3.

### 5.3 Assemble the GW Cost Matrix

Approximate the Gromov-Wasserstein cost using the M sampled anchor distances:

```
Lambda_gw[i, k] = mean_m( D_X[i,m]^2 ) - (2/M) * sum_m( D_X[i,m] * D_Y[k,m] ) + mean_m( D_Y[k,m]^2 )
```

In matrix form:

```
Lambda_gw = mean(D_X^2, axis=1) * 1^T  -  (2/M) * D_X @ D_Y^T  +  1 * mean(D_Y^2, axis=1)^T
```

### 5.4 Fused GW Blending (optional)

If fgw_alpha > 0, blend the structural GW cost with the feature cost:

```
Lambda = (1 - fgw_alpha) * Lambda_gw + fgw_alpha * C_linear
```

If fgw_alpha = 0 (default), Lambda = Lambda_gw. If fgw_alpha = 1, Lambda = C_linear
(pure Wasserstein, no structural distances needed).

**Augmented cost**: Pad Lambda to (N+1) x (K+1) with a penalty of 100 * max(Lambda)
in the slack row and column.

### 5.5 Sinkhorn Step

Solve the augmented entropic OT problem using the log-domain Sinkhorn algorithm
(see Section 6 for details). Extract the real block from the result:

```
T_new = T_aug[0:N, 0:K]
```

### 5.6 Momentum Update

```
T <- (1 - alpha) * T_prev + alpha * T_new
```

### 5.7 Convergence Check

```
err_t = || T - T_prev ||_F
```

Stop if err_t < tol and t >= t_min (default t_min = 50).

---

## 6. Log-Domain Sinkhorn Algorithm

Given marginals **a** in R^{N+1}, **b** in R^{K+1} and cost matrix C in R^{(N+1) x (K+1)},
the Sinkhorn algorithm finds the transport plan that minimizes:

```
min_{T >= 0}  <C, T> + epsilon * KL(T || a (x) b)
subject to    T * 1 = a,  T^T * 1 = b
```

### 6.1 Log-Domain Formulation

All computations are performed in log-space to avoid numerical overflow/underflow.

- **Log-kernel**: log K[i,j] = -C[i,j] / epsilon
- **Dual variables**: log **u** in R^{N+1}, log **v** in R^{K+1}, initialized to zero.

### 6.2 Iteration

Alternate between updating the two dual variables (until marginal error < tol or
max 100 inner iterations):

```
log u[i]     <-  log a[i] - logsumexp_j( log K[i,j] + log v[j] )
log v[j]_raw <-  log b[j] - logsumexp_i( log K[i,j] + log u[i] )
log v[j]     <-  tau * log v[j]_raw
```

The relaxation factor tau controls the target marginal constraint:

| Mode | tau | Effect |
|------|-----|--------|
| Balanced GW | 1 | Both marginals are hard constraints |
| Semi-relaxed GW | rho / (rho + epsilon) | Target marginal softened via KL penalty weighted by rho |

As rho -> infinity, tau -> 1 (recovers balanced). As rho -> 0, tau -> 0 (target unconstrained).

### 6.3 Recover Transport Plan

```
T[i,j] = exp( log u[i] + log K[i,j] + log v[j] )
```

Equivalently: T[i,j] = u[i] * exp(-C[i,j] / epsilon) * v[j].

### 6.4 Convergence Check

Every `check_every` iterations (default 10), compute the source marginal and check:

```
max_i | sum_j T[i,j] - a[i] | < tol
```

---

## 7. Output

- **Transport plan** T (torch.Tensor, shape N x K): T[i,k] is the coupling weight between X[i] and Y[k].
- **GW cost** (optional): GW = sum_{i,k} Lambda[i,k] * T[i,k].
- **Convergence log** (optional): error history, iteration count.

---

## 8. Differentiable Mode

When `differentiable=True`, a custom `torch.autograd.Function` is used:

- **Forward**: Runs Sinkhorn **without** recording the computation graph; saves only T.
- **Backward**: Applies the **envelope theorem**:

  ```
  dL/dLambda = -(1/epsilon) * T (.) dL/dT
  ```

This is memory-efficient: no need to backprop through hundreds of Sinkhorn iterations.

Note: gradients flow from the GW cost back to the cost matrix Lambda, but do **not**
flow back to the input features X, because Dijkstra and landmark embedding are
non-differentiable. For end-to-end differentiability, the Sinkhorn envelope theorem
gradient w.r.t. Lambda is sufficient when Lambda is constructed from learnable parameters.

---

## 9. Joint Embedding

Given an anchor dataset and one or more query datasets with their transport plans,
`joint_embedding` computes a shared low-dimensional representation:

1. **Graph Laplacians** — For each dataset, compute L = D - W where W is the symmetrized
   inverse-distance kNN weight matrix and D is the degree matrix.

2. **Linear system** — For each query q:

   ```
   S_xx(q) = L_q + lambda * diag(T(q) * 1)
   ```

   For the anchor:

   ```
   S_yy = L_a + lambda * sum_q diag(T(q)^T * 1)
   ```

3. **Implicit operator** — Define H: R^{N_a} -> R^{N_q} as:

   ```
   H * v = S_xx(q)^{-1} * T(q) * S_yy^{-1} * v
   ```

   where the inverses are applied via conjugate gradient (CG).

4. **Truncated SVD** — Compute the top-d singular vectors of H:

   ```
   H ~ U * Sigma * V^T
   ```

   - V in R^{N_a x d}: anchor embedding
   - U in R^{N_q x d}: query embeddings

---

## 10. Complexity

### Precomputation (one-time)

| Strategy | Distance precompute | Memory |
|----------|:-------------------:|:------:|
| Precomputed: full pairwise | O(N^2 log N + K^2 log K) | O(N^2 + K^2) |
| Dijkstra: on-the-fly | None | None |
| Landmark: d landmarks | O(d * N log N + d * K log K) | O(Nd + Kd) |

### Per-iteration cost

| Component | Standard GW | Precomputed (lookup) | Dijkstra | Landmark |
|-----------|:-----------:|:--------------------:|:--------:|:--------:|
| Distance | O(N^2 + K^2) precomputed | O(NM + KM) | O(M(N+K) log(N+K)) | O((NM + KM) * d) |
| Cost matrix | O(NK(N+K)) | O(NKM) | O(NKM) | O(NKM) |
| Sinkhorn | O(NK) | O(NK) | O(NK) | O(NK) |
| **Iteration total** | **O(NK(N+K))** | **O(NKM)** | **O(NKM + M(N+K) log(N+K))** | **O(NKM + (NM+KM)d)** |

### Overall (T iterations)

| Strategy | Total time | Bottleneck |
|----------|:----------:|:----------:|
| Standard GW | O(T * NK(N+K)) | Full cost tensor |
| Precomputed | O(N^2 log N) + O(T * NKM) | Precomputation |
| Dijkstra | O(T * (NKM + MN log N)) | Dijkstra (CPU) |
| Landmark | O(dN log N) + O(T * NKMd) | Precomputation |

With M << min(N, K), all three TorchGW strategies are substantially faster than standard GW.

---

## 11. Numerical Details

- **float64** for marginals and Sinkhorn (numerical stability).
- **float32** for distances and cost matrix (GPU throughput).
- Garbage collection runs every 50 iterations to avoid overhead.
- Dijkstra switches from serial to parallel at >= 64 unique source nodes.
- Input coercion: numpy arrays are converted to tensors via `torch.as_tensor` (zero-copy).
- Output: `torch.Tensor` (use `.numpy()` or `.cpu().numpy()` to convert back).
