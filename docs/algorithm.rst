Algorithm
=========

Gromov-Wasserstein Optimal Transport
------------------------------------

Given two metric spaces :math:`(\mathcal{X}, C_1)` and :math:`(\mathcal{Y}, C_2)`
with distributions :math:`p` and :math:`q`, the Gromov-Wasserstein distance finds
a transport plan :math:`T` that minimizes:

.. math::

   \text{GW}(T) = \sum_{i,j,k,l} \left| C_1(x_i, x_k) - C_2(y_j, y_l) \right|^2 T_{ij} T_{kl}

subject to :math:`T \mathbf{1} = p` and :math:`T^\top \mathbf{1} = q`.

Unlike Wasserstein distance, GW does not require the two spaces to share
a common metric — it compares **intra-domain distances**, making it suitable
for cross-domain alignment (e.g., different modalities, different dimensionalities).

Sampled GW (TorchGW)
-----------------

Standard entropic GW computes a cost matrix of size :math:`N \times K` at
each iteration using all :math:`N \times N` and :math:`K \times K` pairwise
distances. TorchGW reduces this by **sampling** :math:`M` anchor pairs per iteration.

Each iteration:

1. **Sample** :math:`M` anchor pairs :math:`(i, j)` from the current transport
   plan :math:`T`, weighted by coupling mass.

2. **Dijkstra** shortest paths from the :math:`\leq M` unique sampled source
   nodes on both kNN graphs.

3. **Cost matrix** assembly on GPU:

   .. math::

      \Lambda = \text{mean}(D_{\text{left}}^2) - \frac{2}{M} D_{\text{left}} D_{\text{tgt}}^\top + \text{mean}(D_{\text{tgt}}^2)

4. **Augmented Sinkhorn** with slack variables for partial transport. The cost
   matrix is augmented to :math:`(N+1) \times (K+1)` with penalty rows/columns,
   allowing the solver to assign mass to "slack" when alignment is poor.

5. **Momentum update**:

   .. math::

      T \leftarrow (1 - \alpha) T_{\text{prev}} + \alpha T_{\text{new}}

Complexity
^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Component
     - Standard GW
     - TorchGW
   * - Cost matrix per iter
     - :math:`O(NK(N+K))`
     - :math:`O(NKM)`
   * - Dijkstra per iter
     - :math:`O((N+K)(N+K) \log(N+K))`
     - :math:`O(M(N+K) \log(N+K))`
   * - Sinkhorn per iter
     - :math:`O(NK)`
     - :math:`O(NK)` (same)

With :math:`M \ll \min(N, K)`, TorchGW achieves sub-quadratic scaling in the
number of Dijkstra computations while maintaining the same Sinkhorn cost.

Log-Domain Sinkhorn
-------------------

TorchGW uses a pure-PyTorch log-domain Sinkhorn implementation for numerical
stability with small regularization :math:`\varepsilon`:

.. math::

   \log u &\leftarrow \log a - \text{logsumexp}(\log K + \log v, \text{dim}=1) \\
   \log v &\leftarrow \tau \left[ \log b - \text{logsumexp}(\log K + \log u, \text{dim}=0) \right]

where :math:`\log K = -C / \varepsilon` and :math:`\tau = 1` for balanced GW.

Semi-Relaxed GW
^^^^^^^^^^^^^^^^

Setting :math:`\tau = \rho / (\rho + \varepsilon)` with :math:`\tau < 1` relaxes
the target marginal constraint via a KL divergence penalty:

- :math:`\rho \to \infty`: :math:`\tau \to 1`, recovers balanced GW
- :math:`\rho \to 0`: :math:`\tau \to 0`, target marginal is completely free

This is useful when source and target have different compositions.

Differentiable Sinkhorn
^^^^^^^^^^^^^^^^^^^^^^^

When ``differentiable=True``, TorchGW uses a custom ``torch.autograd.Function``
that:

1. Runs the Sinkhorn loop in forward **without** recording the computation graph
2. Saves only the resulting transport plan :math:`T`
3. Computes gradients via the **envelope theorem**:

   .. math::

      \frac{\partial \mathcal{L}}{\partial C} = -\frac{T}{\varepsilon} \cdot \frac{\partial \mathcal{L}}{\partial T}

This avoids backpropagating through all Sinkhorn iterations, making memory
cost :math:`O(NK)` regardless of the number of Sinkhorn steps.

Regularization Schedule
-----------------------

The entropic regularization :math:`\varepsilon` is decayed exponentially
during optimization:

.. math::

   \varepsilon_t = \varepsilon_0 \cdot \gamma^t, \quad \gamma = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{1/T}

where :math:`\varepsilon_{\min} = \min(5 \times 10^{-4}, \varepsilon_0)`.
Large initial regularization helps exploration; small final regularization
sharpens the transport plan.
