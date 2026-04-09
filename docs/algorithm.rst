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
-----------------------

When ``differentiable=True``, TorchGW computes exact gradients through the
Sinkhorn solver, enabling end-to-end learning where the transport plan is a
differentiable function of the cost matrix.


The problem
^^^^^^^^^^^

The entropic OT solution is :math:`T^*_{ij} = \exp\bigl((f_i + g_j - C_{ij})/\varepsilon\bigr)`,
where :math:`f, g` are the Sinkhorn dual potentials (Kantorovich potentials).
To backpropagate through :math:`T^*(C)`, we need :math:`\partial T^* / \partial C`.

A naive approach — freeze :math:`f, g` and differentiate the exponential directly —
gives :math:`\partial T^*_{ij}/\partial C_{ij} \approx -T^*_{ij}/\varepsilon`.
This **frozen-potentials approximation** ignores that :math:`f, g` themselves
depend on :math:`C` through the Sinkhorn iterations.  In practice this produces
gradients with cosine similarity as low as 0.07 against the true gradient,
especially at small :math:`\varepsilon`.


Implicit differentiation (default: ``grad_mode="implicit"``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of differentiating through the iterative Sinkhorn algorithm, we use the
**implicit function theorem** (IFT) at the converged fixed point.

**Step 1: Fixed-point conditions.**
At convergence, the Sinkhorn potentials satisfy:

.. math::

   F_1: \quad \log u_i + \log\!\Bigl(\sum_j K_{ij} v_j\Bigr) = \log a_i \\
   F_2: \quad \log v_j + \log\!\Bigl(\sum_i K_{ij} u_i\Bigr) = \log b_j

where :math:`K_{ij} = e^{-C_{ij}/\varepsilon}`.

**Step 2: Jacobian of the fixed-point map.**
The Jacobian :math:`J = \partial F / \partial (\log u, \log v)` has a clean block structure:

.. math::

   J = \begin{pmatrix} I_N & P \\ R & I_K \end{pmatrix}

where :math:`P_{ij} = T^*_{ij}/a_i` and :math:`R_{ji} = T^*_{ij}/b_j` are the
**row-normalized** and **column-normalized** transport plans (softmax outputs from
each Sinkhorn half-step).  Both :math:`P` and :math:`R` are row-stochastic, so
the eigenvalues of :math:`J` lie in :math:`[0, 2]` — a well-conditioned system.

**Step 3: Adjoint equation.**
By the IFT, the vector-Jacobian product (VJP) for an upstream loss
:math:`\mathcal{L}(T^*)` requires solving the adjoint system:

.. math::

   J^\top \begin{pmatrix} \lambda_u \\ \lambda_v \end{pmatrix}
   = \begin{pmatrix} r_u \\ r_v \end{pmatrix}

where :math:`r_u = (G \odot T^*)\,\mathbf{1}`,
:math:`r_v = (G \odot T^*)^\top \mathbf{1}`, and
:math:`G = \partial\mathcal{L}/\partial T^*` is the upstream gradient.

**Step 4: Schur complement solve.**
Eliminating :math:`\lambda_u` gives a :math:`K \times K` system:

.. math::

   \underbrace{(I_K - P^\top R^\top)}_{S}\;\lambda_v = r_v - P^\top r_u

:math:`S` has a rank-1 null space (eigenvector :math:`\mathbf{1}_K`,
from the constant ambiguity in Sinkhorn potentials: :math:`f + c, g - c`
yield the same :math:`T^*`).  This null mode cancels in the final gradient,
so we remove it by adding :math:`\mathbf{1}\mathbf{1}^\top\!/K` to :math:`S`,
replacing the zero eigenvalue with 1.  The solve is then a standard
``torch.linalg.solve`` call.

**Step 5: Final VJP.**

.. math::

   \frac{\partial \mathcal{L}}{\partial C_{kl}}
   = \frac{T^*_{kl}}{\varepsilon}
     \Bigl(-G_{kl} + \frac{\lambda_{u,k}}{a_k} + \frac{\lambda_{v,l}}{b_l}\Bigr)


**Complexity:** :math:`O(NK^2 + K^3)` for the Schur complement construction and
solve.  Memory: :math:`O(K^2)` for the Schur complement matrix plus :math:`O(NK)`
for :math:`T^*`.  No Sinkhorn iterations are stored.


Unrolled autograd (``grad_mode="unrolled"``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative is to simply run the Sinkhorn loop under ``torch.enable_grad()``,
letting PyTorch's autograd record and differentiate through every iteration.
This gives **exact** gradients (matching the implicit mode up to floating-point
precision) but stores all intermediate states:

- **Memory:** :math:`O(NK \times \text{sinkhorn\_iters})`
- **Speed:** ~1.5–2x slower than implicit (extra graph bookkeeping)
- **When to use:** debugging, or when :math:`\varepsilon` is extremely small
  (< 0.001) and the transport plan has severe floating-point underflow that limits
  the implicit mode's accuracy.


Summary
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - ``grad_mode``
     - Gradient
     - Memory
     - Notes
   * - ``"implicit"`` (default)
     - Exact (IFT)
     - :math:`O(NK + K^2)`
     - Best default; Schur solve
   * - ``"unrolled"``
     - Exact (autograd)
     - :math:`O(NK \times \text{iters})`
     - Fallback for extreme :math:`\varepsilon`


Regularization Schedule
-----------------------

The entropic regularization :math:`\varepsilon` is decayed exponentially
during optimization:

.. math::

   \varepsilon_t = \varepsilon_0 \cdot \gamma^t, \quad \gamma = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{1/T}

where :math:`\varepsilon_{\min} = \min(5 \times 10^{-4}, \varepsilon_0)`.
Large initial regularization helps exploration; small final regularization
sharpens the transport plan.
