Benchmark
=========

Setup
-----

- **Task**: Align a 2D spiral to a 3D Swiss roll (same angular parameterization)
- **Preprocessing**: kNN graph (k=5) + Dijkstra shortest paths (identical for both methods)
- **Hardware**: NVIDIA H100 GPU, Intel Xeon CPU
- **Metric**: Spearman rank correlation between matched angular positions

Pure OT Solver Time
--------------------

Timings measure **only** the GPU computation (cost matrix assembly + Sinkhorn
projection), excluding graph construction, Dijkstra, and sampling overhead.

.. list-table::
   :header-rows: 1

   * - Scale
     - Method
     - Solver time (100 iter)
     - GW distance
     - Spearman ρ
   * - 400 vs 500
     - POT ``entropic_gromov_wasserstein``
     - 1.6s
     - 3.57e-03
     - 0.999
   * - 400 vs 500
     - SGW ``sampled_gw``
     - **0.9s**
     - **1.39e-03**
     - 0.998
   * - 4000 vs 5000
     - POT ``entropic_gromov_wasserstein``
     - 183s
     - 3.21e-03
     - 0.999
   * - 4000 vs 5000
     - SGW ``sampled_gw``
     - **2.4s**
     - **1.17e-03**
     - **0.999**

At 4000×5000, SGW's pure OT solver is **~75× faster** than POT.

Time Breakdown (4000 vs 5000, 100 iterations)
----------------------------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Time
     - Share
   * - Sampling (CPU)
     - 9.0s
     - 38%
   * - Dijkstra (CPU)
     - 12.2s
     - 52%
   * - GPU cost matrix + transfer
     - 1.4s
     - 6%
   * - Sinkhorn (GPU)
     - 1.1s
     - 4%
   * - **Total**
     - **23.6s**
     -

The GPU solver is not the bottleneck — CPU-side Dijkstra and Python sampling
dominate at large scale.

400 vs 500
-----------

.. image:: demo_spiral_to_swissroll_400v500.png
   :width: 100%
   :alt: 400 vs 500 comparison

4000 vs 5000
-------------

.. image:: demo_spiral_to_swissroll_4000v5000.png
   :width: 100%
   :alt: 4000 vs 5000 comparison

Reproducing
-----------

.. code-block:: bash

   pip install pot  # needed for the POT baseline only
   PYTHONPATH=. python examples/demo_spiral_to_swissroll.py
