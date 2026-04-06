API Reference
=============

Solver
------

.. autofunction:: sgw.sampled_gw

Graph Construction
------------------

.. autofunction:: sgw.build_knn_graph

Joint Embedding
---------------

.. autofunction:: sgw.joint_embedding

Internal Modules
----------------

Sinkhorn
^^^^^^^^

.. autofunction:: sgw._solver._sinkhorn_torch

.. autofunction:: sgw._solver._sinkhorn_differentiable

Sampling
^^^^^^^^

.. autofunction:: sgw._sampling.sample_pairs_from_plan

Utilities
^^^^^^^^^

.. autofunction:: sgw._utils.get_device

.. autofunction:: sgw._utils.maybe_gc
