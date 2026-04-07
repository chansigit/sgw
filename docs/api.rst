API Reference
=============

Solver
------

.. autofunction:: torchgw.sampled_gw

.. autofunction:: torchgw.sampled_lowrank_gw

Graph Construction
------------------

.. autofunction:: torchgw.build_knn_graph

Joint Embedding
---------------

.. autofunction:: torchgw.joint_embedding

Internal Modules
----------------

Sinkhorn
^^^^^^^^

.. autofunction:: torchgw._solver._sinkhorn_torch

.. autofunction:: torchgw._solver._sinkhorn_differentiable

Sampling
^^^^^^^^

.. autofunction:: torchgw._sampling.sample_pairs_from_plan

.. autofunction:: torchgw._sampling.sample_pairs_gpu

Utilities
^^^^^^^^^

.. autofunction:: torchgw._utils.get_device
