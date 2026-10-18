"""Self-pruning system: importance scoring, schedule, masks, sparse inference."""
from prune.importance import (
    compute_importance,
    magnitude_importance,
    taylor_importance,
)
from prune.mask import achieved_sparsity, prune_to_sparsity
from prune.schedule import cubic_sparsity
from prune.sparse_inference import SparseLinear, SparseMLP, dense_macs

__all__ = [
    "compute_importance",
    "magnitude_importance",
    "taylor_importance",
    "achieved_sparsity",
    "prune_to_sparsity",
    "cubic_sparsity",
    "SparseLinear",
    "SparseMLP",
    "dense_macs",
]
