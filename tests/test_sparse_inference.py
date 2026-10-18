"""Sparse-aware inference must match dense masked inference within tolerance."""
import numpy as np

from engine import Tensor
from nn.model import build_mlp
from nn.optim import Adam
from prune.importance import magnitude_importance
from prune.mask import prune_to_sparsity
from prune.sparse_inference import SparseMLP, dense_macs


def test_sparse_matches_dense_predictions():
    model = build_mlp(seed=2)
    opt = Adam(model.parameters(), masked_layers=model.linear_layers())
    prune_to_sparsity(model, opt, 0.85, magnitude_importance(model))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    dense_logits = model(Tensor(x, requires_grad=False)).data
    sparse = SparseMLP(model)
    sparse_logits = sparse.forward(x)

    assert np.allclose(dense_logits, sparse_logits, atol=1e-10)
    assert np.array_equal(np.argmax(dense_logits, 1), sparse.predict(x))


def test_sparse_handles_dead_neuron():
    model = build_mlp(seed=3)
    # Kill all incoming connections to one hidden neuron in the first layer.
    first = model.linear_layers()[0]
    first.mask[:, 0] = 0.0
    first.enforce_mask()

    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 2))
    dense = model(Tensor(x, requires_grad=False)).data
    sparse = SparseMLP(model).forward(x)
    assert np.allclose(dense, sparse, atol=1e-10)


def test_mac_counts_reflect_sparsity():
    model = build_mlp(seed=4)
    opt = Adam(model.parameters(), masked_layers=model.linear_layers())
    prune_to_sparsity(model, opt, 0.9, magnitude_importance(model))
    sparse = SparseMLP(model)
    assert sparse.active_macs() < dense_macs(model)
