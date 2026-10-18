"""Mandatory masked-weight correctness regression tests.

These guard the highest-priority requirement: a pruned connection must stay
exactly zero after optimizer updates, even when it carried nonzero Adam
momentum before being pruned, while active connections keep learning.
"""
import numpy as np

from engine import Tensor
from engine.losses import softmax_cross_entropy
from nn.layers import Linear
from nn.optim import Adam
from prune.mask import prune_to_sparsity
from nn.model import build_mlp
from prune.importance import magnitude_importance


def _single_layer_setup(seed=0):
    rng = np.random.default_rng(seed)
    layer = Linear(4, 3, rng, init="xavier")
    opt = Adam(layer.parameters(), lr=1e-2, masked_layers=[layer])
    return layer, opt, rng


def test_masked_forward_excludes_connection():
    layer, _, rng = _single_layer_setup()
    layer.mask[:, 0] = 0.0  # kill all inputs to output neuron 0
    x = Tensor(rng.normal(size=(5, 4)))
    out = layer(x)
    # Output neuron 0 should equal only its bias (zero here).
    assert np.allclose(out.data[:, 0], layer.b.data[0])


def test_masked_weight_has_zero_gradient():
    layer, _, rng = _single_layer_setup()
    layer.mask[1, 2] = 0.0
    x = Tensor(rng.normal(size=(6, 4)))
    y = np.array([0, 1, 2, 0, 1, 2])
    logits = layer(x)
    loss = softmax_cross_entropy(logits, y)
    loss.backward()
    assert layer.W.grad[1, 2] == 0.0


def test_pruned_weight_stays_zero_despite_adam_momentum():
    layer, opt, rng = _single_layer_setup()
    x = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 3, size=16)

    # Build up nonzero Adam momentum for all weights.
    for _ in range(5):
        opt.zero_grad()
        loss = softmax_cross_entropy(layer(x), y)
        loss.backward()
        opt.step()

    # Pick a weight with genuinely nonzero momentum, then prune it.
    key = id(layer.W)
    assert np.any(opt.m[key] != 0.0)
    i, j = 0, 0
    newly = np.zeros_like(layer.mask, dtype=bool)
    newly[i, j] = True
    layer.mask[i, j] = 0.0
    opt.reset_masked_state(layer.W, newly)
    layer.enforce_mask()

    # Several more optimizer steps must not resurrect the pruned weight.
    for _ in range(10):
        opt.zero_grad()
        loss = softmax_cross_entropy(layer(x), y)
        loss.backward()
        opt.step()
        assert layer.effective_weight()[i, j] == 0.0
        assert layer.W.grad[i, j] == 0.0

    assert layer.W.data[i, j] == 0.0


def test_moment_reset_alone_prevents_adam_resurrection():
    """Isolate the Adam fix: with enforce_mask DISABLED, only the moment reset
    keeps a pruned weight at zero.

    This is the subtle case: Adam's update is ``m_hat / (sqrt(v_hat) + eps)``,
    which is nonzero whenever the first moment ``m`` is nonzero -- *even when the
    current gradient is exactly zero*. So setting the mask and zeroing the
    gradient does NOT freeze the weight; momentum accumulated before pruning
    will drift it away from zero. Clearing the Adam moments is what actually
    prevents resurrection. ``enforce_mask`` is intentionally not used here so it
    cannot mask the bug.
    """
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng, init="xavier")
    opt = Adam(layer.parameters(), lr=0.05, masked_layers=[])  # NO enforce_mask
    x = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 3, size=16)

    # Build genuinely nonzero momentum for the target weight.
    for _ in range(8):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    i, j = 0, 0
    assert opt.m[id(layer.W)][i, j] != 0.0

    # Prune: set the weight to zero and mask it, then reset its Adam moments.
    layer.W.data[i, j] = 0.0
    layer.mask[i, j] = 0.0
    opt.reset_masked_state(layer.W, layer.mask < 0.5)

    # Without any enforce_mask, the weight must stay exactly zero purely because
    # its gradient is zero AND its moments were cleared.
    for _ in range(15):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
        assert layer.W.data[i, j] == 0.0


def test_skipping_moment_reset_causes_resurrection():
    """Guard the guard: prove the scenario above is real.

    If the Adam moments are NOT reset (and enforce_mask is off), the pruned
    weight demonstrably drifts away from zero. This documents the failure mode
    and ensures the reset in the test above is actually doing the work.
    """
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng, init="xavier")
    opt = Adam(layer.parameters(), lr=0.05, masked_layers=[])  # NO enforce_mask
    x = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 3, size=16)
    for _ in range(8):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    i, j = 0, 0
    layer.W.data[i, j] = 0.0
    layer.mask[i, j] = 0.0
    # Deliberately DO NOT reset moments.
    for _ in range(12):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    assert abs(layer.W.data[i, j]) > 1e-3  # resurrection actually happens


def test_active_weights_keep_learning_after_pruning():
    layer, opt, rng = _single_layer_setup()
    x = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 3, size=16)
    layer.mask[0, 0] = 0.0
    opt.reset_masked_state(layer.W, layer.mask < 0.5)
    layer.enforce_mask()

    before = layer.W.data.copy()
    for _ in range(10):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    after = layer.W.data

    # An active weight changed; the pruned one did not.
    assert after[0, 0] == 0.0
    assert not np.allclose(after[1, 1], before[1, 1])


def test_optimizer_state_cleared_for_pruned():
    layer, opt, rng = _single_layer_setup()
    x = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 3, size=16)
    for _ in range(3):
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    newly = np.zeros_like(layer.mask, dtype=bool)
    newly[2, 1] = True
    opt.reset_masked_state(layer.W, newly)
    key = id(layer.W)
    assert opt.m[key][2, 1] == 0.0
    assert opt.v[key][2, 1] == 0.0


def test_full_model_prune_keeps_masked_zero():
    model = build_mlp(seed=1)
    opt = Adam(model.parameters(), lr=1e-2, masked_layers=model.linear_layers())
    prune_to_sparsity(model, opt, 0.5, magnitude_importance(model))
    masks_before = [l.mask.copy() for l in model.linear_layers()]
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(32, 2)))
    y = rng.integers(0, 2, size=32)
    for _ in range(5):
        opt.zero_grad()
        softmax_cross_entropy(model(x), y).backward()
        opt.step()
    for layer, m0 in zip(model.linear_layers(), masks_before):
        # Masked connections remain exactly zero in the effective weights.
        assert np.all(layer.effective_weight()[m0 < 0.5] == 0.0)
