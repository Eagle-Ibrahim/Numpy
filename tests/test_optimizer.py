"""Adam optimizer correctness tests."""
import numpy as np

from engine import Tensor
from nn.optim import Adam


def test_adam_minimizes_quadratic():
    # Minimize f(w) = sum((w - target)^2); gradient = 2(w - target).
    target = np.array([1.0, -2.0, 3.0])
    w = Tensor(np.zeros(3), requires_grad=True)
    opt = Adam([w], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        w.grad = 2.0 * (w.data - target)
        opt.step()
    assert np.allclose(w.data, target, atol=1e-2)


def test_adam_bias_correction_first_step():
    w = Tensor(np.array([0.0]), requires_grad=True)
    opt = Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    opt.zero_grad()
    w.grad = np.array([1.0])
    opt.step()
    # With full bias correction the first step magnitude ~= lr.
    assert np.isclose(abs(w.data[0]), 0.1, atol=1e-6)


def test_adam_skips_param_without_grad():
    w = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    opt = Adam([w], lr=0.1)
    opt.zero_grad()
    opt.step()  # grad is None -> no change
    assert np.allclose(w.data, [1.0, 2.0])


def test_invalid_lr_raises():
    import pytest

    w = Tensor(np.zeros(2), requires_grad=True)
    with pytest.raises(ValueError):
        Adam([w], lr=0.0)
