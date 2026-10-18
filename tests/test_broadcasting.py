"""Broadcasting gradient correctness, including the (N,D)+(D,) bias case."""
import numpy as np

from engine import Tensor, check_gradient


def test_bias_broadcast_grad():
    # Y = X + b with X:(N,D), b:(D,). dL/db must sum over the batch axis.
    x = Tensor(np.random.default_rng(0).normal(size=(4, 3)))
    check_gradient(lambda b: (x + b).sum(), np.array([0.1, -0.2, 0.3]))


def test_scalar_broadcast_grad():
    x = Tensor(np.random.default_rng(1).normal(size=(3, 2)))
    check_gradient(lambda s: (x * s).sum(), np.array(2.0))


def test_singleton_dim_broadcast_grad():
    # a:(3,1) broadcast against b:(3,4)
    b = Tensor(np.random.default_rng(2).normal(size=(3, 4)))
    check_gradient(lambda a: (a * b).sum(), np.random.default_rng(3).normal(size=(3, 1)))


def test_multiple_broadcast_dims_grad():
    # a:(1,4) broadcast against b:(3,4)
    b = Tensor(np.random.default_rng(4).normal(size=(3, 4)))
    check_gradient(lambda a: (a + b).sum(), np.random.default_rng(5).normal(size=(1, 4)))


def test_bias_grad_is_batch_sum():
    x = Tensor(np.ones((5, 3)))
    b = Tensor(np.zeros(3), requires_grad=True)
    out = (x + b).sum()
    out.backward()
    assert np.allclose(b.grad, [5.0, 5.0, 5.0])
