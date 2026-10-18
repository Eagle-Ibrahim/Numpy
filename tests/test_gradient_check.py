"""Central finite-difference gradient checks for every differentiable op."""
import numpy as np

from engine import Tensor, check_gradient
from engine.losses import softmax_cross_entropy


def test_add_grad():
    other = Tensor(np.array([1.0, -2.0, 0.5]))
    check_gradient(lambda x: (x + other).sum(), np.array([0.3, -1.0, 2.0]))


def test_sub_grad():
    other = Tensor(np.array([1.0, -2.0, 0.5]))
    check_gradient(lambda x: (x - other).sum(), np.array([0.3, -1.0, 2.0]))


def test_mul_grad():
    other = Tensor(np.array([1.5, -2.0, 0.5]))
    check_gradient(lambda x: (x * other).sum(), np.array([0.3, -1.0, 2.0]))


def test_div_grad():
    denom = Tensor(np.array([1.5, -2.0, 0.5]))
    check_gradient(lambda x: (x / denom).sum(), np.array([0.3, -1.0, 2.0]))


def test_matmul_grad():
    w = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    check_gradient(
        lambda x: (x @ w).sum(),
        np.random.default_rng(1).normal(size=(2, 3)),
    )


def test_sum_grad():
    check_gradient(lambda x: (x * 2.0).sum(), np.random.default_rng(0).normal(size=(2, 3)))


def test_mean_grad():
    check_gradient(lambda x: x.mean(), np.random.default_rng(0).normal(size=(2, 3)))
    check_gradient(lambda x: x.mean(axis=0).sum(),
                   np.random.default_rng(1).normal(size=(2, 3)))


def test_relu_grad_away_from_zero():
    # Avoid the nondifferentiable point at exactly 0.
    x = np.array([-2.0, -0.7, 0.4, 1.3])
    check_gradient(lambda t: t.relu().sum(), x)


def test_tanh_grad():
    check_gradient(lambda x: x.tanh().sum(), np.array([-1.2, 0.3, 0.9, 2.0]))


def test_cross_entropy_grad():
    y = np.array([0, 2, 1])

    def build(logits):
        return softmax_cross_entropy(logits, y)

    check_gradient(build, np.random.default_rng(3).normal(size=(3, 3)),
                   atol=1e-6, rtol=1e-4)


def test_repeated_path_grad():
    # x used twice: gradient must accumulate along both paths.
    check_gradient(lambda x: (x * x).sum(), np.array([0.5, -1.0, 2.0]))


def test_composed_mlp_like_grad():
    w1 = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    w2 = Tensor(np.random.default_rng(1).normal(size=(4, 2)))
    y = np.array([0, 1])

    def build(x):
        h = (x @ w1).tanh()
        logits = h @ w2
        return softmax_cross_entropy(logits, y)

    check_gradient(build, np.random.default_rng(2).normal(size=(2, 3)))
