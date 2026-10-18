"""Forward-value correctness tests for autodiff operations."""
import numpy as np

from engine import Tensor, softmax, log_softmax
from engine.losses import softmax_cross_entropy


def test_add_sub_mul_div_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    assert np.allclose((a + b).data, [5, 7, 9])
    assert np.allclose((b - a).data, [3, 3, 3])
    assert np.allclose((a * b).data, [4, 10, 18])
    assert np.allclose((b / a).data, [4, 2.5, 2.0])


def test_matmul_forward():
    a = Tensor(np.arange(6).reshape(2, 3).astype(float))
    b = Tensor(np.arange(12).reshape(3, 4).astype(float))
    assert np.allclose((a @ b).data, a.data @ b.data)


def test_sum_mean_forward():
    a = Tensor(np.arange(6).reshape(2, 3).astype(float))
    assert np.isclose(a.sum().data, 15)
    assert np.isclose(a.mean().data, 2.5)
    assert np.allclose(a.sum(axis=0).data, [3, 5, 7])
    assert np.allclose(a.mean(axis=1).data, [1, 4])


def test_relu_tanh_forward():
    a = Tensor([-2.0, -0.5, 0.5, 2.0])
    assert np.allclose(a.relu().data, [0, 0, 0.5, 2.0])
    assert np.allclose(a.tanh().data, np.tanh([-2.0, -0.5, 0.5, 2.0]))


def test_softmax_is_stable_for_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)
    assert np.all(np.isfinite(log_softmax(z)))


def test_cross_entropy_value_matches_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 3))
    y = rng.integers(0, 3, size=5)
    logp = log_softmax(z)
    expected = -logp[np.arange(5), y].mean()
    loss = softmax_cross_entropy(Tensor(z), y)
    assert np.isclose(loss.data, expected)


def test_matmul_rank_guard():
    import pytest

    a = Tensor(np.zeros((2, 3, 4)))
    b = Tensor(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        a @ b
