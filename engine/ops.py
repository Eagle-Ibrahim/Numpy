"""Differentiable operations on :class:`~engine.tensor.Tensor`.

Each operation computes a forward value and installs a ``_backward`` closure
that propagates the upstream gradient into its parents, correctly reversing
NumPy broadcasting via :func:`~engine.tensor.unbroadcast`.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from engine.tensor import ArrayLike, Tensor, as_tensor, unbroadcast

Axis = Optional[Union[int, Tuple[int, ...]]]


def _result_requires_grad(*tensors: Tensor) -> bool:
    """
    Determines if the resulting tensor should track gradients based on its inputs.

    Args:
        *tensors (Tensor): Variable number of input tensors.

    Returns:
        bool: True if at least one input tensor requires gradients.
    """
    return any(t.requires_grad for t in tensors)


def add(a: ArrayLike, b: ArrayLike) -> Tensor:
    """
    Performs element-wise addition of two tensors.

    Mathematical Context:
        The gradient of f(a, b) = a + b with respect to both a and b is 1.
        ∇a(a + b) = 1, ∇b(a + b) = 1.

    Args:
        a (ArrayLike): The first operand.
        b (ArrayLike): The second operand.

    Returns:
        Tensor: A new tensor containing the element-wise sum.

    Role in Pipeline:
        Fundamental operation for bias addition and residual connections.
    """
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data + b.data, _result_requires_grad(a, b), (a, b), "add")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(unbroadcast(out.grad, a.data.shape))
        b._accumulate(unbroadcast(out.grad, b.data.shape))

    out._backward = _backward
    return out


def sub(a: ArrayLike, b: ArrayLike) -> Tensor:
    """
    Performs element-wise subtraction of two tensors.

    Mathematical Context:
        The gradient of f(a, b) = a - b with respect to a is 1 and with respect to b is -1.
        ∇a(a - b) = 1, ∇b(a - b) = -1.

    Args:
        a (ArrayLike): The minuend.
        b (ArrayLike): The subtrahend.

    Returns:
        Tensor: A new tensor containing the element-wise difference.

    Role in Pipeline:
        Used in weight updates and distance calculations.
    """
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data - b.data, _result_requires_grad(a, b), (a, b), "sub")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(unbroadcast(out.grad, a.data.shape))
        b._accumulate(unbroadcast(-out.grad, b.data.shape))

    out._backward = _backward
    return out


def mul(a: ArrayLike, b: ArrayLike) -> Tensor:
    """
    Performs element-wise multiplication of two tensors.

    Mathematical Context:
        The gradient of f(a, b) = a * b follows the product rule:
        ∇a(a * b) = b, ∇b(a * b) = a.

    Args:
        a (ArrayLike): The first operand.
        b (ArrayLike): The second operand.

    Returns:
        Tensor: A new tensor containing the element-wise product.

    Role in Pipeline:
        Critical for implementing masks in pruned layers and scaling operations.
    """
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data * b.data, _result_requires_grad(a, b), (a, b), "mul")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(unbroadcast(out.grad * b.data, a.data.shape))
        b._accumulate(unbroadcast(out.grad * a.data, b.data.shape))

    out._backward = _backward
    return out


def div(a: ArrayLike, b: ArrayLike) -> Tensor:
    """
    Performs element-wise division of two tensors.

    Mathematical Context:
        The gradient of f(a, b) = a / b:
        ∇a(a / b) = 1 / b
        ∇b(a / b) = -a / (b^2)

    Args:
        a (ArrayLike): The dividend.
        b (ArrayLike): The divisor.

    Returns:
        Tensor: A new tensor containing the element-wise quotient.

    Role in Pipeline:
        Used in normalization and mean calculations.
    """
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data / b.data, _result_requires_grad(a, b), (a, b), "div")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(unbroadcast(out.grad / b.data, a.data.shape))
        b._accumulate(
            unbroadcast(-out.grad * a.data / (b.data ** 2), b.data.shape)
        )

    out._backward = _backward
    return out


def matmul(a: ArrayLike, b: ArrayLike) -> Tensor:
    """
    Performs matrix multiplication for 2-D operands (N, K) @ (K, M) -> (N, M).

    Mathematical Context:
        Given C = A @ B, the gradients are:
        ∇A = ∇C @ B^T
        ∇B = A^T @ ∇C

    Args:
        a (ArrayLike): The left operand (matrix of shape N x K).
        b (ArrayLike): The right operand (matrix of shape K x M).

    Returns:
        Tensor: The resulting matrix of shape N x M.

    Raises:
        ValueError: If either operand is not 2-D, as other ranks are intentionally unsupported.

    Role in Pipeline:
        The primary linear transformation operation in MLP layers.
    """
    a, b = as_tensor(a), as_tensor(b)
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(
            f"matmul only supports 2-D @ 2-D; got shapes {a.data.shape} @ "
            f"{b.data.shape}. Other ranks are intentionally unsupported."
        )
    out = Tensor(a.data @ b.data, _result_requires_grad(a, b), (a, b), "matmul")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(out.grad @ b.data.T)
        b._accumulate(a.data.T @ out.grad)

    out._backward = _backward
    return out


def sum(a: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """
    Computes the sum of elements across the specified axis.

    Mathematical Context:
        The gradient of ∑x is 1 for all x. In reverse mode, the upstream gradient 
        is broadcast back to the original shape of the input.

    Args:
        a (ArrayLike): The input tensor.
        axis (Axis, optional): Axis or axes along which to sum. Defaults to None (sum all).
        keepdims (bool, optional): Whether to maintain the reduced dimensions. Defaults to False.

    Returns:
        Tensor: A tensor containing the sum.

    Role in Pipeline:
        Used in loss calculations and computing global norms.
    """
    a = as_tensor(a)
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims),
                 a.requires_grad, (a,), "sum")

    def _backward() -> None:
        if out.grad is None:
            return
        grad = out.grad
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis)
        a._accumulate(np.broadcast_to(grad, a.data.shape).copy())

    out._backward = _backward
    return out


def mean(a: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """
    Computes the arithmetic mean of elements across the specified axis.

    Mathematical Context:
        The gradient of (1/N)∑x is 1/N for all x.

    Args:
        a (ArrayLike): The input tensor.
        axis (Axis, optional): Axis or axes along which to compute the mean. Defaults to None.
        keepdims (bool, optional): Whether to maintain the reduced dimensions. Defaults to False.

    Returns:
        Tensor: A tensor containing the mean.

    Role in Pipeline:
        Crucial for computing Mean Squared Error or averaging losses across a batch.
    """
    a = as_tensor(a)
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims),
                 a.requires_grad, (a,), "mean")

    if axis is None:
        count = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        count = int(np.prod([a.data.shape[ax] for ax in axes]))

    def _backward() -> None:
        if out.grad is None:
            return
        grad = out.grad
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis)
        a._accumulate(np.broadcast_to(grad, a.data.shape).copy() / count)

    out._backward = _backward
    return out


def relu(a: ArrayLike) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Mathematical Context:
        f(x) = max(0, x)
        ∇f(x) = 1 if x > 0 else 0.

    Args:
        a (ArrayLike): The input tensor.

    Returns:
        Tensor: The tensor after applying ReLU.

    Role in Pipeline:
        Non-linear activation used in hidden layers to prevent vanishing gradients.
    """
    a = as_tensor(a)
    out = Tensor(np.maximum(a.data, 0.0), a.requires_grad, (a,), "relu")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(out.grad * (a.data > 0.0))

    out._backward = _backward
    return out


def tanh(a: ArrayLike) -> Tensor:
    """
    Applies the hyperbolic tangent (tanh) activation function.

    Mathematical Context:
        f(x) = tanh(x)
        ∇f(x) = 1 - tanh(x)^2.

    Args:
        a (ArrayLike): The input tensor.

    Returns:
        Tensor: The tensor after applying tanh.

    Role in Pipeline:
        The primary activation function for the current MLP implementation.
    """
    a = as_tensor(a)
    t = np.tanh(a.data)
    out = Tensor(t, a.requires_grad, (a,), "tanh")

    def _backward() -> None:
        if out.grad is None:
            return
        a._accumulate(out.grad * (1.0 - t * t))

    out._backward = _backward
    return out

