"""Autodiff helpers: topological ordering utilities and gradient checking.

The backward pass itself lives on :meth:`engine.tensor.Tensor.backward`. This
module exposes reusable *numerical* gradient-checking tools used by the test
suite to validate every differentiable operation against central finite
differences.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from engine.tensor import Tensor


def topological_order(root: Tensor) -> List[Tensor]:
    """
    Returns all nodes reachable from the root tensor in topological (parents-first) order.

    Mathematical Context:
        In reverse-mode autodiff, a node's gradient can only be computed after all its 
        children (in the forward pass) have contributed their gradients. Topological 
        sorting ensures that we process nodes in the reverse order of their dependency.

    Args:
        root (Tensor): The output tensor (usually the loss) from which to start the traversal.

    Returns:
        List[Tensor]: A list of tensors sorted such that every tensor appears after 
                      all its predecessors in the computation graph.

    Role in Pipeline:
        Used by `Tensor.backward()` to determine the execution sequence of the 
        gradient accumulation process.
    """
    topo: List[Tensor] = []
    visited: set[int] = set()

    def build(node: Tensor) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node._prev:
            build(parent)
        topo.append(node)

    build(root)
    return topo


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes the central finite-difference gradient of a scalar function f at point x.

    Mathematical Context:
        The derivative of f with respect to x_i is approximated as:
        df/dx_i ≈ (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
        where e_i is the unit vector in the i-th dimension.
        Central difference is used as it provides a more accurate approximation 
        (O(eps^2)) than forward difference (O(eps)).

    Args:
        f (Callable[[np.ndarray], float]): The scalar-valued function to differentiate.
        x (np.ndarray): The point at which to evaluate the gradient.
        eps (float, optional): The small perturbation step. Defaults to 1e-6.

    Returns:
        np.ndarray: The numerical gradient of f at x.

    Role in Pipeline:
        Provides a ground-truth baseline to verify the correctness of the 
        analytical gradients implemented in the tensor engine.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readonly"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        f_plus = f(x)
        x[idx] = original - eps
        f_minus = f(x)
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def check_gradient(
    build_scalar: Callable[[Tensor], Tensor],
    x: np.ndarray,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Validates the analytical gradient against the numerical finite-difference gradient.

    Args:
        build_scalar (Callable[[Tensor], Tensor]): A function that takes a leaf Tensor 
                                                and returns a scalar Tensor.
        x (np.ndarray): The input values for the leaf Tensor.
        eps (float, optional): Step size for the numerical gradient. Defaults to 1e-6.
        atol (float, optional): Absolute tolerance for `np.allclose`. Defaults to 1e-6.
        rtol (float, optional): Relative tolerance for `np.allclose`. Defaults to 1e-4.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: A tuple containing (analytical_grad, 
                                             numerical_grad, max_absolute_error).

    Raises:
        ValueError: If the `build_scalar` function does not return a scalar.
        AssertionError: If the analytical and numerical gradients differ beyond the 
                        specified tolerances.

    Role in Pipeline:
        Crucial for unit testing the autodiff engine; ensures that every operator's 
        backward pass is mathematically correct.
    """
    x = x.astype(np.float64)

    leaf = Tensor(x.copy(), requires_grad=True)
    out = build_scalar(leaf)
    if out.data.size != 1:
        raise ValueError("build_scalar must return a scalar tensor")
    leaf.zero_grad()
    out.backward()
    analytical = leaf.grad.copy()

    def f(arr: np.ndarray) -> float:
        node = Tensor(arr.copy(), requires_grad=False)
        return float(build_scalar(node).data)

    numerical = numerical_gradient(f, x, eps=eps)
    max_err = float(np.max(np.abs(analytical - numerical)))
    assert np.allclose(analytical, numerical, atol=atol, rtol=rtol), (
        f"gradient check failed: max_abs_error={max_err:.3e}\n"
        f"analytical=\n{analytical}\nnumerical=\n{numerical}"
    )
    return analytical, numerical, max_err
