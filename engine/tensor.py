"""Core reverse-mode autodiff tensor.

A :class:`Tensor` wraps a NumPy ``ndarray`` and records the operations that
produced it so that gradients can be computed by reverse-mode automatic
differentiation. The implementation is intentionally small and explicit so the
whole backward pass can be read and defended in an interview.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int, list]


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduces a gradient tensor to match the shape of its original source tensor after broadcasting.

    Mathematical Context:
        In NumPy, broadcasting expands a smaller array to a larger shape. According to the 
        Chain Rule, the gradient of a broadcasted operation is the sum of the gradients 
        along the axes that were expanded.

    Args:
        grad (np.ndarray): The gradient coming back from the downstream operation.
        shape (Tuple[int, ...]): The original shape of the tensor before broadcasting.

    Returns:
        np.ndarray: The summed gradient that matches the original tensor's shape.

    Role in Pipeline:
        Esssential for the Backward Pass. It ensures that when a scalar or small vector is 
        added to a large matrix, the resulting gradient is correctly aggregated back into 
        the smaller shape.
    """
    # Sum away extra leading dimensions that broadcasting prepended.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum away dimensions that were broadcast from size 1.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """
    A node in the computational graph that facilitates reverse-mode automatic differentiation.

    The Tensor class wraps a NumPy array and maintains a lineage of operations (_prev) 
    and a gradient calculation closure (_backward), allowing the system to traverse 
    the graph in reverse topological order to compute gradients via the chain rule.

    Attributes:
        data (np.ndarray): The actual numerical value of the tensor.
        grad (Optional[np.ndarray]): The accumulated gradient w.r.t. the data.
        requires_grad (bool): Indicator if the tensor is a leaf or parameter requiring optimization.
        _prev (Tuple[Tensor, ...]): The immediate parent tensors in the computation graph.
        _backward (Callable): The closure that computes and accumulates gradients in parents.
        _op (str): The operation that produced this tensor (e.g., "matmul", "add").
    """

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """
        Initializes a Tensor node.

        Args:
            data (ArrayLike): The initial value. If a Tensor is passed, its raw data is extracted.
            requires_grad (bool): Whether this tensor should track and accumulate gradients.
            _prev (Tuple[Tensor, ...]): Parent nodes in the graph (used by ops).
            _op (str): Name of the operation that created this tensor.
        """
        if isinstance(data, Tensor):
            data = data.data
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._prev: Tuple["Tensor", ...] = tuple(_prev)
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op

    # -- basic helpers -------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the dimensions of the underlying NumPy array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions (rank) of the tensor."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self.data.size

    def zero_grad(self) -> None:
        """
        Resets the stored gradient of this tensor to None.

        Role in Pipeline:
            Critical at the start of every optimization step. This prevents gradients from 
            the previous batch from accumulating into the current update.
        """
        self.grad = None

    def _accumulate(self, grad: np.ndarray) -> None:
        """
        Adds a gradient contribution to the tensor's existing gradient.

        This method implements the additive nature of reverse-mode autodiff: if a tensor 
        is used in multiple downstream operations, its total gradient is the sum of 
        gradients from all those operations.

        Args:
            grad (np.ndarray): The gradient contribution to add.

        Role in Pipeline:
            Used by the `_backward` closures of operations to push gradients back 
            to their parents.
        """
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    # -- backward pass -------------------------------------------------
    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Initiates the reverse-mode automatic differentiation process from this tensor.

        The function performs a topological sort of the computation graph to ensure that 
        each node's gradient is computed only after all its downstream consumers have 
        been processed. It then seeds the gradient and triggers the `_backward` 
        closures in reverse order.

        Args:
            gradient (Optional[np.ndarray]): The seed gradient. If None, it defaults 
                to ones (assuming this tensor is a scalar loss).

        Raises:
            RuntimeError: If the tensor is non-scalar and no seed gradient is provided.
            ValueError: If the provided seed gradient shape does not match the tensor shape.

        Role in Pipeline:
            The core of the training loop. This converts the scalar loss into 
            gradients for every parameter in the network.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: "Tensor") -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar tensor requires an explicit "
                    "gradient argument of matching shape."
                )
            gradient = np.ones_like(self.data)
        else:
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.shape != self.data.shape:
                raise ValueError(
                    f"seed gradient shape {gradient.shape} does not match "
                    f"tensor shape {self.data.shape}"
                )

        # Seed the output gradient, then process nodes in reverse topo order.
        self.grad = gradient if self.grad is None else self.grad + gradient
        for node in reversed(topo):
            node._backward()


    # -- operator overloads (delegated to ops module) ------------------
    def __add__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise addition.
        
        Delegates to engine.ops.add.
        """
        from engine import ops

        return ops.add(self, other)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise addition (right-hand side).
        
        Delegates to engine.ops.add.
        """
        from engine import ops

        return ops.add(self, other)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise subtraction.
        
        Delegates to engine.ops.sub.
        """
        from engine import ops

        return ops.sub(self, other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise subtraction (right-hand side).
        
        Delegates to engine.ops.sub.
        """
        from engine import ops

        return ops.sub(other, self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise multiplication.
        
        Delegates to engine.ops.mul.
        """
        from engine import ops

        return ops.mul(self, other)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise multiplication (right-hand side).
        
        Delegates to engine.ops.mul.
        """
        from engine import ops

        return ops.mul(self, other)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise division.
        
        Delegates to engine.ops.div.
        """
        from engine import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        """
        Performs element-wise division (right-hand side).
        
        Delegates to engine.ops.div.
        """
        from engine import ops

        return ops.div(other, self)

    def __neg__(self) -> "Tensor":
        """
        Performs unary negation.
        
        Delegates to engine.ops.mul with a constant -1.0.
        """
        from engine import ops

        return ops.mul(self, -1.0)

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        """
        Performs matrix multiplication.
        
        Delegates to engine.ops.matmul.
        """
        from engine import ops

        return ops.matmul(self, other)

    def matmul(self, other: ArrayLike) -> "Tensor":
        """
        Performs matrix multiplication.
        
        Delegates to engine.ops.matmul.
        """
        from engine import ops

        return ops.matmul(self, other)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        """
        Computes the sum of elements across specified axes.
        
        Delegates to engine.ops.sum.
        """
        from engine import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        """
        Computes the arithmetic mean of elements across specified axes.
        
        Delegates to engine.ops.mean.
        """
        from engine import ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> "Tensor":
        """
        Applies the Rectified Linear Unit (ReLU) activation function.
        
        Delegates to engine.ops.relu.
        """
        from engine import ops

        return ops.relu(self)

    def tanh(self) -> "Tensor":
        """
        Applies the hyperbolic tangent (tanh) activation function.
        
        Delegates to engine.ops.tanh.
        """
        from engine import ops

        return ops.tanh(self)

    def __repr__(self) -> str:
        """Returns a string representation of the Tensor for debugging."""
        return (
            f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, "
            f"op={self._op!r})"
        )


def as_tensor(x: ArrayLike) -> Tensor:
    """Wrap ``x`` in a (non-grad) tensor if it is not already a tensor."""
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
