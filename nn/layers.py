"""Composable neural-network layers built on the custom autodiff engine.

The :class:`Linear` layer is *mask aware*: it multiplies its weight matrix by a
binary mask before the forward matmul, so that pruned connections contribute
nothing to the output and—crucially—receive exactly zero gradient through the
fixed mask (``dL/dW = M * dL/dW_eff``).
"""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np

from engine.tensor import Tensor


class Module:
    """
    Base class for all neural network components.

    Exposes a uniform interface for forward passes and parameter tracking,
    mimicking the design patterns of modern deep learning frameworks.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed by the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The processed output tensor.
        """
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        """
        Allow modules to be called as functions, routing logic to `forward`.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output of the forward pass.
        """
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """
        Returns all trainable parameters of the module.

        Returns:
            List[Tensor]: A list of Tensors representing the weights and biases.
        """
        return []


class Linear(Module):
    """
    Fully connected (dense) layer implementing the operation: y = x @ (M * W) + b.

    Mathematical Context:
        The layer applies a linear transformation. To support pruning, the 
        weight matrix W is element-wise multiplied by a binary mask M.
        The gradient flow is: dL/dW = M * (dL/dy @ x^T).
        This ensures that pruned weights (M=0) never receive updates.

    Args:
        in_features (int): Dimensionality of the input space.
        out_features (int): Dimensionality of the output space.
        rng (np.random.Generator): Generator for reproducible parameter initialization.
        init (Literal["xavier", "he"]): Initialization strategy. 
                                      "xavier" is optimized for Tanh; "he" for ReLU.
        bias (bool): Whether to include a learnable bias vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init: Literal["xavier", "he"] = "xavier",
        bias: bool = True,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if init == "xavier":
            # Xavier/Glorot keeps tanh activation variance stable across layers.
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "he":
            std = np.sqrt(2.0 / in_features)
        else:
            raise ValueError(f"unknown init {init!r}")

        w = rng.normal(0.0, std, size=(in_features, out_features))
        self.W = Tensor(w, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        # Binary mask over W; 1 = active connection, 0 = pruned.
        self.mask = np.ones((in_features, out_features), dtype=np.float64)

    def effective_weight(self) -> np.ndarray:
        """
        Computes the pruned weight matrix (M * W).

        Returns:
            np.ndarray: The weight matrix with pruned connections zeroed out.

        Role in Pipeline:
            Used during sparse inference to avoid multiplying by the mask repeatedly.
        """
        return self.mask * self.W.data

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes the linear transformation: y = x @ (mask * W) + b.

        Args:
            x (Tensor): Input tensor of shape (N, in_features).

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        mask_t = Tensor(self.mask, requires_grad=False)
        w_eff = self.W * mask_t               # dL/dW = mask * dL/dw_eff
        out = x.matmul(w_eff)
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self) -> List[Tensor]:
        """
        Returns the trainable weight and bias tensors.

        Returns:
            List[Tensor]: A list containing [W, b] (if bias is enabled).
        """
        return [self.W] + ([self.b] if self.b is not None else [])

    def num_eligible(self) -> int:
        """
        Calculates the total number of weights available for pruning.

        Returns:
            int: Total elements in the weight matrix.
        """
        return self.mask.size

    def num_active(self) -> int:
        """
        Calculates the number of weights that have not been pruned.

        Returns:
            int: Sum of the binary mask elements.
        """
        return int(self.mask.sum())

    def enforce_mask(self) -> None:
        """
        Hard-zeros the underlying Tensor data according to the mask.

        Role in Pipeline:
            Defensive operation to ensure that any floating point drift in 
            pruned weights is eliminated.
        """
        self.W.data *= self.mask


class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) activation layer.
    
    Mathematical Context: f(x) = max(0, x).
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies ReLU activation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Activated tensor.
        """
        return x.relu()


class Tanh(Module):
    """
    Hyperbolic Tangent (Tanh) activation layer.
    
    Mathematical Context: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies Tanh activation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Activated tensor.
        """
        return x.tanh()


class Sequential(Module):
    """
    A container for stacking modules in a linear sequence.

    Args:
        *layers (Module): Variable number of Module instances to chain.
    """

    def __init__(self, *layers: Module) -> None:
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes the input through each layer in the sequence.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The final output after all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        """
        Aggregates parameters from all nested modules.

        Returns:
            List[Tensor]: A flat list of all trainable tensors in the sequence.
        """
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def linear_layers(self) -> List[Linear]:
        """
        Filters the sequence to return only the Linear layers.

        Returns:
            List[Linear]: A list of Linear modules for pruning or analysis.

        Role in Pipeline:
            Used by the pruning engine to identify target weight matrices.
        """
        return [layer for layer in self.layers if isinstance(layer, Linear)]
