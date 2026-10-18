"""Sparse-aware forward inference.

A dense matmul that multiplies by zeroed weights does *not* save computation. To
demonstrate genuine savings we build a compressed representation that stores,
for each output neuron, only its active incoming connections, and a forward
kernel that touches only those connections.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from nn.layers import Linear, ReLU, Sequential, Tanh


class SparseLinear:
    """
    A column-compressed sparse implementation of a Linear layer.

    Description:
        Standard dense matrix multiplication (matmul) does not provide computational 
        savings even if the weights are zeroed. `SparseLinear` achieves genuine 
        efficiency by storing only the active incoming connections for each output 
        neuron, bypassing all pruned connections during the forward pass.

    Mathematical Context:
        The forward pass for output neuron $j$ is computed as:
        $y_j = \sum_{i \in \text{active}(j)} x_i \cdot w_{ij} + b_j$
        where $\text{active}(j)$ is the set of indices $i$ such that the mask 
        $M_{ij} = 1$. This reduces the complexity from $O(\text{in\_features} \cdot \text{out\_features})$ 
        to $O(\text{NNZ})$, where NNZ is the number of non-zero elements.

    Args:
        layer (Linear): A pruned `Linear` layer from which to extract 
                        the active weights and indices.

    Role in Pipeline:
        Acts as the optimized deployment format. After a model is pruned and 
        stabilized, it is converted to `SparseLinear` to demonstrate the theoretical 
        and practical reduction in multiply-accumulate (MAC) operations.
    """

    def __init__(self, layer: Linear) -> None:
        w_eff = layer.effective_weight()
        self.in_features, self.out_features = w_eff.shape
        self.rows: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []
        for j in range(self.out_features):
            rows = np.flatnonzero(layer.mask[:, j] > 0.5)
            self.rows.append(rows)
            self.weights.append(w_eff[rows, j].copy())
        self.bias = (
            layer.b.data.copy() if layer.b is not None
            else np.zeros(self.out_features)
        )
        self.nnz = int(sum(r.size for r in self.rows))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a sparse forward pass across the layer.

        Args:
            x (np.ndarray): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            np.ndarray: Output tensor of shape `(batch_size, out_features)`.
        """
        n = x.shape[0]
        out = np.empty((n, self.out_features), dtype=np.float64)
        for j in range(self.out_features):
            rows = self.rows[j]
            if rows.size == 0:
                out[:, j] = self.bias[j]          # neuron with no active inputs
            else:
                out[:, j] = x[:, rows] @ self.weights[j] + self.bias[j]
        return out


class SparseMLP:
    """
    A sparse-aware inference engine for a pruned MLP.

    Description:
        Wraps a `Sequential` model and replaces all `Linear` layers with 
        `SparseLinear` counterparts. All non-linear activations (Tanh, ReLU) 
        remain as dense element-wise operations.

    Args:
        model (Sequential): The trained and pruned `Sequential` model to convert.

    Role in Pipeline:
        The final stage of the hardware-efficiency demonstration. It allows 
        the user to run predictions on a pruned model and calculate the 
        exact number of active MACs used per sample.
    """

    def __init__(self, model: Sequential) -> None:
        self.ops: List = []
        for layer in model.layers:
            if isinstance(layer, Linear):
                self.ops.append(SparseLinear(layer))
            elif isinstance(layer, Tanh):
                self.ops.append(np.tanh)
            elif isinstance(layer, ReLU):
                self.ops.append(lambda a: np.maximum(a, 0.0))
            else:
                raise TypeError(f"unsupported layer for sparse inference: {layer}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executes the sparse forward pass through the entire network.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Raw output logits.
        """
        for op in self.ops:
            x = op.forward(x) if isinstance(op, SparseLinear) else op(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs sparse inference and returns the class with the highest logit.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.argmax(self.forward(x), axis=1)

    def active_macs(self) -> int:
        """
        Calculates the total number of active multiply-accumulate operations.

        Returns:
            int: Total MACs per single sample.
        """
        return int(sum(op.nnz for op in self.ops if isinstance(op, SparseLinear)))


def dense_macs(model: Sequential) -> int:
    """
    Calculates the total MACs for the dense version of the model.

    Description:
        Sums the product of input and output features for all linear layers, 
        ignoring any masks. This provides the baseline complexity.

    Args:
        model (Sequential): The model to analyze.

    Returns:
        int: Total MACs per single sample for a dense implementation.
    """
    return int(
        sum(l.in_features * l.out_features for l in model.linear_layers())
    )
