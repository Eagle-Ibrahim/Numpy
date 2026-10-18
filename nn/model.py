"""MLP model builder for the spiral classification task."""
from __future__ import annotations

from typing import List

import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear, Sequential, Tanh


def build_mlp(
    in_features: int = 2,
    hidden: int = 64,
    out_features: int = 2,
    seed: int = 0,
) -> Sequential:
    """
    Builds a Multi-Layer Perceptron (MLP) architecture for the spiral classification task.

    Architectural Context:
        The default structure is a 2-64-64-2 network. It consists of three 
        linear layers with Tanh activations between them. Xavier initialization 
        is applied to ensure the variance of activations remains stable, 
        preventing vanishing or exploding gradients in a Tanh-based network.

    Args:
        in_features (int): Input dimension (typically 2 for the Spiral dataset).
        hidden (int): Number of neurons in the hidden layers.
        out_features (int): Number of output classes (typically 2).
        seed (int): Random seed for reproducible initialization.

    Returns:
        Sequential: A Sequential model containing the linear and activation layers.
    """
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(in_features, hidden, rng, init="xavier"),
        Tanh(),
        Linear(hidden, hidden, rng, init="xavier"),
        Tanh(),
        Linear(hidden, out_features, rng, init="xavier"),
    )


def count_parameters(model: Sequential) -> int:
    """
    Calculates the total number of trainable scalar parameters.

    Args:
        model (Sequential): The neural network model.

    Returns:
        int: Total count of elements across all weight and bias tensors.
    """
    return int(sum(p.data.size for p in model.parameters()))


def count_active_weights(model: Sequential) -> int:
    """
    Determines the number of currently active (unpruned) weight connections.

    Args:
        model (Sequential): The neural network model.

    Returns:
        int: Sum of all active weights (where mask == 1) across all Linear layers.

    Role in Pipeline:
        Used to track the actual sparsity of the model during the pruning process.
    """
    return int(sum(layer.num_active() for layer in model.linear_layers()))


def count_eligible_weights(model: Sequential) -> int:
    """
    Determines the total number of prunable weight connections.

    Args:
        model (Sequential): The neural network model.

    Returns:
        int: Total count of weight elements. Biases are excluded as they are 
              typically not subject to pruning.
    """
    return int(sum(layer.num_eligible() for layer in model.linear_layers()))


def predict(model: Sequential, x: np.ndarray) -> np.ndarray:
    """
    Generates class predictions for a batch of inputs.

    Args:
        model (Sequential): The trained neural network.
        x (np.ndarray): Input data array of shape (N, in_features).

    Returns:
        np.ndarray: An array of predicted class indices (argmax of logits).

    Role in Pipeline:
        Used during evaluation to calculate accuracy on the test set.
    """
    logits = model(Tensor(x, requires_grad=False)).data
    return np.argmax(logits, axis=1)
