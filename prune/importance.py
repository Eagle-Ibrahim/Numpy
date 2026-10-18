"""Connection importance scoring for pruning decisions.

Two criteria are provided:

* **Magnitude** ``|W|`` - the required baseline.
* **First-order Taylor saliency** ``|W * dL/dW_eff|`` - a local estimate of the
  loss increase caused by setting a weight to zero.

Only *active* connections are scored. Masked connections are never scored using
their (zero) masked gradients, which would be a meaningless regrowth signal.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from engine.losses import softmax_cross_entropy
from engine.tensor import Tensor
from nn.layers import Linear, Sequential


def magnitude_importance(model: Sequential) -> Dict[int, np.ndarray]:
    """
    Calculates weight importance based on absolute magnitude.

    Mathematical Context:
        Importance Score = |W|.
        This assumes that weights with small magnitudes contribute least 
        to the network's output.

    Args:
        model (Sequential): The neural network model.

    Returns:
        Dict[int, np.ndarray]: A mapping from layer index to a tensor of 
                               importance scores (same shape as weight matrix).
    """
    scores: Dict[int, np.ndarray] = {}
    for i, layer in enumerate(model.linear_layers()):
        scores[i] = np.abs(layer.W.data)
    return scores


def taylor_importance(
    model: Sequential,
    x: np.ndarray,
    y: np.ndarray,
    num_batches: int = 4,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
) -> Dict[int, np.ndarray]:
    """
    Calculates connection importance using first-order Taylor saliency.

    Mathematical Context:
        The saliency is an estimate of the increase in loss if a weight is 
        removed (set to zero). It is calculated as:
        Saliency = |W * (dL/dW)|.
        This incorporates both the magnitude of the weight and its 
        sensitivity (gradient), providing a more informed pruning metric 
        than magnitude alone.

    Args:
        model (Sequential): The neural network model.
        x (np.ndarray): Calibration input data.
        y (np.ndarray): Calibration labels.
        num_batches (int): Number of mini-batches to average gradients over 
                           to reduce variance.
        batch_size (int): Size of each calibration mini-batch.
        rng (np.random.Generator, optional): RNG for sampling calibration batches.

    Returns:
        Dict[int, np.ndarray]: A mapping from layer index to Taylor saliency scores.

    Role in Pipeline:
        Operates as a "calibration" phase. It triggers a set of backward passes 
        to capture gradients without executing optimizer steps, ensuring that 
        the importance scores reflect the current model state.
    """
    rng = rng or np.random.default_rng(0)
    layers: List[Linear] = model.linear_layers()
    accum = [np.zeros_like(layer.W.data) for layer in layers]
    n = x.shape[0]

    batches = 0
    for _ in range(num_batches):
        idx = rng.integers(0, n, size=min(batch_size, n))
        xb, yb = x[idx], y[idx]
        for p in model.parameters():
            p.zero_grad()
        logits = model(Tensor(xb, requires_grad=False))
        loss = softmax_cross_entropy(logits, yb)
        loss.backward()
        for j, layer in enumerate(layers):
            if layer.W.grad is not None:
                accum[j] += layer.W.grad
        batches += 1

    # Clear gradients so calibration does not contaminate the training step.
    for p in model.parameters():
        p.zero_grad()

    scores: Dict[int, np.ndarray] = {}
    for j, layer in enumerate(layers):
        mean_grad = accum[j] / max(batches, 1)
        scores[j] = np.abs(layer.W.data * mean_grad)
    return scores


def compute_importance(
    model: Sequential,
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    num_batches: int = 4,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
) -> Dict[int, np.ndarray]:
    """
    Dispatches the importance calculation to the specified method.

    Args:
        model (Sequential): The neural network model.
        method (str): The scoring method to use ("magnitude" or "taylor").
        x (np.ndarray): Calibration input data.
        y (np.ndarray): Calibration labels.
        num_batches (int): Number of batches for Taylor importance.
        batch_size (int): Batch size for Taylor importance.
        rng (np.random.Generator, optional): RNG for Taylor importance.

    Returns:
        Dict[int, np.ndarray]: The resulting importance scores.

    Raises:
        ValueError: If an unsupported method string is provided.
    """
    if method == "magnitude":
        return magnitude_importance(model)
    if method == "taylor":
        return taylor_importance(
            model, x, y, num_batches=num_batches, batch_size=batch_size, rng=rng
        )
    raise ValueError(f"unknown importance method {method!r}")
