"""Loss functions with analytically correct, numerically stable gradients."""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, as_tensor


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Computes the numerically stable log-softmax along the last axis.

    Mathematical Context:
        log_softmax(x)_i = x_i - log(∑ exp(x_j))
        To prevent overflow, we use the Log-Sum-Exp trick:
        log(∑ exp(x_j)) = max(x) + log(∑ exp(x_j - max(x)))

    Args:
        logits (np.ndarray): The input unnormalized scores (logits).

    Returns:
        np.ndarray: The log-probabilities.

    Role in Pipeline:
        Internal helper for cross-entropy loss to ensure numerical stability.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_sum_exp


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Computes the numerically stable softmax along the last axis.

    Mathematical Context:
        softmax(x)_i = exp(x_i) / ∑ exp(x_j)
        The subtraction of max(x) prevents exponential overflow without changing the result.

    Args:
        logits (np.ndarray): The input unnormalized scores (logits).

    Returns:
        np.ndarray: The normalized probability distribution.

    Role in Pipeline:
        Used for final predictions and calculating probabilities in the loss function.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Computes the mean softmax cross-entropy loss over a mini-batch.

    Mathematical Context:
        Loss = (-1/N) ∑ log(softmax(logits)_{target})
        The gradient with respect to the logits is the stable closed-form:
        ∇L = (softmax(logits) - one_hot(targets)) / N.

    Args:
        logits (Tensor): Tensor of shape (N, C) containing unnormalized scores.
        targets (np.ndarray): Integer class labels of shape (N,).

    Returns:
        Tensor: A scalar Tensor holding the mean cross-entropy loss.

    Raises:
        ValueError: If logits are not 2-D or if targets shape is incompatible.

    Role in Pipeline:
        The objective function for the supervised classification task.
    """
    logits = as_tensor(logits)
    z = logits.data
    if z.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C); got shape {z.shape}")
    targets = np.asarray(targets).astype(np.int64)
    n, c = z.shape
    if targets.shape != (n,):
        raise ValueError(
            f"targets shape {targets.shape} incompatible with logits {z.shape}"
        )

    logp = log_softmax(z)                       # (N, C)
    nll = -logp[np.arange(n), targets]          # (N,)
    loss_value = float(nll.mean())

    out = Tensor(np.array(loss_value), logits.requires_grad, (logits,),
                 "softmax_cross_entropy")

    probs = np.exp(logp)                         # == softmax(z), stable

    def _backward() -> None:
        if out.grad is None:
            return
        one_hot = np.zeros_like(z)
        one_hot[np.arange(n), targets] = 1.0
        grad = (probs - one_hot) / n            # dL/dlogits
        logits._accumulate(grad * out.grad)

    out._backward = _backward
    return out
