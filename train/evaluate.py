"""
Evaluation utilities for assessing model performance.

Description:
    Provides metric computation functions to evaluate a model's predictive 
    power and loss convergence on independent datasets. These utilities 
    act as the final a-posteriori check for both dense and pruned models.
"""
from __future__ import annotations

import numpy as np

from engine.losses import softmax_cross_entropy
from engine.tensor import Tensor
from nn.layers import Sequential


def accuracy(model: Sequential, x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the classification accuracy on a given dataset.

    Description:
        Passes the input features through the model in inference mode, 
        extracts the class with the highest logit (argmax), and calculates 
        the fraction of correct predictions.

    Args:
        model (Sequential): The model to evaluate.
        x (np.ndarray): Input features of shape `(N, dim)`.
        y (np.ndarray): Ground truth integer labels of shape `(N,)`.

    Returns:
        float: The accuracy score in the range $[0, 1]$.

    Role in Pipeline:
        Primary performance metric used to determine if pruning has 
        degraded the model's ability to generalize.
    """
    logits = model(Tensor(x, requires_grad=False)).data
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def mean_loss(model: Sequential, x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the mean softmax cross-entropy loss across a dataset.

    Description:
        Evaluates the average divergence between the model's predicted 
        probability distribution and the one-hot encoded ground truth labels.

    Mathematical Context:
        Calculates $\frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f(x_i), y_i)$, 
        where $\mathcal{L}$ is the numerically stable softmax cross-entropy.

    Args:
        model (Sequential): The model to evaluate.
        x (np.ndarray): Input features.
        y (np.ndarray): Ground truth labels.

    Returns:
        float: The average cross-entropy loss.

    Role in Pipeline:
        Used to monitor the model's confidence and ensure that the loss 
        remains stable during the pruning-recovery phases.
    """
    logits = model(Tensor(x, requires_grad=False))
    return float(softmax_cross_entropy(logits, y).data)


def evaluate(model: Sequential, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Performs a full evaluation pass returning both loss and accuracy.

    Args:
        model (Sequential): The model to evaluate.
        x (np.ndarray): Input features.
        y (np.ndarray): Ground truth labels.

    Returns:
        dict: A dictionary containing {"loss": float, "accuracy": float}.

    Role in Pipeline:
        Convenience wrapper used in the training loop to compute validation 
        metrics at the end of each epoch.
    """
    return {"loss": mean_loss(model, x, y), "accuracy": accuracy(model, x, y)}
