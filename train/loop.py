"""
Core training orchestration logic for dense and pruned models.

Description:
    This module implements the shared training loop used by both baseline 
    and self-pruning experiments. It manages the interaction between the 
    autodiff engine, the Adam optimizer, and the pruning scheduler to 
    effectively minimize loss while simultaneously inducing sparsity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from engine.losses import softmax_cross_entropy
from engine.tensor import Tensor
from nn.layers import Sequential
from nn.optim import Adam
from prune.importance import compute_importance
from prune.mask import achieved_sparsity, prune_to_sparsity
from prune.schedule import cubic_sparsity
from train.dataset import SpiralData
from train.evaluate import accuracy, mean_loss


@dataclass
class PruneConfig:
    """
    Configuration parameters for the iterative pruning process.

    Args:
        method (str): Importance scoring strategy. Options: "magnitude" 
                     (weight absolute value) or "taylor" (first-order 
                     saliency based on gradient).
        final_sparsity (float): The target fraction of zeroed weights 
                               at the end of the pruning period (e.g., 0.90).
        start_epoch (int): Epoch at which pruning begins, allowing an initial 
                           period of dense "warm-up" training.
        end_epoch (int): Epoch at which pruning concludes.
        interval (int): Frequency (in epochs) of pruning events between 
                        start_epoch and end_epoch.
        calib_batches (int): Number of batches used to estimate Taylor saliency 
                             gradients during pruning events.
    """
    method: str = "magnitude"          # "magnitude" | "taylor"
    final_sparsity: float = 0.90
    start_epoch: int = 10
    end_epoch: int = 60
    interval: int = 5                   # epochs between pruning events
    calib_batches: int = 4


def _iterate_minibatches(
    x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
):
    """
    Generates shuffled mini-batches from the input dataset.

    Description:
        Permutes the dataset indices using the provided generator to ensure 
        stochasticity while maintaining determinism for reproducibility.

    Args:
        x (np.ndarray): Input feature matrix.
        y (np.ndarray): Target label vector.
        batch_size (int): Number of samples per batch.
        rng (np.random.Generator): NumPy random generator for shuffling.

    Returns:
        Generator yielding tuples of (batch_x, batch_y).

    Role in Pipeline:
        Data feeding for Stochastic Gradient Descent (SGD).
    """
    idx = rng.permutation(x.shape[0])      # deterministic given rng
    for start in range(0, x.shape[0], batch_size):
        sel = idx[start:start + batch_size]
        yield x[sel], y[sel]


def train_model(
    model: Sequential,
    data: SpiralData,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
    prune_cfg: Optional[PruneConfig] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Trains a model in-place using reverse-mode autodiff and optional pruning.

    Description:
        Orchestrates the training process. If `prune_cfg` is provided, it 
        integrates a pruning event into the loop based on a cubic schedule. 
        Each epoch consists of a full pass over the training data using 
        mini-batches, followed by a validation check.

    Mathematical Context:
        Iteratively updates parameters $\theta$ using the update rule 
        $\theta_{t+1} = \text{Adam}(\theta_t, \nabla_{\theta} \mathcal{L})$. 
        During pruning events, the mask $M$ is updated based on the 
        importance $I = f(\theta, \nabla_{\theta} \mathcal{L})$, and 
        weights are constrained such that $\theta \leftarrow \theta \odot M$.

    Args:
        model (Sequential): The NN architecture to train.
        data (SpiralData): The Spiral dataset containing train/val splits.
        epochs (int): Total number of training iterations.
        batch_size (int): Size of mini-batches.
        lr (float): Learning rate for the Adam optimizer.
        seed (int): Random seed for reproducibility.
        prune_cfg (Optional[PruneConfig]): Configuration for sparsity 
                                           induction. If None, training 
                                           remains dense.
        verbose (bool): Whether to print epoch statistics to the console.

    Returns:
        List[dict]: A history of metrics (loss, accuracy, sparsity) 
                   recorded at the end of each epoch.

    Role in Pipeline:
        The central engine for model optimization and sparsity induction.
    """
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr, masked_layers=model.linear_layers())
    history: List[dict] = []

    for epoch in range(epochs):
        # -- pruning event (before training this epoch, after warm-up) --
        if prune_cfg is not None and prune_cfg.start_epoch <= epoch <= prune_cfg.end_epoch:
            if (epoch - prune_cfg.start_epoch) % prune_cfg.interval == 0 or epoch == prune_cfg.end_epoch:
                target = cubic_sparsity(
                    epoch, prune_cfg.start_epoch, prune_cfg.end_epoch,
                    prune_cfg.final_sparsity,
                )
                importance = compute_importance(
                    model, prune_cfg.method, data.x_train, data.y_train,
                    num_batches=prune_cfg.calib_batches, batch_size=batch_size,
                    rng=np.random.default_rng(seed + 1000 + epoch),
                )
                prune_to_sparsity(model, optimizer, target, importance)

        epoch_losses: List[float] = []
        for xb, yb in _iterate_minibatches(data.x_train, data.y_train, batch_size, rng):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=False))
            loss = softmax_cross_entropy(logits, yb)
            loss_val = float(loss.data)
            if not np.isfinite(loss_val):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_val)

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": mean_loss(model, data.x_val, data.y_val),
            "val_acc": accuracy(model, data.x_val, data.y_val),
            "sparsity": achieved_sparsity(model),
        }
        history.append(rec)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch:3d} | train_loss {rec['train_loss']:.4f} | "
                f"val_loss {rec['val_loss']:.4f} | val_acc {rec['val_acc']:.4f} | "
                f"sparsity {rec['sparsity']:.3f}"
            )
    return history
