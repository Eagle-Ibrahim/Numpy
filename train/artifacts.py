"""
Persistence and visualization utilities for training artifacts.

Description:
    Provides a suite of helper functions to export training metrics to 
    structured formats (CSV, JSON) and generate performance visualizations 
    (learning curves). This ensures that experiments are reproducible 
    and results can be audited independently of the runtime environment.
"""
from __future__ import annotations

import csv
import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(path: str) -> None:
    """
    Recursively creates the directory path if it does not already exist.

    Args:
        path (str): The absolute or relative path to the directory.

    Role in Pipeline:
        Prevents I/O errors when saving artifacts to new subdirectories.
    """
    os.makedirs(path, exist_ok=True)


def save_history_csv(history: List[dict], path: str) -> None:
    """
    Writes the training history log to a CSV file.

    Description:
        Converts a list of per-epoch metric dictionaries into a tabular 
        CSV format. This is the primary method for persisting longitudinal 
        training data for external statistical analysis.

    Args:
        history (List[dict]): A list of dictionaries where each entry 
                              represents a training epoch.
        path (str): The destination file path.

    Role in Pipeline:
        Artifact persistence for auditability and PhD-level review.
    """
    if not history:
        return
    ensure_dir(os.path.dirname(path) or ".")
    fields = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_json(obj: dict, path: str) -> None:
    """
    Serializes a Python object to a JSON file.

    Description:
        Saves configuration, final model hyper-parameters, or summary 
        statistics to a JSON file with indentation for readability.

    Args:
        obj (dict): The dictionary to serialize.
        path (str): The destination file path.

    Role in Pipeline:
        Persistence of static experiment metadata.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curves(history: List[dict], path: str, title: str) -> None:
    """
    Generates an epoch-wise visualization of model performance.

    Description:
        Creates a dual-axis plot. The left Y-axis tracks the Cross-Entropy 
        Loss (training vs. validation), while the right Y-axis tracks 
        Accuracy and, if applicable, the Network Sparsity.

    Mathematical Context:
        Tracks the convergence of the loss function $\mathcal{L}$ and the 
        asymptotic behavior of accuracy $\text{Acc} \in [0, 1]$. 
        Sparsity is plotted as a percentage of zeroed weights over time.

    Args:
        history (List[dict]): Training logs containing 'epoch', 'train_loss', 
                              'val_loss', 'val_acc', and optional 'sparsity'.
        path (str): Path to save the resulting PNG image.
        title (str): The title of the generated plot.

    Role in Pipeline:
        Critical for diagnosing overfitting, convergence speed, and the 
        impact of the pruning schedule on model stability.
    """
    ensure_dir(os.path.dirname(path) or ".")
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss", color="tab:blue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val loss", color="tab:cyan")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val acc", color="tab:red")
    if "sparsity" in history[0]:
        ax2.plot(epochs, [h["sparsity"] for h in history], label="sparsity",
                 color="tab:green", linestyle="--")
    ax2.set_ylabel("accuracy / sparsity")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="upper right")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
