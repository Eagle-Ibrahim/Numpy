"""Synthetic 2-D spiral dataset implemented with NumPy only.

Generation is fully reproducible from a seed, and the train/validation/test
split is disjoint (no leakage): points are generated once, shuffled with a
seeded generator, and partitioned by index. Features are standardized using
**training-set statistics only**, so no test information leaks into scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpiralData:
    """
    Container for the split and normalized spiral dataset.

    Description:
        Encapsulates the train, validation, and test sets to ensure consistency 
        across the training pipeline and prevent data leakage.

    Attributes:
        x_train (np.ndarray): Normalized training features.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Normalized validation features.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Normalized test features.
        y_test (np.ndarray): Test labels.
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_features(self) -> int:
        """Returns the dimensionality of the input features."""
        return self.x_train.shape[1]

    @property
    def num_classes(self) -> int:
        """Returns the number of unique classes in the dataset."""
        return int(max(self.y_train.max(), self.y_val.max(), self.y_test.max())) + 1


def make_spiral(
    n_per_class: int = 400,
    n_classes: int = 2,
    noise: float = 0.10,
    revolutions: float = 1.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic 2-D spiral dataset.

    Mathematical Context:
        Each class is generated as a spiral arm where points are placed along 
        a linear radius $r \in [0.3, 1.0]$ and an angular component $\theta$.
        The angular component is: $\theta = \text{base} + \phi + \epsilon$, 
        where $\phi$ is the linear interpolation across the spiral span 
        ($2\pi \times \text{revolutions}$) and $\epsilon \sim \mathcal{N}(0, \text{noise})$.
        Coordinates are then mapped as: $x = r \sin(\theta), y = r \cos(\theta)$.

    Args:
        n_per_class (int): Number of samples to generate per spiral arm.
        n_classes (int): Number of distinct spiral arms (classes).
        noise (float): Standard deviation of Gaussian noise added to the angles.
        revolutions (float): The total angular span of each arm in turns.
        seed (int): RNG seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple (X, y) where X has shape 
                                       (n_per_class * n_classes, 2) and 
                                       y has shape (n_per_class * n_classes,).

    Role in Pipeline:
        Provides a non-linearly separable dataset that serves as a 
        benchmark for the MLP's ability to learn complex decision boundaries 
        and maintain accuracy under pruning.
    """
    rng = np.random.default_rng(seed)
    span = 2.0 * np.pi * revolutions
    xs = np.zeros((n_per_class * n_classes, 2), dtype=np.float64)
    ys = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        r = np.linspace(0.3, 1.0, n_per_class)
        base = c * 2.0 * np.pi / n_classes
        t = (
            np.linspace(base, base + span, n_per_class)
            + rng.normal(0.0, noise, n_per_class)
        )
        sl = slice(c * n_per_class, (c + 1) * n_per_class)
        xs[sl, 0] = r * np.sin(t)
        xs[sl, 1] = r * np.cos(t)
        ys[sl] = c
    return xs, ys


def load_spiral(
    n_per_class: int = 400,
    n_classes: int = 2,
    noise: float = 0.10,
    revolutions: float = 1.25,
    seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SpiralData:
    """
    Orchestrates the generation, splitting, and normalization of the spiral dataset.

    Description:
        This function creates the spiral data and partitions it into 
        training, validation, and test sets. Crucially, it implements 
        Z-score normalization based *strictly* on training set statistics 
        to avoid data leakage.

    Mathematical Context:
        Normalization is applied as: $x_{norm} = (x - \mu_{train}) / \sigma_{train}$, 
        where $\mu_{train}$ and $\sigma_{train}$ are the mean and standard 
        deviation of the training partition.

    Args:
        n_per_class (int): Samples per class.
        n_classes (int): Number of classes.
        noise (float): Noise level.
        revolutions (float): Number of spiral turns.
        seed (int): Seed for reproducibility.
        val_frac (float): Fraction of data reserved for validation.
        test_frac (float): Fraction of data reserved for testing.

    Returns:
        SpiralData: A data object containing normalized splits.

    Role in Pipeline:
        Prepares the data for training. Ensuring normalization is based 
        only on the training set is critical for a PhD-level evaluation 
        of the model's generalization capabilities.
    """
    x, y = make_spiral(n_per_class, n_classes, noise, revolutions, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(x.shape[0])
    x, y = x[perm], y[perm]

    n = x.shape[0]
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test

    x_train = x[:n_train]
    # Standardize with training statistics only (no leakage).
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0

    def norm(a: np.ndarray) -> np.ndarray:
        return (a - mu) / sd

    return SpiralData(
        x_train=norm(x_train), y_train=y[:n_train],
        x_val=norm(x[n_train:n_train + n_val]), y_val=y[n_train:n_train + n_val],
        x_test=norm(x[n_train + n_val:]), y_test=y[n_train + n_val:],
    )
