"""Training scripts, dataset, evaluation, and the shared training loop."""
from train.dataset import SpiralData, load_spiral, make_spiral
from train.evaluate import accuracy, evaluate, mean_loss
from train.loop import PruneConfig, train_model

__all__ = [
    "SpiralData",
    "load_spiral",
    "make_spiral",
    "accuracy",
    "evaluate",
    "mean_loss",
    "PruneConfig",
    "train_model",
]
