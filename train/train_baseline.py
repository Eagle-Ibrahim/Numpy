"""
Entry point for training the dense baseline MLP.

Description:
    This script establishes the performance upper bound by training a 
    dense MLP (no pruning) on the Spiral dataset. It serves as the 
    control group to compare against self-pruning configurations.

Usage:
    python -m train.train_baseline
    python -m train.train_baseline --epochs 150 --lr 1e-3 --seed 0
"""
from __future__ import annotations

import argparse

from nn.model import build_mlp, count_parameters
from train.artifacts import plot_learning_curves, save_history_csv, save_json
from train.dataset import load_spiral
from train.evaluate import evaluate
from train.loop import train_model


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the baseline experiment.

    Returns:
        argparse.Namespace: Config object containing hyperparameters.
    """
    p = argparse.ArgumentParser(description="Train dense baseline MLP")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    """
    Main execution flow for the baseline experiment.

    Description:
        1. Loads the Z-score normalized Spiral dataset.
        2. Constructs the dense MLP architecture.
        3. Executes the training loop with `prune_cfg=None`.
        4. Evaluates final accuracy and loss on the test set.
        5. Exports metrics and learning curves for comparative analysis.

    Role in Pipeline:
        Provides the gold-standard accuracy metric used to determine 
        the 'accuracy drop' acceptable during aggressive pruning.
    """
    args = parse_args()
    data = load_spiral(seed=args.seed)
    model = build_mlp(in_features=data.num_features, hidden=args.hidden,
                      out_features=data.num_classes, seed=args.seed)

    history = train_model(
        model, data, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, prune_cfg=None,
    )
    test_metrics = evaluate(model, data.x_test, data.y_test)

    save_history_csv(history, f"{args.out_dir}/raw/baseline_history.csv")
    plot_learning_curves(history, f"{args.out_dir}/plots/baseline_learning_curve.png",
                         title="Baseline MLP learning curves")
    save_json(
        {
            "config": vars(args),
            "params": count_parameters(model),
            "test": test_metrics,
            "final_val_acc": history[-1]["val_acc"],
        },
        f"{args.out_dir}/raw/baseline_summary.json",
    )
    print(f"\nBaseline test accuracy: {test_metrics['accuracy']:.4f} "
          f"| test loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()

