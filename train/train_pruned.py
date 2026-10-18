"""
Entry point for training an MLP with iterative self-pruning.

Description:
    This script executes the full self-pruning pipeline: initializing the 
    Spiral dataset, building the MLP architecture, and running the 
    training loop with a dynamic sparsity schedule. It persists 
    metrics and weight masks for subsequent audit and sparse inference.

Usage:
    python -m train.train_pruned --target-sparsity 0.90
    python -m train.train_pruned --target-sparsity 0.90 --importance taylor
"""
from __future__ import annotations

import argparse

import numpy as np

from nn.model import (
    build_mlp,
    count_active_weights,
    count_eligible_weights,
)
from prune.mask import achieved_sparsity
from train.artifacts import plot_learning_curves, save_history_csv, save_json
from train.dataset import load_spiral
from train.evaluate import evaluate
from train.loop import PruneConfig, train_model


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the pruning experiment.

    Returns:
        argparse.Namespace: Config object containing hyperparameters 
                            and pruning settings.
    """
    p = argparse.ArgumentParser(description="Train self-pruning MLP")
    p.add_argument("--target-sparsity", type=float, default=0.90)
    p.add_argument("--importance", choices=["magnitude", "taylor"], default="magnitude")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--start-epoch", type=int, default=20)
    p.add_argument("--end-epoch", type=int, default=120)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    """
    Main execution flow for the self-pruning experiment.

    Description:
        1. Loads the Z-score normalized Spiral dataset.
        2. Constructs the MLP (Typically 2 -> 64 -> 64 -> 2).
        3. Configures the pruning scheduler (cubic decay).
        4. Executes the training loop with integrated pruning events.
        5. Evaluates final performance on the test set.
        6. Exports logs, plots, and the final weight masks.

    Role in Pipeline:
        The primary driver for verifying the hypothesis that Taylor saliency 
        or magnitude pruning allows for high sparsity without significant 
        accuracy degradation.
    """
    args = parse_args()
    data = load_spiral(seed=args.seed)
    model = build_mlp(in_features=data.num_features, hidden=args.hidden,
                      out_features=data.num_classes, seed=args.seed)

    prune_cfg = PruneConfig(
        method=args.importance,
        final_sparsity=args.target_sparsity,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        interval=args.interval,
        calib_batches=args.calib_batches,
    )
    history = train_model(
        model, data, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, prune_cfg=prune_cfg,
    )

    test_metrics = evaluate(model, data.x_test, data.y_test)
    final_sparsity = achieved_sparsity(model)
    active = count_active_weights(model)
    eligible = count_eligible_weights(model)

    tag = f"pruned_{args.importance}_s{int(round(args.target_sparsity * 100))}"
    save_history_csv(history, f"{args.out_dir}/raw/{tag}_history.csv")
    plot_learning_curves(history, f"{args.out_dir}/plots/{tag}_learning_curve.png",
                         title=f"Self-pruning MLP ({args.importance})")

    # Persist binary masks for reproducibility / sparse inference.
    masks = {f"layer_{i}": layer.mask for i, layer in enumerate(model.linear_layers())}
    np.savez(f"{args.out_dir}/raw/{tag}_masks.npz", **masks)

    save_json(
        {
            "config": vars(args),
            "target_sparsity": args.target_sparsity,
            "achieved_sparsity": final_sparsity,
            "active_weights": active,
            "eligible_weights": eligible,
            "test": test_metrics,
            "final_val_acc": history[-1]["val_acc"],
        },
        f"{args.out_dir}/raw/{tag}_summary.json",
    )
    print(f"\n[{args.importance}] target sparsity {args.target_sparsity:.2f} -> "
          f"achieved {final_sparsity:.4f} ({active}/{eligible} active)")
    print(f"test accuracy: {test_metrics['accuracy']:.4f} | "
          f"test loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()

