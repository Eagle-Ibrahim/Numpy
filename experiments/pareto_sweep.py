"""Sparsity-accuracy Pareto sweep.

Trains a dense baseline and both pruning methods across target sparsities over
several seeds, measuring achieved sparsity and test accuracy for each run. All
numbers are measured from executed training; nothing is fabricated.

Usage:
    python -m experiments.pareto_sweep
    python -m experiments.pareto_sweep --seeds 3 --epochs 120
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
from typing import List

import numpy as np

from nn.model import build_mlp, count_active_weights, count_eligible_weights
from prune.mask import achieved_sparsity
from prune.sparse_inference import SparseMLP, dense_macs
from train.artifacts import ensure_dir
from train.dataset import load_spiral
from train.evaluate import accuracy, mean_loss
from train.loop import PruneConfig, train_model

TARGET_SPARSITIES = [0.0, 0.50, 0.75, 0.90, 0.95]
METHODS = ["magnitude", "taylor"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparsity/accuracy Pareto sweep")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds per config")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sparsities", type=float, nargs="*", default=TARGET_SPARSITIES)
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def run_single(method: str, target: float, seed: int, args) -> dict:
    data = load_spiral(seed=seed)
    model = build_mlp(in_features=data.num_features, out_features=data.num_classes,
                      seed=seed)
    try:
        if target == 0.0:
            train_model(model, data, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, seed=seed, prune_cfg=None, verbose=False)
        else:
            cfg = PruneConfig(method=method, final_sparsity=target,
                              start_epoch=20, end_epoch=120,
                              interval=5, calib_batches=4)
            train_model(model, data, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, seed=seed, prune_cfg=cfg, verbose=False)
        sparse = SparseMLP(model)
        return {
            "method": "dense" if target == 0.0 else method,
            "seed": seed,
            "target_sparsity": target,
            "achieved_sparsity": achieved_sparsity(model),
            "active_params": count_active_weights(model),
            "eligible_params": count_eligible_weights(model),
            "val_acc": accuracy(model, data.x_val, data.y_val),
            "test_acc": accuracy(model, data.x_test, data.y_test),
            "final_loss": mean_loss(model, data.x_test, data.y_test),
            "dense_macs": dense_macs(model),
            "sparse_macs": sparse.active_macs(),
            "completed": True,
            "error": "",
        }
    except Exception as exc:  # preserve failure info rather than dropping it
        return {
            "method": "dense" if target == 0.0 else method, "seed": seed,
            "target_sparsity": target, "achieved_sparsity": float("nan"),
            "active_params": -1, "eligible_params": -1, "val_acc": float("nan"),
            "test_acc": float("nan"), "final_loss": float("nan"),
            "dense_macs": -1, "sparse_macs": -1,
            "completed": False, "error": repr(exc),
        }


def aggregate(rows: List[dict]) -> List[dict]:
    """Aggregate per-seed rows into mean/std by (method, target_sparsity)."""
    groups: dict = {}
    for r in rows:
        if not r["completed"]:
            continue
        groups.setdefault((r["method"], r["target_sparsity"]), []).append(r)
    out = []
    for (method, target), rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        test_accs = np.array([r["test_acc"] for r in rs])
        ach = np.array([r["achieved_sparsity"] for r in rs])
        out.append({
            "method": method,
            "target_sparsity": target,
            "n_seeds": len(rs),
            "mean_achieved_sparsity": float(ach.mean()),
            "mean_test_acc": float(test_accs.mean()),
            "std_test_acc": float(test_accs.std(ddof=1)) if len(rs) > 1 else 0.0,
            "mean_active_params": float(np.mean([r["active_params"] for r in rs])),
        })
    return out


def write_csv(rows: List[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_pareto(agg: List[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ["dense", "magnitude", "taylor"]:
        pts = [a for a in agg if a["method"] == method]
        if not pts:
            continue
        pts.sort(key=lambda a: a["mean_achieved_sparsity"])
        xs = [a["mean_achieved_sparsity"] for a in pts]
        ys = [a["mean_test_acc"] for a in pts]
        errs = [a["std_test_acc"] for a in pts]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=method)
    ax.set_xlabel("achieved weight sparsity")
    ax.set_ylabel("test accuracy")
    ax.set_title("Sparsity vs. accuracy (measured)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_rows: List[dict] = []
    for target in args.sparsities:
        methods = ["magnitude"] if target == 0.0 else METHODS
        for method in methods:
            for s in range(args.seeds):
                rec = run_single(method, target, seed=s, args=args)
                seed_rows.append(rec)
                print(f"{rec['method']:>9} target={target:.2f} seed={s} "
                      f"ach={rec['achieved_sparsity']:.3f} "
                      f"test_acc={rec['test_acc']:.4f} ok={rec['completed']}")

    agg = aggregate(seed_rows)
    write_csv(seed_rows, f"{args.out_dir}/raw/seeds.csv")
    write_csv(agg, f"{args.out_dir}/raw/pareto.csv")
    plot_pareto(agg, f"{args.out_dir}/plots/pareto_curve.png")

    meta = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seeds": args.seeds,
        "epochs": args.epochs,
    }
    with open(f"{args.out_dir}/raw/pareto_meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print("\nSaved results/raw/pareto.csv, seeds.csv and plots/pareto_curve.png")


if __name__ == "__main__":
    main()
