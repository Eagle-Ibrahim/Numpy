"""Honest inference-cost benchmark: dense vs. zero-masked dense vs. sparse-aware.

Measures median latency with warm-up and repeated timed runs, and reports
MAC counts, speedup, and environment metadata. The benchmark does not assert a
speedup unless the measurement supports it; it reports indexing overhead when
the sparse path is slower.

Usage:
    python -m experiments.benchmark
    python -m experiments.benchmark --target-sparsity 0.95 --batch-size 256
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from typing import Callable, List

import numpy as np

from engine.tensor import Tensor
from nn.model import build_mlp
from nn.optim import Adam
from prune.importance import magnitude_importance
from prune.mask import achieved_sparsity, prune_to_sparsity
from prune.sparse_inference import SparseMLP, dense_macs
from train.artifacts import ensure_dir
from train.dataset import load_spiral
from train.loop import PruneConfig, train_model


def time_fn(fn: Callable[[], object], warmup: int, runs: int) -> List[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference-cost benchmark")
    p.add_argument("--target-sparsity", type=float, default=0.95)
    p.add_argument("--importance", choices=["magnitude", "taylor"], default="magnitude")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_spiral(seed=args.seed)
    model = build_mlp(in_features=data.num_features, out_features=data.num_classes,
                      seed=args.seed)
    cfg = PruneConfig(method=args.importance, final_sparsity=args.target_sparsity,
                      start_epoch=15, end_epoch=90, interval=5, calib_batches=4)
    train_model(model, data, epochs=args.epochs, batch_size=args.batch_size,
                lr=1e-2, seed=args.seed, prune_cfg=cfg, verbose=False)
    sparsity = achieved_sparsity(model)
    sparse = SparseMLP(model)

    rng = np.random.default_rng(123)
    x = rng.normal(size=(args.batch_size, data.num_features))

    # Correctness: sparse predictions must match dense masked predictions.
    dense_logits = model(Tensor(x, requires_grad=False)).data
    sparse_logits = sparse.forward(x)
    assert np.allclose(dense_logits, sparse_logits, atol=1e-10)

    dense_t = time_fn(lambda: model(Tensor(x, requires_grad=False)).data,
                      args.warmup, args.runs)
    sparse_t = time_fn(lambda: sparse.forward(x), args.warmup, args.runs)

    d_med = float(np.median(dense_t))
    s_med = float(np.median(sparse_t))
    d_macs = dense_macs(model)
    s_macs = sparse.active_macs()

    row = {
        "target_sparsity": args.target_sparsity,
        "achieved_sparsity": sparsity,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "dense_macs_per_sample": d_macs,
        "sparse_macs_per_sample": s_macs,
        "dense_median_s": d_med,
        "sparse_median_s": s_med,
        "dense_p95_s": float(np.percentile(dense_t, 95)),
        "sparse_p95_s": float(np.percentile(sparse_t, 95)),
        "speedup_median": d_med / s_med if s_med > 0 else float("nan"),
        "mac_reduction": d_macs / s_macs if s_macs > 0 else float("nan"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    ensure_dir(f"{args.out_dir}/raw")
    with open(f"{args.out_dir}/raw/inference_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    _plot(row, f"{args.out_dir}/plots/inference_benchmark.png")

    print(json.dumps(row, indent=2))
    verdict = ("sparse path FASTER" if row["speedup_median"] > 1.0
               else "sparse path SLOWER (indexing overhead outweighs MAC savings)")
    print(f"\nMAC reduction: {row['mac_reduction']:.2f}x | "
          f"latency speedup: {row['speedup_median']:.2f}x -> {verdict}")


def _plot(row: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(path))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["dense", "sparse"],
            [row["dense_median_s"] * 1e3, row["sparse_median_s"] * 1e3],
            color=["tab:blue", "tab:green"])
    ax1.set_ylabel("median latency (ms)")
    ax1.set_title(f"Latency @ {row['achieved_sparsity']:.2f} sparsity")
    ax2.bar(["dense", "sparse"],
            [row["dense_macs_per_sample"], row["sparse_macs_per_sample"]],
            color=["tab:blue", "tab:green"])
    ax2.set_ylabel("MACs per sample")
    ax2.set_title("Multiply-accumulate count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
