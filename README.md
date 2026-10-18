# AQUA Platform — The Self-Pruning Network

A neural network that **learns to prune its own connections during training**,
built from scratch on a custom reverse-mode autodiff engine using **NumPy only**
(plus Matplotlib for plots and pytest for tests). No PyTorch, TensorFlow, JAX,
Keras, autograd, or any ML/pruning library is used for the network, gradients,
optimizer, or pruning.

---

## 1. Executive summary

AQUA implements, end to end and with executed results:

- a reverse-mode automatic differentiation engine (`engine/`),
- a mask-aware MLP and a from-scratch Adam optimizer (`nn/`),
- a self-pruning system with magnitude and first-order Taylor-saliency criteria,
  a cubic sparsity schedule, exact sparsity accounting, and a sparse-aware
  inference path (`prune/`),
- reproducible training and experiments (`train/`, `experiments/`),
- a comprehensive pytest suite (`tests/`, **50 tests, all passing**).

All metrics below were produced by running the code; raw data lives in
`results/raw/` and plots in `results/plots/`.

## 2. Problem statement

Dense networks waste computation on redundant connections. The goal is to prune
most weights *during* training while preserving accuracy, keeping autodiff and
optimizer behavior mathematically correct, and demonstrating an **honest**
inference-cost measurement (not a dense matmul over zeros).

## 3. Goals and non-goals

**Goals:** correct custom autodiff; mask-aware gradients; Adam that cannot
resurrect pruned weights; measured sparsity from real masks; a sparse-aware
forward path; reproducible experiments.
**Non-goals:** production serving, GPU kernels, large-scale models, beating a
tuned framework. This is a correctness-first prototype.

## 4. System architecture

```
engine/  Tensor + ops + reverse-mode backward + stable losses + grad checks
nn/      Linear (mask-aware) / Tanh / ReLU / Sequential + from-scratch Adam
prune/   importance (magnitude, Taylor) + cubic schedule + masks + sparse infer
train/   spiral dataset + shared loop + baseline/pruned scripts + evaluation
experiments/  Pareto sweep + honest inference benchmark
tests/   50 pytest tests across autodiff, masks, pruning, sparse, reproducibility
```

## 5. Repository layout

```
engine/{tensor,ops,autodiff,losses}.py
nn/{layers,model,optim}.py
prune/{importance,schedule,mask,sparse_inference}.py
train/{dataset,loop,train_baseline,train_pruned,evaluate,artifacts}.py
experiments/{pareto_sweep,benchmark}.py
tests/test_*.py
configs/default.json
results/raw/*.csv  results/plots/*.png
README.md  DESIGN.md  pyproject.toml
```

## 6. Mathematical background

- Reverse-mode autodiff: topological order, seed `∂L/∂L = 1`, propagate in
  reverse, accumulate gradients from all paths.
- Broadcasting gradients are reduced with an `unbroadcast` helper.
- Stable softmax cross-entropy via log-sum-exp with closed-form gradient
  `(softmax − onehot)/N`.
- Taylor saliency `|w · ∂L/∂w|` derived in [DESIGN.md](DESIGN.md) §A.
- Masked gradient `∂L/∂W = M ⊙ ∂L/∂W_eff` derived in [DESIGN.md](DESIGN.md) §B.

## 7–11. Engine design, broadcasting, stable CE, grad checks, masked semantics

- **Autodiff** ([engine/tensor.py](engine/tensor.py),
  [engine/ops.py](engine/ops.py)): each op installs a `_backward` closure;
  `Tensor.backward()` builds the topo order and runs it in reverse. Non-scalar
  outputs require an explicit seed gradient. `requires_grad` is respected and
  gradients are cleared between steps to avoid stale reuse.
- **Broadcasting** (`unbroadcast`): sums gradients over prepended and
  size-1 axes. Tested for scalars, vectors, singleton dims, and the `(N,D)+(D,)`
  bias case ([tests/test_broadcasting.py](tests/test_broadcasting.py)).
- **Stable cross-entropy** ([engine/losses.py](engine/losses.py)): log-sum-exp;
  verified finite for logits of magnitude 1000.
- **Gradient checking** ([engine/autodiff.py](engine/autodiff.py)): central
  finite differences in float64 with documented `atol=1e-6`, `rtol=1e-4`,
  avoiding the ReLU kink at 0. Every differentiable op is checked.
- **Masked weights** ([nn/layers.py](nn/layers.py)): `W_eff = M ⊙ W`; masked
  entries get exactly zero ordinary gradient.

## 12–13. Adam and importance criteria

- **Adam** ([nn/optim.py](nn/optim.py)): first/second moments, bias correction,
  step counter, configurable `lr`/`eps`. On pruning it zeros the moments of the
  newly pruned weights and re-enforces the mask after every step so momentum
  cannot resurrect a pruned connection.
- **Importance** ([prune/importance.py](prune/importance.py)): magnitude `|W|`
  (baseline) and Taylor `|W · mean_grad|` averaged over calibration batches.
  Only active connections are scored.

## 14–16. Pruning schedule, training methodology, dataset/architecture

- **Schedule** ([prune/schedule.py](prune/schedule.py)): cubic
  `s(t) = s_f · (1 − (1 − t/T)^3)`, prunes early and tapers near the target.
- **Global pruning** ([prune/mask.py](prune/mask.py)): pools all eligible weight
  connections, ranks by importance with deterministic tie-breaking, masks the
  least important to hit the target active count (rounding documented), resets
  optimizer state, and verifies effective weights are zero.
- **Dataset** ([train/dataset.py](train/dataset.py)): synthetic 2-D spiral,
  2 classes, fixed seed, disjoint train/val/test split, standardized with
  **training statistics only** (no leakage).
- **Architecture**: `2 → 64 (tanh) → 64 (tanh) → 2 logits`, biases included,
  Xavier init (justified in DESIGN.md). Eligible weights = 4352; total params
  (with biases) = 4482.

## 17–18. Reproducibility

All randomness is seeded via explicit `numpy.random.Generator` objects. Identical
seeds reproduce the dataset split, model initialization, and short training runs
(enforced by [tests/test_reproducibility.py](tests/test_reproducibility.py)).

## 19. Installation

```bash
python3 -m pip install numpy matplotlib pytest
# or, from the repo root:
python3 -m pip install -e ".[test]"
```

## 20. Four main execution commands

```bash
python3 -m train.train_baseline                               # dense baseline
python3 -m train.train_pruned --target-sparsity 0.90          # self-pruning
python3 -m experiments.pareto_sweep                           # sparsity/accuracy sweep
python3 -m experiments.benchmark                              # honest inference cost
```

Self-pruning supports both criteria:

```bash
python3 -m train.train_pruned --target-sparsity 0.90 --importance magnitude
python3 -m train.train_pruned --target-sparsity 0.90 --importance taylor
```

## 21. Unit-test commands

```bash
python3 -m pytest -q
python3 -m pytest tests/test_gradient_check.py tests/test_ops.py tests/test_broadcasting.py -v
python3 -m pytest tests/test_masked_weights.py -v
```

Measured outcome on this machine: **50 passed** in ~0.3 s.

## 22. Pareto experiment methodology

Sweeps target sparsities `{0, 0.50, 0.75, 0.90, 0.95}` × `{dense, magnitude,
taylor}` × 5 seeds, with a shared train/val/test split per seed. Records seed,
method, target and **achieved** sparsity, active/eligible params, val/test
accuracy, final loss, MAC counts, and success flag. Per-seed rows →
`results/raw/seeds.csv`; aggregates → `results/raw/pareto.csv`; plot →
`results/plots/pareto_curve.png`. Failures are recorded, never silently dropped.

## 23. Inference-cost measurement methodology

Three paths are compared: dense forward, dense forward over zero-masked weights,
and a **sparse-aware** forward that stores only active incoming connections per
neuron (CSC) and multiplies only those ([prune/sparse_inference.py](prune/sparse_inference.py)).
Benchmark uses warm-up runs, repeated timed runs, and median latency; it reports
MAC counts and explicitly flags when indexing overhead outweighs the savings.

## 24–25. Raw results and generated plots

- `results/raw/`: `baseline_history.csv`, `baseline_summary.json`,
  `pruned_*_history.csv`, `pruned_*_masks.npz`, `pruned_*_summary.json`,
  `pareto.csv`, `seeds.csv`, `pareto_meta.json`, `inference_benchmark.csv`.
- `results/plots/`: `baseline_learning_curve.png`, `pruned_*_learning_curve.png`,
  `pareto_curve.png`, `inference_benchmark.png`.

## 26. Measured results

Environment: python3 3.10.12, NumPy 2.2.6, Linux x86_64.

**Baseline (seed 0):** test accuracy **1.0000**, test loss 0.0024.

**Sparsity vs. mean test accuracy** (5 seeds, from `results/raw/pareto.csv`):

| Method | Target | Achieved | Mean test acc | Std |
|---|---|---|---|---|
| dense | 0.00 | 0.000 | 0.918 | 0.173 |
| magnitude | 0.50 | 0.500 | 0.915 | 0.181 |
| magnitude | 0.75 | 0.750 | 0.953 | 0.091 |
| magnitude | 0.90 | 0.900 | 0.928 | 0.142 |
| magnitude | 0.95 | 0.950 | 0.863 | 0.172 |
| taylor | 0.50 | 0.500 | 0.913 | 0.180 |
| taylor | 0.75 | 0.750 | 0.915 | 0.181 |
| taylor | 0.90 | 0.900 | 0.915 | 0.181 |
| taylor | 0.95 | 0.950 | 0.913 | 0.180 |

The large standard deviations are driven almost entirely by **one seed (seed 1)**
on which even the *dense* baseline collapses to 0.608 — a bad initialization /
data draw, not a pruning artifact. This is reported honestly rather than removed.

**Inference benchmark** (95% achieved sparsity, batch 256, 20 warm-up / 200 timed
runs, from `results/raw/inference_benchmark.csv`):

| Metric | Value |
|---|---|
| Dense MACs / sample | 4352 |
| Sparse MACs / sample | 218 |
| MAC reduction | **19.96×** |
| Dense median latency | 0.614 ms |
| Sparse median latency | 0.975 ms |
| Latency speedup | **0.63× (sparse is slower)** |

**Honest conclusion:** pruning removes ~95% of MACs, but for this tiny model the
python3 per-neuron indexing overhead of the sparse path **outweighs** the saved
multiplies, so it runs *slower* than dense NumPy. Genuine latency wins would
require much larger models or a vectorized/block-sparse kernel. We do **not**
claim a speedup, because the measurement does not support one.

## 27. Quantitative claim

> At approximately **95% achieved weight sparsity**, Taylor-saliency pruning
> produces a mean test accuracy of **91.3%** versus **86.3%** for magnitude
> pruning over **N = 5** fixed seeds (seeds 0–4), a difference of **+5.0
> percentage points** in favor of Taylor.

- **Architecture:** 2-64-64-2 MLP, tanh, biases, Xavier init.
- **Dataset:** synthetic 2-class spiral, standardized; evaluation on the held-out
  **test** split.
- **Target / achieved sparsity:** 0.95 target, 0.9499 achieved (218/4352 active
  weights).
- **Variability:** high. Per-seed test accuracy at 95% — magnitude:
  `{0.975, 0.667, 1.000, 0.683, 0.992}`; taylor: `{0.992, 0.592, 1.000, 0.983,
  1.000}`. The mean paired difference (taylor − magnitude) is +5.0 pp, but the
  standard deviation (~0.17) exceeds the gap, so the evidence is **suggestive,
  not conclusive**. Seed 1 is pathological for both methods (dense baseline also
  fails there).
- **Supporting data:** `results/raw/seeds.csv` and `results/raw/pareto.csv`.

Caveat: with only 5 seeds and one outlier, this difference is not statistically
robust. Taylor is *at least as good as* magnitude here and degrades more
gracefully at the highest sparsity, which is consistent with the theory, but a
larger seed budget would be needed to claim superiority confidently.

## 28. Known limitations

- Small model/dataset; results have high seed variance (one pathological seed).
- Sparse inference is slower than dense for this model size (reported honestly).
- Backward-pass bottlenecks are anticipated, not profiled (DESIGN.md §C).
- Permanent pruning only; regrowth semantics are documented but not implemented.
- `matmul` supports 2-D @ 2-D only (explicit error otherwise).

## 29. Design trade-offs

Explicit python3 closures and per-op NumPy calls favor clarity and correctness
over speed. Global pruning favors simplicity over per-layer sparsity control.
Permanent pruning favors a defensible baseline over regrowth complexity.

## 30. Production-serving considerations

See [DESIGN.md](DESIGN.md) §D. The prototype is **not** production-ready.

## 31. Permitted libraries

Runtime: **NumPy** (all numerics, by hand), **Matplotlib** (plots only).
Dev: **pytest** (tests). Standard library otherwise. No PyTorch/TensorFlow/JAX/
Keras/autograd/scikit-learn/ML/pruning libraries are used for the network,
gradients, optimizer, or pruning. The dataset is synthetic NumPy (no external
dataset dependency).
