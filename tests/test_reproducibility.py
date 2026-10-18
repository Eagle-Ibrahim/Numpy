"""Reproducibility: identical seeds reproduce data, init, and short runs."""
import numpy as np

from nn.model import build_mlp
from train.dataset import load_spiral
from train.loop import train_model


def test_dataset_reproducible():
    d1 = load_spiral(seed=7)
    d2 = load_spiral(seed=7)
    assert np.array_equal(d1.x_train, d2.x_train)
    assert np.array_equal(d1.y_train, d2.y_train)
    assert np.array_equal(d1.x_test, d2.x_test)


def test_dataset_splits_disjoint():
    d = load_spiral(seed=1)
    train = {tuple(r) for r in d.x_train}
    val = {tuple(r) for r in d.x_val}
    test = {tuple(r) for r in d.x_test}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_model_init_reproducible():
    m1 = build_mlp(seed=3)
    m2 = build_mlp(seed=3)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert np.array_equal(p1.data, p2.data)


def test_short_training_run_reproducible():
    results = []
    for _ in range(2):
        data = load_spiral(seed=0)
        model = build_mlp(seed=0)
        hist = train_model(model, data, epochs=3, batch_size=32, lr=1e-3,
                           seed=0, verbose=False)
        results.append(hist[-1]["train_loss"])
    assert np.isclose(results[0], results[1])


def test_history_schema_stable():
    data = load_spiral(seed=0)
    model = build_mlp(seed=0)
    hist = train_model(model, data, epochs=2, batch_size=32, lr=1e-3,
                       seed=0, verbose=False)
    assert set(hist[0].keys()) == {"epoch", "train_loss", "val_loss", "val_acc", "sparsity"}
