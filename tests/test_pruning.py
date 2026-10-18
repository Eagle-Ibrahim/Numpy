"""Pruning: schedule, exact sparsity accounting, determinism, and persistence."""
import numpy as np

from nn.model import build_mlp, count_eligible_weights
from nn.optim import Adam
from prune.importance import compute_importance, magnitude_importance
from prune.mask import achieved_sparsity, prune_to_sparsity
from prune.schedule import cubic_sparsity


def test_cubic_schedule_endpoints_and_monotonicity():
    assert cubic_sparsity(0, 0, 10, 0.9) == 0.0
    assert np.isclose(cubic_sparsity(10, 0, 10, 0.9), 0.9)
    vals = [cubic_sparsity(t, 0, 10, 0.9) for t in range(11)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))  # non-decreasing


def test_target_sparsity_achieved_with_rounding():
    model = build_mlp(seed=0)
    opt = Adam(model.parameters(), masked_layers=model.linear_layers())
    eligible = count_eligible_weights(model)
    achieved = prune_to_sparsity(model, opt, 0.90, magnitude_importance(model))
    expected_active = round(0.10 * eligible)
    active = sum(l.num_active() for l in model.linear_layers())
    assert active == expected_active
    assert np.isclose(achieved, 1.0 - expected_active / eligible)


def test_permanent_pruning_is_monotone():
    model = build_mlp(seed=0)
    opt = Adam(model.parameters(), masked_layers=model.linear_layers())
    prune_to_sparsity(model, opt, 0.5, magnitude_importance(model))
    masks_50 = [l.mask.copy() for l in model.linear_layers()]
    prune_to_sparsity(model, opt, 0.75, magnitude_importance(model))
    for layer, m50 in zip(model.linear_layers(), masks_50):
        # Everything pruned at 50% stays pruned at 75%.
        assert np.all(layer.mask[m50 < 0.5] == 0.0)


def test_importance_scores_finite():
    model = build_mlp(seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 2))
    y = rng.integers(0, 2, size=64)
    for method in ("magnitude", "taylor"):
        scores = compute_importance(model, method, x, y, rng=rng)
        for s in scores.values():
            assert np.all(np.isfinite(s))


def test_pruning_deterministic_given_same_scores():
    m1 = build_mlp(seed=0)
    m2 = build_mlp(seed=0)
    o1 = Adam(m1.parameters(), masked_layers=m1.linear_layers())
    o2 = Adam(m2.parameters(), masked_layers=m2.linear_layers())
    prune_to_sparsity(m1, o1, 0.8, magnitude_importance(m1))
    prune_to_sparsity(m2, o2, 0.8, magnitude_importance(m2))
    for l1, l2 in zip(m1.linear_layers(), m2.linear_layers()):
        assert np.array_equal(l1.mask, l2.mask)


def test_repeated_pruning_events_do_not_corrupt_masks():
    model = build_mlp(seed=0)
    opt = Adam(model.parameters(), masked_layers=model.linear_layers())
    for target in (0.3, 0.5, 0.7, 0.9):
        prune_to_sparsity(model, opt, target, magnitude_importance(model))
        # masks are strictly binary
        for layer in model.linear_layers():
            assert set(np.unique(layer.mask)).issubset({0.0, 1.0})
    assert np.isclose(achieved_sparsity(model),
                      1.0 - round(0.1 * count_eligible_weights(model))
                      / count_eligible_weights(model))
