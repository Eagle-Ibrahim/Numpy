"""Binary mask management and the global pruning operation.

Pruning is **global** across all eligible weight connections: connections from
every Linear layer are pooled, ranked by importance, and the least important
ones are masked until the achieved active count matches the target implied by
the schedule. Pruning is permanent (monotone): a masked connection is never
revived in the baseline submission.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from nn.layers import Linear, Sequential
from nn.optim import Adam


def achieved_sparsity(model: Sequential) -> float:
    """
    Calculates the current global sparsity of the model.

    Args:
        model (Sequential): The neural network model.

    Returns:
        float: The fraction of eligible weight connections that are currently masked.
               Calculated as: (Total Eligible - Total Active) / Total Eligible.
    """
    total = sum(layer.num_eligible() for layer in model.linear_layers())
    active = sum(layer.num_active() for layer in model.linear_layers())
    if total == 0:
        return 0.0
    return 1.0 - active / total


def prune_to_sparsity(
    model: Sequential,
    optimizer: Adam,
    target_sparsity: float,
    importance: Dict[int, np.ndarray],
) -> float:
    """
    Performs global pruning to reach a specific target sparsity.

    Description:
        This function implements a "global" pruning strategy. Rather than pruning 
        each layer to a fixed percentage, it pools all active weights across the 
        entire network, ranks them by the provided importance scores, and 
        masks the absolute least important connections globally.

    Mathematical Context:
        1. Target Active Count = round((1 - target_sparsity) * Total_Eligible_Weights).
        2. Rank active weights by importance score. In case of ties, a 
           deterministic global position is used as a secondary key.
        3. Set mask[i] = 0 for the bottom (Current_Active - Target_Active) weights.

    Args:
        model (Sequential): The neural network model.
        optimizer (Adam): The optimizer used for training, required to reset 
                          momentums for pruned weights.
        target_sparsity (float): The desired fraction of weights to be masked.
        importance (Dict[int, np.ndarray]): A mapping of layer indices to 
                                            importance score tensors.

    Returns:
        float: The actual achieved sparsity after the pruning operation.

    Role in Pipeline:
        This is the primary actuators of the pruning process. It is called by the 
        training loop whenever the pruning schedule dictates a sparsity increase. 
        It ensures that "Momentum Resurrection" is prevented by resetting the 
        optimizer state for the newly pruned weights.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")

    layers: List[Linear] = model.linear_layers()
    total_eligible = sum(layer.num_eligible() for layer in layers)
    target_active = int(round((1.0 - target_sparsity) * total_eligible))
    current_active = sum(layer.num_active() for layer in layers)

    to_prune = current_active - target_active
    if to_prune <= 0:
        return achieved_sparsity(model)

    # Pool every active connection: (score, layer_index, flat_index).
    scores_list: List[np.ndarray] = []
    layer_ids: List[np.ndarray] = []
    flat_ids: List[np.ndarray] = []
    for li, layer in enumerate(layers):
        active_flat = np.flatnonzero(layer.mask.reshape(-1) > 0.5)
        if active_flat.size == 0:
            continue
        s = importance[li].reshape(-1)[active_flat]
        scores_list.append(s)
        layer_ids.append(np.full(active_flat.shape, li, dtype=np.int64))
        flat_ids.append(active_flat)

    all_scores = np.concatenate(scores_list)
    all_layers = np.concatenate(layer_ids)
    all_flats = np.concatenate(flat_ids)

    # Deterministic ranking: primary key = score, secondary = global position.
    global_pos = np.arange(all_scores.size)
    order = np.lexsort((global_pos, all_scores))  # ascending by score, then pos
    prune_sel = order[:to_prune]

    newly_pruned: Dict[int, np.ndarray] = {
        li: np.zeros(layers[li].mask.size, dtype=bool) for li in range(len(layers))
    }
    for k in prune_sel:
        li = int(all_layers[k])
        fi = int(all_flats[k])
        newly_pruned[li][fi] = True

    for li, layer in enumerate(layers):
        mask_flat = layer.mask.reshape(-1)
        np_mask = newly_pruned[li]
        if not np_mask.any():
            continue
        mask_flat[np_mask] = 0.0
        layer.mask = mask_flat.reshape(layer.mask.shape)
        optimizer.reset_masked_state(
            layer.W, np_mask.reshape(layer.W.data.shape)
        )
        layer.enforce_mask()

    return achieved_sparsity(model)
