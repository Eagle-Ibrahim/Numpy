"""Gradual pruning schedule.

Implements the cubic sparsity schedule of Zhu & Gupta (2017):

    s(t) = s_final * (1 - (1 - (t - t0) / (T - t0))^3)

which prunes aggressively early and tapers off near the target, giving the
network time to recover between the final, most damaging pruning events.
"""
from __future__ import annotations


def cubic_sparsity(
    step: int,
    start_step: int,
    end_step: int,
    final_sparsity: float,
    initial_sparsity: float = 0.0,
) -> float:
    """
    Calculates the target sparsity for a given training step using a cubic schedule.

    Mathematical Context:
        Implements the sparsity schedule from Zhu & Gupta (2017). 
        The sparsity s(t) evolves as:
        s(t) = s_final + (s_initial - s_final) * (1 - (t - t_0) / (T - t_0))^3
        where:
        - t: current step
        - t_0: start_step
        - T: end_step
        - s_final: final_sparsity
        - s_initial: initial_sparsity

        This cubic curve prunes aggressively in the early stages and tapers off 
        as it approaches the final sparsity target, allowing the model to 
        better recover and stabilize.

    Args:
        step (int): The current training iteration.
        start_step (int): The step at which pruning begins.
        end_step (int): The step at which pruning reaches the final target.
        final_sparsity (float): The target sparsity (fraction of weights to remove).
        initial_sparsity (float): The starting sparsity. Defaults to 0.0.

    Returns:
        float: The target sparsity value for the current step.

    Raises:
        ValueError: If final_sparsity is not in [0, 1) or if end_step <= start_step.

    Role in Pipeline:
        Provides the target sparsity percentage to the pruning orchestrator 
        at each training epoch, which is then used to determine how many 
        weights should be masked.
    """
    if not 0.0 <= final_sparsity < 1.0:
        raise ValueError("final_sparsity must be in [0, 1)")
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    if step <= start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    progress = (step - start_step) / (end_step - start_step)
    return (
        final_sparsity
        + (initial_sparsity - final_sparsity) * (1.0 - progress) ** 3
    )
