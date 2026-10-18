"""From-scratch Adam optimizer with mask-aware state handling.

Beyond the standard Adam update this optimizer knows how to **reset the moment
estimates of newly pruned weights** and **re-enforce binary masks after each
step**, so that momentum accumulated before pruning can never resurrect a
pruned connection.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear


class Adam:
    """
    Adaptive Moment Estimation (Adam) optimizer with mask-aware state management.

    Mathematical Context:
        Adam computes individual adaptive learning rates for different parameters from 
        estimates of first and second moments of the gradients.
        Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Pruning Integration:
        Standard Adam can "resurrect" pruned weights because the momentum buffers (m, v) 
        persist even after a weight is masked. This implementation prevents this by:
        1. Explicitly calling `enforce_mask()` after each update.
        2. Providing `reset_masked_state()` to zero out momentum for newly pruned connections.
    """
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        masked_layers: Optional[List[Linear]] = None,
    ) -> None:
        """
        Initializes the Adam optimizer.

        Args:
            params (List[Tensor]): List of tensors to optimize.
            lr (float): Learning rate.
            betas (tuple[float, float]): Coefficients for computing running averages of 
                                        gradient and its square.
            eps (float): Small constant for numerical stability to avoid division by zero.
            masked_layers (Optional[List[Linear]]): List of Linear layers whose masks 
                                                   must be enforced after each step.
        """
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.params = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.t = 0
        # Per-parameter first/second moments keyed by object id.
        self.m: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data) for p in self.params
        }
        self.v: Dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data) for p in self.params
        }
        self.masked_layers = list(masked_layers) if masked_layers else []

    def zero_grad(self) -> None:
        """
        Resets gradients of all optimized parameters to zero.

        Role in Pipeline:
            Must be called before the backward pass in every training iteration to 
            prevent gradient accumulation.
        """
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).

        Mathematical Context:
            Executes the Adam update rule for each parameter that has a gradient.
            After updating weights, it iterates through `masked_layers` to ensure 
            pruned weights are strictly zero, counteracting potential momentum drift.

        Role in Pipeline:
            Called after the backward pass to update the weights based on 
            computed gradients.
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            key = id(p)
            m = self.m[key]
            v = self.v[key]
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Re-enforce masks so masked weights stay exactly zero.
        for layer in self.masked_layers:
            layer.enforce_mask()

    def reset_masked_state(self, param: Tensor, newly_pruned: np.ndarray) -> None:
        """
        Zeroes the Adam momentum buffers for connections that were just pruned.

        Args:
            param (Tensor): The weight tensor being pruned.
            newly_pruned (np.ndarray): Boolean mask (same shape as param) where True 
                                       indicates a connection that has just been pruned.

        Role in Pipeline:
            Critical for preventing "Momentum Resurrection". By zeroing buffers, 
            we ensure that the optimizer does not use historical gradients to 
            push a pruned weight back into existence.
        """
        key = id(param)
        if key not in self.m:
            raise KeyError("parameter is not tracked by this optimizer")
        self.m[key][newly_pruned] = 0.0
        self.v[key][newly_pruned] = 0.0

    def reset_param_state(self, param: Tensor) -> None:
        """
        Fully resets the Adam state (m and v) for a specific parameter.

        Args:
            param (Tensor): The target parameter tensor.

        Role in Pipeline:
            Used in scenarios where a parameter is structurally reset or revised, 
            requiring a clean slate for adaptive learning rates.
        """
        key = id(param)
        self.m[key] = np.zeros_like(param.data)
        self.v[key] = np.zeros_like(param.data)
