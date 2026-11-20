"""Conditional flow-matching policy over an action chunk.

The policy learns a velocity field v_theta(x_tau, tau, s) that transports Gaussian noise
x_0 ~ N(0, I) to the expert chunk x_1 along the straight path x_tau = (1 - tau) x_0 + tau x_1
(rectified flow / conditional flow matching).  Sampling integrates the field with a fixed
number of Euler steps; the integration is ordinary PyTorch, so gradients of a loss on the
sampled chunk reach the policy parameters through every Euler step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class FlowChunkPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        time_frequencies: int = 4,
        sample_steps: int = 8,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, horizon) <= 0:
            raise ValueError("observation_dim, action_dim, and horizon must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if time_frequencies < 0 or sample_steps <= 0:
            raise ValueError("time_frequencies must be >= 0 and sample_steps > 0")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.flat_dim = horizon * action_dim
        self.time_frequencies = time_frequencies
        self.sample_steps = sample_steps

        layers: list[nn.Module] = []
        input_dim = self.flat_dim + observation_dim + 1 + 2 * time_frequencies
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, self.flat_dim))
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def time_features(self, tau: Tensor) -> Tensor:
        """[batch] -> [batch, 1 + 2 F]: tau and sinusoids of tau."""
        tau = tau.reshape(-1, 1)
        if self.time_frequencies == 0:
            return tau
        k = torch.arange(1, self.time_frequencies + 1, device=tau.device, dtype=tau.dtype) * (2 * math.pi)
        angles = tau * k
        return torch.cat([tau, torch.sin(angles), torch.cos(angles)], dim=1)

    def velocity(self, x: Tensor, tau: Tensor, observation: Tensor) -> Tensor:
        """Velocity field at (x_tau, tau) given the (normalised) observation; same shape as x."""
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(f"Expected observation shape [batch, {self.observation_dim}], got {tuple(observation.shape)}")
        if x.shape != (observation.shape[0], self.horizon, self.action_dim):
            raise ValueError(f"Expected x shape [batch, {self.horizon}, {self.action_dim}], got {tuple(x.shape)}")
        features = torch.cat([x.reshape(x.shape[0], -1), observation, self.time_features(tau)], dim=1)
        return self.network(features).reshape(x.shape)

    def flow_matching_loss(
        self, observation: Tensor, target_chunk: Tensor, tau: Tensor | None = None, noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Conditional flow-matching MSE  || v_theta(x_tau, tau, s) - (x_1 - x_0) ||^2."""
        if target_chunk.shape != (observation.shape[0], self.horizon, self.action_dim):
            raise ValueError(f"Expected target shape [batch, {self.horizon}, {self.action_dim}], got {tuple(target_chunk.shape)}")
        batch = observation.shape[0]
        if tau is None:
            tau = torch.rand(batch, generator=generator, dtype=target_chunk.dtype)
        if noise is None:
            noise = torch.randn(target_chunk.shape, generator=generator, dtype=target_chunk.dtype)
        t = tau.reshape(batch, 1, 1)
        x_tau = (1.0 - t) * noise + t * target_chunk
        predicted = self.velocity(x_tau, tau, observation)
        return torch.nn.functional.mse_loss(predicted, target_chunk - noise)

    def sample(self, observation: Tensor, noise: Tensor | None = None, steps: int | None = None,
               generator: torch.Generator | None = None) -> Tensor:
        """Euler integration from x_0 = noise (default N(0, I)) to x_1.  Deterministic for a
        given noise; differentiable with respect to the parameters when autograd is enabled."""
        steps = steps or self.sample_steps
        batch = observation.shape[0]
        if noise is None:
            noise = torch.randn((batch, self.horizon, self.action_dim), generator=generator, dtype=observation.dtype)
        x = noise
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full((batch,), index * dt, dtype=observation.dtype, device=observation.device)
            x = x + dt * self.velocity(x, tau, observation)
        return x

    def forward(self, observation: Tensor) -> Tensor:
        return self.sample(observation)
