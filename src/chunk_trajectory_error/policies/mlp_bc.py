from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MLPChunkPolicy(nn.Module):
    """State-to-action-chunk MLP for the low-cost ManiSkill experiment."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, horizon) <= 0:
            raise ValueError("observation_dim, action_dim, and horizon must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, horizon * action_dim))

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.network = nn.Sequential(*layers)

    def forward(self, observation: Tensor) -> Tensor:
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation shape [batch, {self.observation_dim}], "
                f"got {tuple(observation.shape)}"
            )
        actions = self.network(observation)
        return actions.reshape(observation.shape[0], self.horizon, self.action_dim)


def behavior_cloning_loss(predicted_actions: Tensor, expert_actions: Tensor) -> Tensor:
    if predicted_actions.shape != expert_actions.shape:
        raise ValueError(
            "Predicted and expert action chunks must have the same shape; "
            f"got {tuple(predicted_actions.shape)} and {tuple(expert_actions.shape)}"
        )
    return torch.nn.functional.mse_loss(predicted_actions, expert_actions)