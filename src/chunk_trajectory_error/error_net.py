import torch
from torch import Tensor, nn


class HistoryConditionedErrorNetwork(nn.Module):
    """Predict same-start trajectory cost from history, state, actions, and delay."""

    def __init__(
        self,
        history_steps: int,
        robot_state_dim: int,
        evaluation_steps: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        dimensions = (history_steps, robot_state_dim, evaluation_steps, action_dim, hidden_dim)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("All error-network dimensions must be positive")

        self.history_steps = history_steps
        self.robot_state_dim = robot_state_dim
        self.evaluation_steps = evaluation_steps
        self.action_dim = action_dim
        history_dim = history_steps * 3
        action_chunk_dim = evaluation_steps * action_dim

        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(robot_state_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_chunk_dim, hidden_dim), nn.ReLU())
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        history: Tensor,
        robot_state: Tensor,
        candidate_actions: Tensor,
        delay: int | Tensor,
    ) -> Tensor:
        batch_size = robot_state.shape[0]
        expected_history = (batch_size, self.history_steps, 3)
        expected_state = (batch_size, self.robot_state_dim)
        expected_actions = (batch_size, self.evaluation_steps, self.action_dim)
        if tuple(history.shape) != expected_history:
            raise ValueError(f"Expected history shape {expected_history}, got {tuple(history.shape)}")
        if tuple(robot_state.shape) != expected_state:
            raise ValueError(f"Expected robot state shape {expected_state}, got {tuple(robot_state.shape)}")
        if tuple(candidate_actions.shape) != expected_actions:
            raise ValueError(
                f"Expected candidate action shape {expected_actions}, got {tuple(candidate_actions.shape)}"
            )

        delay_tensor = torch.as_tensor(delay, dtype=robot_state.dtype, device=robot_state.device)
        if delay_tensor.ndim == 0:
            delay_tensor = delay_tensor.expand(batch_size).reshape(batch_size, 1)
        elif delay_tensor.ndim == 1 and delay_tensor.shape[0] == batch_size:
            delay_tensor = delay_tensor.reshape(batch_size, 1)
        elif tuple(delay_tensor.shape) != (batch_size, 1):
            raise ValueError(f"delay must be scalar or have shape [{batch_size}]")

        encoded = torch.cat(
            (
                self.history_encoder(history.flatten(start_dim=1)),
                self.state_encoder(robot_state),
                self.action_encoder(candidate_actions.flatten(start_dim=1)),
                delay_tensor,
            ),
            dim=-1,
        )
        return self.regressor(encoded).squeeze(-1)


def error_network_loss(predicted_cost: Tensor, target_cost_m2: Tensor) -> Tensor:
    target = target_cost_m2.to(device=predicted_cost.device, dtype=predicted_cost.dtype).reshape(-1)
    if predicted_cost.shape != target.shape:
        raise ValueError(
            f"Predicted and target costs must have the same batch shape, got "
            f"{tuple(predicted_cost.shape)} and {tuple(target.shape)}"
        )
    return torch.nn.functional.mse_loss(predicted_cost, target)