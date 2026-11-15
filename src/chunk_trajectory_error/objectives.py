from torch import Tensor, nn

from chunk_trajectory_error.error_net import HistoryConditionedErrorNetwork


def surrogate_regularized_policy_loss(
    imitation_loss: Tensor,
    error_network: HistoryConditionedErrorNetwork,
    history: Tensor,
    robot_state: Tensor,
    candidate_actions: Tensor,
    delay: int | Tensor,
    weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine imitation and critic losses while retaining d(error)/d(actions)."""
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if imitation_loss.ndim != 0:
        raise ValueError("imitation_loss must be a scalar")

    for parameter in error_network.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    predicted_cost = error_network(history, robot_state, candidate_actions, delay)
    surrogate_loss = predicted_cost.mean()
    total_loss = imitation_loss + weight * surrogate_loss
    return total_loss, {
        "imitation_loss": imitation_loss.detach(),
        "surrogate_loss": surrogate_loss.detach(),
        "total_loss": total_loss.detach(),
    }


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.grad = None