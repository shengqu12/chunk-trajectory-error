import pytest

torch = pytest.importorskip("torch")

from chunk_trajectory_error.error_net import HistoryConditionedErrorNetwork, error_network_loss
from chunk_trajectory_error.objectives import surrogate_regularized_policy_loss
from chunk_trajectory_error.policies.mlp_bc import MLPChunkPolicy


def test_policy_outputs_configured_action_chunk_shape() -> None:
    policy = MLPChunkPolicy(observation_dim=8, action_dim=7, horizon=20)
    actions = policy(torch.randn(4, 8))
    assert actions.shape == (4, 20, 7)


def test_error_network_predicts_nonnegative_cost_and_fits_label() -> None:
    network = HistoryConditionedErrorNetwork(15, 8, 10, 7)
    history = torch.randn(4, 15, 3)
    state = torch.randn(4, 8)
    actions = torch.randn(4, 10, 7)
    prediction = network(history, state, actions, delay=0)
    loss = error_network_loss(prediction, torch.rand(4))
    assert prediction.shape == (4,)
    assert torch.all(prediction >= 0)
    assert torch.isfinite(loss)


def test_surrogate_loss_backpropagates_to_policy_not_error_network() -> None:
    policy = MLPChunkPolicy(observation_dim=8, action_dim=7, horizon=20)
    error_network = HistoryConditionedErrorNetwork(15, 8, 10, 7)
    observation = torch.randn(3, 8)
    history = torch.randn(3, 15, 3)
    robot_state = torch.randn(3, 8)
    policy_actions = policy(observation)
    candidate_actions = policy_actions[:, :10]
    imitation_loss = policy_actions.sum() * 0.0

    total_loss, losses = surrogate_regularized_policy_loss(
        imitation_loss,
        error_network,
        history,
        robot_state,
        candidate_actions,
        delay=0,
        weight=0.1,
    )
    total_loss.backward()

    policy_gradients = [parameter.grad for parameter in policy.parameters()]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient) > 0
        for gradient in policy_gradients
    )
    assert all(parameter.grad is None for parameter in error_network.parameters())
    assert set(losses) == {"imitation_loss", "surrogate_loss", "total_loss"}