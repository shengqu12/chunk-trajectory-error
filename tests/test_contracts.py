import pytest

from chunk_trajectory_error.contracts import ChunkSpec
from chunk_trajectory_error.replay import position_errors, trajectory_cost_m2


def test_chunk_spec_requires_full_evaluation_window() -> None:
    ChunkSpec(20, 5, 15, 10, delay=4).validate()
    with pytest.raises(ValueError, match="horizon"):
        ChunkSpec(13, 5, 15, 10, delay=4).validate()


def test_expert_replay_has_zero_cost() -> None:
    positions = [(0.0, 0.1, 0.2), (0.2, 0.1, 0.2)]
    distances = position_errors(positions, positions)
    assert trajectory_cost_m2(distances) == 0.0


def test_position_alignment_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        position_errors([(0.0, 0.0, 0.0)], [])