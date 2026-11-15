import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from chunk_trajectory_error.backends.maniskill import (  # noqa: E402
    check_action_state_boundary,
    leading_length,
    load_demo_episodes,
)
from chunk_trajectory_error.replay import position_errors, trajectory_cost_m2  # noqa: E402


def write_demo(tmp_path, num_actions, num_states):
    path = tmp_path / "demo.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_3")
        group["actions"] = np.zeros((num_actions, 4), dtype=np.float32)
        group["env_states/actors/cube"] = np.arange(num_states * 13, dtype=np.float32).reshape(num_states, 13)
        group["env_states/articulations/panda"] = np.zeros((num_states, 31), dtype=np.float32)
    meta = {
        "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}},
        "episodes": [{"episode_id": 3, "episode_seed": 11, "control_mode": "pd_ee_delta_pos", "success": True}],
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def test_reader_returns_t_actions_and_t_plus_one_states(tmp_path):
    episode = load_demo_episodes(write_demo(tmp_path, 6, 7))[0]
    assert (episode.episode_id, episode.seed, episode.num_actions, episode.num_states) == (3, 11, 6, 7)
    check_action_state_boundary(episode.num_actions, episode.num_states)
    assert episode.state_at(2)["actors"]["cube"][0] == 26.0
    with pytest.raises(IndexError):
        episode.state_at(7)


def test_boundary_check_rejects_equal_lengths(tmp_path):
    episode = load_demo_episodes(write_demo(tmp_path, 6, 6))[0]
    with pytest.raises(ValueError):
        check_action_state_boundary(episode.num_actions, episode.num_states)


def test_inconsistent_state_leaves_are_rejected():
    with pytest.raises(ValueError):
        leading_length({"a": np.zeros((3, 2)), "b": np.zeros((4, 2))})


def test_one_frame_misalignment_gives_nonzero_cost():
    positions = [[0.01 * step, 0.0, 0.0] for step in range(12)]
    aligned = position_errors(positions[1:11], positions[1:11])
    shifted = position_errors(positions[0:10], positions[1:11])
    assert trajectory_cost_m2(aligned) == 0.0
    assert trajectory_cost_m2(shifted) == pytest.approx(1e-4)
