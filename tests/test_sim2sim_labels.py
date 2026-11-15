"""Label alignment for the sim-to-sim gap labels.

An identity shift must give J_gap = 0 exactly on every sample.  This holds exactly on a
deterministic simulator whose restore is exact (the fake below), which pins the indexing:
p_target[t+j] is compared with the source position after the j-th replayed command.  On
ManiSkill/PhysX a restore is not bit-exact (solver internals are not part of the state), so the
exact-zero check is recorded as a strict expected failure and the residual is bounded instead.
"""

from pathlib import Path

import numpy as np
import pytest

from chunk_trajectory_error.shifts import LatencyBackend
from chunk_trajectory_error.sim2sim import (
    collect_target_episode, gap_labels_for_episode, history_features, ee_at_states, label_times, roughness,
)

ROOT = Path(__file__).resolve().parents[1]
L, K, H, E = 15, 10, 20, 5


class FakeSim:
    """Deterministic integrator with an exact restore: p <- p + 0.01 * a[:3] (+ a small nonlinearity)."""

    def __init__(self) -> None:
        self.p = np.zeros(3)
        self.cube = np.zeros(3)

    def reset(self, seed=None, init_state=None):
        rng = np.random.default_rng(seed)
        self.p, self.cube = rng.uniform(-0.1, 0.1, 3), rng.uniform(-0.1, 0.1, 3)
        return self.observe()

    def observe(self):
        return {"state": np.concatenate([self.p, self.cube]).astype(np.float32)}

    def step(self, action):
        action = np.asarray(action, np.float64)
        self.p = self.p + 0.01 * action[:3] + 0.001 * np.sin(10 * self.p)
        self.cube = self.cube + 0.5 * (self.p - self.cube) * (action[3] < 0)
        return self.observe()

    def ee_position(self):
        return tuple(self.p)

    def cube_position(self):
        return self.cube.copy()

    def is_success(self):
        return False

    def state_dict_numpy(self):
        return {"p": self.p.copy(), "cube": self.cube.copy()}

    def restore(self, state):
        self.p, self.cube = np.array(state["p"]), np.array(state["cube"])


def policy(observation):
    phase = observation[:3].sum()
    chunk = np.stack([np.array([np.sin(phase + 0.3 * h), np.cos(phase - 0.2 * h), 0.5 * np.sin(2 * h), 1.0 if h < 12 else -1.0])
                      for h in range(H)])
    return np.clip(chunk, -1, 1).astype(np.float32)


def collect(target, perturb):
    rng = np.random.default_rng(3) if perturb else None
    return collect_target_episode(target, policy, H, E, seed=7, max_steps=60, gripper_dims=[3], perturb_rng=rng,
                                  amplitude_range=(0.01, 0.2), step_budget=60)


@pytest.mark.parametrize("perturb", [False, True])
def test_identity_shift_gives_exactly_zero_gap_on_every_sample(perturb):
    episode = collect(FakeSim(), perturb)
    source = FakeSim()
    source.reset(seed=0)
    rows = gap_labels_for_episode(source, episode, 0, L, K, slice(0, 3))
    assert len(rows) == len(label_times(60, L, K)) == 60 - L - K + 1
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["J_gap_m2"] == 0.0 and row["J_gap_cube_m2"] == 0.0 for row in rows)


def test_latency_labels_match_the_closed_form_and_use_the_logged_pending_command():
    target = LatencyBackend(FakeSim(), 1, gripper_dims=[3])
    episode = collect(target, perturb=False)
    np.testing.assert_array_equal(episode["pending"][0][0, :3], 0.0)  # start-of-episode hold is logged as in flight
    for t in range(1, episode["steps"]):
        np.testing.assert_array_equal(episode["pending"][t][0], episode["commanded"][t - 1])  # d = 1: previous command
        np.testing.assert_array_equal(episode["applied"][t], episode["commanded"][t - 1])
    source = FakeSim()
    source.reset(seed=0)
    rows = gap_labels_for_episode(source, episode, 0, L, K, slice(0, 3))
    for row in rows:
        t = row["t"]
        # recompute both branches by hand from the logged state
        replay = FakeSim()
        replay.restore({"p": episode["ee_before"][t], "cube": episode["cube_before"][t]})
        p_source = []
        for action in episode["commanded"][t : t + K]:
            replay.step(action)
            p_source.append(replay.p.copy())
        expected = np.mean(np.sum((episode["ee_after"][t : t + K] - np.array(p_source)) ** 2, axis=1))
        assert row["J_gap_m2"] == pytest.approx(expected, rel=0, abs=0)
        assert row["J_gap_m2"] > 0.0
        np.testing.assert_array_equal(row["pending"], episode["pending"][t])
        np.testing.assert_array_equal(row["actions"], episode["commanded"][t : t + K])


def test_history_and_roughness_features():
    episode = collect(FakeSim(), perturb=False)
    states = ee_at_states(episode)
    assert states.shape == (61, 3)
    history = history_features(states, 20, L)
    np.testing.assert_allclose(history[-1], states[19] - states[20])
    actions = np.zeros((2, K, 4), np.float32)
    actions[1, ::2, 0] = 1.0  # alternating 0/1 on one channel -> every consecutive difference is 1
    actions[1, :, 3] = np.linspace(-1, 1, K)  # gripper must be ignored
    np.testing.assert_allclose(roughness(actions, [0, 1, 2]), [0.0, 1.0])


# --------------------------------------------------------------------------- #
# ManiSkill
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sim_labels():
    pytest.importorskip("mani_skill")
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.config import load_config
    from chunk_trajectory_error.pipeline import PolicyBundle, load_demoset
    from chunk_trajectory_error.shifts import ShiftSpec, make_target_backend

    config = load_config(ROOT / "configs/maniskill_pickcube.yaml")
    for key in ("replayed_trajectory_path", "split_manifest"):
        config["data"][key] = str(ROOT / config["data"][key])
    checkpoint = ROOT / "runs/maniskill_pickcube/checkpoints/pi_bc/pi.pt"
    if not checkpoint.is_file():
        pytest.skip("pi_bc checkpoint missing")
    demos = load_demoset(config)
    bundle = PolicyBundle.load(checkpoint)
    target, _ = make_target_backend(config, ShiftSpec("nominal"), demos.gripper_dims)
    episode = collect_target_episode(target, bundle.act, H, E, demos.episodes[demos.ids("train")[0]].seed, 200,
                                     demos.gripper_dims, None, (0.01, 0.2), 60)
    target.close()
    source = ManiSkillBackend(config)
    source.reset(seed=0)
    rows = gap_labels_for_episode(source, episode, 0, L, K, slice(0, 26))
    source.close()
    return np.array([row["J_gap_m2"] for row in rows])


@pytest.mark.xfail(strict=True, reason="PhysX restore is not bit-exact: solver internals are not part of the ManiSkill state dict")
def test_identity_shift_gives_exactly_zero_gap_in_maniskill(sim_labels):
    assert np.all(sim_labels == 0.0)


def test_identity_shift_gap_in_maniskill_is_below_the_replay_tolerance(sim_labels):
    assert len(sim_labels) == 60 - L - K + 1 and np.all(np.isfinite(sim_labels))
    assert sim_labels.max() < 1e-6  # (1 mm)^2, the replay audit tolerance
