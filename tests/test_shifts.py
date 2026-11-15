"""Shift module: latency command ordering (pure), and on the simulator: shifts are applied
once, persist across resets, and identity shifts reproduce the nominal rollout exactly."""

from pathlib import Path

import numpy as np
import pytest

from chunk_trajectory_error.shifts import LatencyBackend, ShiftSpec

ROOT = Path(__file__).resolve().parents[1]


class RecordingBackend:
    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []
        self.resets = 0

    def reset(self, seed=None, init_state=None):
        self.resets += 1
        self.applied = []

    def step(self, action):
        self.applied.append(np.asarray(action).copy())

    def snapshot(self):
        return {"applied": [a.copy() for a in self.applied]}

    def restore(self, snapshot):
        self.applied = [a.copy() for a in snapshot["applied"]]


def commands(n: int) -> list[np.ndarray]:
    return [np.array([k + 1.0, -(k + 1.0), 0.5 * k, 1.0 if k < 3 else -1.0], np.float32) for k in range(n)]


@pytest.mark.parametrize("delay", [0, 1, 2, 4])
def test_latency_applies_command_k_at_step_k_plus_d(delay):
    inner = RecordingBackend()
    backend = LatencyBackend(inner, delay, gripper_dims=[3])
    backend.reset(seed=0)
    sent = commands(8)
    for k, command in enumerate(sent):
        backend.step(command)
        assert len(backend.pending()) == delay  # d commands are always in flight
    applied = inner.applied
    assert len(applied) == len(sent)
    for k in range(delay):  # hold: zero translation, gripper of the first (current) command
        np.testing.assert_array_equal(applied[k][:3], 0.0)
        assert applied[k][3] == sent[0][3]
    for k in range(delay, len(sent)):
        np.testing.assert_array_equal(applied[k], sent[k - delay])
    # the queue holds exactly the d most recent commands, oldest first
    np.testing.assert_array_equal(backend.pending(), np.array(sent[len(sent) - delay :]).reshape(delay, 4))


def test_latency_queue_is_cleared_on_reset_and_round_trips_through_snapshots():
    inner = RecordingBackend()
    backend = LatencyBackend(inner, 2, gripper_dims=[3])
    backend.reset(seed=0)
    for command in commands(5):
        backend.step(command)
    snapshot = backend.snapshot()
    pending = backend.pending().copy()
    backend.step(commands(6)[5])
    backend.restore(snapshot)
    np.testing.assert_array_equal(backend.pending(), pending)
    backend.reset(seed=1)
    assert backend.pending().shape[0] == 0 and backend.last_applied is None
    first = commands(1)[0]
    backend.step(first)
    np.testing.assert_array_equal(inner.applied[0][:3], 0.0)  # hold again after the reset


def test_shift_spec_names_and_validation():
    assert ShiftSpec("actuation", stiffness_scale=0.6).name == "actuation_s0.6"
    assert ShiftSpec("contact", friction_scale=0.3, mass_scale=3).name == "contact_f0.3_m3"
    assert ShiftSpec("contact", friction_scale=0.3, mass_scale=3, contact_mode="shared_in_place").name == "contact_shared_f0.3_m3"
    with pytest.raises(ValueError):
        ShiftSpec("gravity")


# --------------------------------------------------------------------------- #
# simulator tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sim():
    pytest.importorskip("mani_skill")
    from chunk_trajectory_error.config import load_config
    from chunk_trajectory_error.pipeline import load_demoset

    config = load_config(ROOT / "configs/maniskill_pickcube.yaml")
    for key in ("replayed_trajectory_path", "split_manifest"):
        path = ROOT / config["data"][key]
        if not path.is_file():
            pytest.skip(f"missing {path}")
        config["data"][key] = str(path)
    demos = load_demoset(config)
    return config, demos


def rollout(backend, demos, steps=30):
    episode = demos.episodes[demos.ids("test")[0]]
    backend.reset(seed=episode.seed, init_state=episode.state_at(0))
    positions = []
    for action in episode.actions[:steps]:
        backend.step(action)
        positions.append(backend.ee_position())
    cube = backend.base.cube.pose.p[0].numpy().copy() if not isinstance(backend, LatencyBackend) else backend.inner.base.cube.pose.p[0].numpy().copy()
    return np.array(positions), cube


@pytest.mark.parametrize(
    "shift",
    [ShiftSpec("actuation", stiffness_scale=0.4), ShiftSpec("contact", friction_scale=0.3, mass_scale=3),
     ShiftSpec("latency", delay_steps=2)],
    ids=lambda s: s.name,
)
def test_shift_is_applied_once_and_persists_across_resets(sim, shift):
    from chunk_trajectory_error.shifts import apply_shift, make_target_backend, read_physics, verify_shift

    config, demos = sim
    backend, record = make_target_backend(config, shift, demos.gripper_dims)
    try:
        nominal, applied = record["nominal"], record["applied"]
        if shift.family == "actuation":
            for index, (name, values) in enumerate(applied["joints"].items()):
                factor = shift.stiffness_scale if index < 7 else 1.0
                assert values["stiffness"] == pytest.approx(nominal["joints"][name]["stiffness"] * factor)
                assert values["damping"] == pytest.approx(nominal["joints"][name]["damping"] * factor)
            assert nominal["joints"]["panda_joint1"]["stiffness"] == pytest.approx(1000.0)
            assert nominal["joints"]["panda_joint1"]["damping"] == pytest.approx(100.0)
        if shift.family == "contact":
            assert nominal["cube_mass_kg"][0] == pytest.approx(0.064)
            assert applied["cube_mass_kg"][0] == pytest.approx(0.064 * shift.mass_scale)
            assert applied["cube_friction"][0]["static"] == pytest.approx(nominal["cube_friction"][0]["static"] * shift.friction_scale)
            assert applied["table_friction"] == nominal["table_friction"]  # cube_only leaves the table alone
            assert applied["default_material_friction"] == nominal["default_material_friction"]
        for seed in (3, 4, 5):  # several resets, with and without an init state; values must not drift or compound
            backend.reset(seed=seed)
            verify_shift(backend, record)
        rollout(backend, demos, steps=5)
        verify_shift(backend, record)
        inner = backend.inner if isinstance(backend, LatencyBackend) else backend
        assert read_physics(inner.base) == record["after_reset"]
        with pytest.raises(RuntimeError):
            apply_shift(inner, shift, demos.gripper_dims)
    finally:
        backend.close()


@pytest.mark.parametrize(
    "shift",
    [ShiftSpec("actuation", stiffness_scale=1.0), ShiftSpec("contact", friction_scale=1.0, mass_scale=1.0),
     ShiftSpec("latency", delay_steps=0)],
    ids=lambda s: "identity_" + s.family,
)
def test_identity_shift_reproduces_the_nominal_rollout_exactly(sim, shift):
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.shifts import make_target_backend

    config, demos = sim
    nominal = ManiSkillBackend(config)
    nominal.reset(seed=0)
    nominal.reset(seed=1)  # same reset history as make_target_backend
    reference, reference_cube = rollout(nominal, demos)
    nominal.close()
    target, _ = make_target_backend(config, shift, demos.gripper_dims)
    shifted, shifted_cube = rollout(target, demos)
    target.close()
    np.testing.assert_array_equal(shifted, reference)
    np.testing.assert_array_equal(shifted_cube, reference_cube)


def test_latency_in_simulation_equals_nominal_with_shifted_commands(sim):
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.shifts import make_target_backend

    config, demos = sim
    episode = demos.episodes[demos.ids("test")[0]]
    actions = episode.actions[:20]
    delay = 2
    hold = np.zeros_like(actions[0])
    hold[demos.gripper_dims] = actions[0][demos.gripper_dims]
    shifted_actions = np.concatenate([np.repeat(hold[None], delay, 0), actions[:-delay]])

    nominal = ManiSkillBackend(config)
    nominal.reset(seed=0)
    nominal.reset(seed=episode.seed, init_state=episode.state_at(0))
    expected = []
    for action in shifted_actions:
        nominal.step(action)
        expected.append(nominal.ee_position())
    nominal.close()

    target, _ = make_target_backend(config, ShiftSpec("latency", delay_steps=delay), demos.gripper_dims)
    target.reset(seed=episode.seed, init_state=episode.state_at(0))
    got = []
    for action in actions:
        target.step(action)
        got.append(target.ee_position())
    target.close()
    np.testing.assert_array_equal(np.array(got), np.array(expected))
