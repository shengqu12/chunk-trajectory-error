import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chunk_trajectory_error.contracts import ChunkSpec  # noqa: E402
from chunk_trajectory_error.pipeline import (  # noqa: E402
    ErrorBundle, PolicyBundle, anchor_times, bootstrap_ci, branch_cost, hold_padded_chunk, low_frequency_perturbation,
)

SPEC = ChunkSpec(horizon=20, execute_steps=5, history_steps=15, evaluation_steps=10, delay=0)


def test_anchor_times_respect_history_and_future_window():
    times = anchor_times(74, SPEC, 4)
    assert times[0] == 15 and times[-1] == 64 and len(times) == 4
    assert anchor_times(20, SPEC, 3) == []  # L + K = 25 steps are needed


def test_hold_padding_keeps_gripper_and_zeroes_motion():
    actions = np.arange(24, dtype=np.float32).reshape(6, 4)
    chunk = hold_padded_chunk(actions, 4, 5, gripper_dims=[3])
    assert chunk.shape == (5, 4)
    np.testing.assert_array_equal(chunk[:2], actions[4:])
    np.testing.assert_array_equal(chunk[2:, :3], 0.0)
    np.testing.assert_array_equal(chunk[2:, 3], actions[-1, 3])


def test_perturbation_is_low_frequency_bounded_and_leaves_gripper_alone():
    rng = np.random.default_rng(0)
    for _ in range(50):
        delta = low_frequency_perturbation(rng, 10, 4, [3], (0.01, 0.2))
        assert np.all(delta[:, 3] == 0.0)
        assert np.abs(delta).max() <= 0.2 * 1.5 + 1e-6  # bias plus half-amplitude ramp
        np.testing.assert_allclose(np.diff(delta[:, :3], n=2, axis=0), 0.0, atol=1e-6)  # bias + linear ramp only


def test_branch_cost_is_zero_for_identical_trajectories_and_matches_definition():
    expert = np.zeros((10, 3))
    candidate = expert.copy()
    assert branch_cost(candidate, expert) == 0.0
    candidate[:, 0] = 0.01
    assert branch_cost(candidate, expert) == pytest.approx(1e-4)


def test_error_bundle_targets_agree_on_cost_units_and_freeze_cleanly():
    history, state, actions = torch.randn(4, 15, 3), torch.randn(4, 26), torch.rand(4, 10, 4)
    labels = torch.tensor([1e-5, 4e-5, 1e-4, 9e-4])
    for target in ("J", "sqrtJ"):
        bundle = ErrorBundle(15, 26, 10, 4, 32, target=target)
        bundle.set_normalizer(history, state, labels)
        assert bundle.meta["j_ref_m2"] == pytest.approx(float(labels.mean()))
        cost = bundle.cost_m2(history, state, actions)
        assert torch.all(cost >= 0)
        torch.testing.assert_close(bundle.normalized_cost(history, state, actions), cost / float(labels.mean()))
        expected = labels if target == "J" else labels.sqrt()
        torch.testing.assert_close(bundle.label(labels) * bundle.j_scale, expected)


def test_no_history_bundle_ignores_history():
    bundle = ErrorBundle(15, 26, 10, 4, 32, use_history=False)
    state, actions = torch.randn(2, 26), torch.rand(2, 10, 4)
    first = bundle(torch.randn(2, 15, 3), state, actions)
    second = bundle(torch.randn(2, 15, 3), state, actions)
    torch.testing.assert_close(first, second)


def test_policy_bundle_round_trip_and_clipped_actions(tmp_path):
    bundle = PolicyBundle(42, 4, 20, [16])
    bundle.set_normalizer(torch.randn(50, 42))
    path = tmp_path / "pi.pt"
    bundle.save(path, {"note": "x"})
    loaded = PolicyBundle.load(path)
    observation = np.random.default_rng(0).standard_normal(42).astype(np.float32) * 50
    chunk = loaded.act(observation)
    assert chunk.shape == (20, 4) and np.abs(chunk).max() <= 1.0
    np.testing.assert_allclose(chunk, bundle.eval().act(observation))


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(1).normal(1.0, 0.5, 200)
    mean, low, high = bootstrap_ci(values, groups=np.repeat(np.arange(50), 4))
    assert low < mean < high
