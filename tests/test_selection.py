import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle
from chunk_trajectory_error.selection import ChunkSelector


def make_selector(mode: str, num_samples: int = 6) -> ChunkSelector:
    torch.manual_seed(0)
    policy = PolicyBundle(10, 4, 8, [32, 32], policy_type="flow_chunk", flow={"time_frequencies": 2, "sample_steps": 3}, sample_seed=1)
    error = ErrorBundle(history_steps=3, state_dim=4, evaluation_steps=5, action_dim=4, hidden_dim=16)
    return ChunkSelector(policy, error, num_samples, delay=1, history_steps=3, tcp_slice=slice(0, 3), robot_slice=slice(0, 4), mode=mode)


def test_first_sample_until_history_is_complete_then_selects() -> None:
    selector = make_selector("best")
    selector.reset_episode()
    obs = np.random.default_rng(0).normal(size=(6, 10)).astype(np.float32)
    for t in range(3):
        selector.observe(obs[t])
        selector.act(obs[t])
    assert all(not s["selected"] for s in selector.stats)
    selector.observe(obs[3])
    chunk = selector.act(obs[3])
    assert chunk.shape == (8, 4) and selector.stats[-1]["selected"]
    assert np.allclose(selector.current_history()[-1], obs[2, :3] - obs[3, :3])


def test_best_and_worst_pick_opposite_ends_of_the_critic_score() -> None:
    best, worst = make_selector("best"), make_selector("worst")
    worst.policy.load_state_dict(best.policy.state_dict())
    worst.error.load_state_dict(best.error.state_dict())
    obs = np.random.default_rng(1).normal(size=10).astype(np.float32)
    history = np.random.default_rng(2).normal(size=(3, 3)).astype(np.float32)
    best.reseed(5)
    worst.reseed(5)  # identical samples
    best.act_with_history(obs, history)
    worst.act_with_history(obs, history)
    assert best.stats[-1]["score_chosen"] == pytest.approx(best.stats[-1]["score_min"])
    assert worst.stats[-1]["score_chosen"] == pytest.approx(worst.stats[-1]["score_max"])
    assert best.stats[-1]["score_min"] <= worst.stats[-1]["score_max"]


def test_selector_does_not_touch_parameters() -> None:
    selector = make_selector("best")
    before = [p.clone() for p in selector.policy.parameters()]
    obs = np.zeros(10, np.float32)
    selector.act_with_history(obs, np.zeros((3, 3), np.float32))
    assert all(torch.equal(a, b) for a, b in zip(before, selector.policy.parameters()))
    assert selector.summary()["selected_decisions"] == 1
