import pytest

torch = pytest.importorskip("torch")

from chunk_trajectory_error.pipeline import PolicyBundle
from chunk_trajectory_error.policies.flow_chunk import FlowChunkPolicy


def make_policy() -> FlowChunkPolicy:
    torch.manual_seed(0)
    return FlowChunkPolicy(observation_dim=6, action_dim=4, horizon=5, hidden_dims=(32, 32), time_frequencies=2, sample_steps=4)


def test_sample_has_chunk_shape_and_is_deterministic_given_noise() -> None:
    policy = make_policy()
    obs = torch.randn(3, 6)
    noise = torch.randn(3, 5, 4)
    a = policy.sample(obs, noise=noise)
    b = policy.sample(obs, noise=noise.clone())
    assert a.shape == (3, 5, 4)
    assert torch.allclose(a, b)
    c = policy.sample(obs, noise=torch.randn(3, 5, 4))
    assert not torch.allclose(a, c)


def test_loss_on_sample_reaches_every_parameter_through_the_euler_chain() -> None:
    policy = make_policy()
    obs = torch.randn(8, 6)
    chunk = policy.sample(obs, noise=torch.randn(8, 5, 4))
    loss = (chunk[:, 1:] ** 2).mean()  # any loss on the sampled tail
    grads = torch.autograd.grad(loss, list(policy.parameters()))
    assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in grads)


def test_flow_matching_loss_is_finite_and_trains_towards_a_fixed_chunk() -> None:
    policy = make_policy()
    obs = torch.zeros(64, 6)
    target = torch.full((64, 5, 4), 0.5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(1)
    first = None
    for _ in range(300):
        loss = policy.flow_matching_loss(obs, target, generator=generator)
        if first is None:
            first = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        sample = policy.sample(obs[:4], noise=torch.randn(4, 5, 4, generator=generator), steps=16)
    assert torch.isfinite(loss)
    assert float(loss) < first
    assert (sample - 0.5).abs().mean() < 0.25


def test_bundle_round_trips_flow_policy_and_keeps_mlp_default(tmp_path) -> None:
    bundle = PolicyBundle(6, 4, 5, [32, 32], policy_type="flow_chunk", flow={"time_frequencies": 2, "sample_steps": 4}, sample_seed=3)
    bundle.set_normalizer(torch.randn(100, 6))
    obs = torch.randn(2, 6)
    bundle.reseed()
    first = bundle(obs)
    path = tmp_path / "flow.pt"
    bundle.save(path)
    loaded = PolicyBundle.load(path)
    assert loaded.is_flow and loaded.meta["flow"]["sample_steps"] == 4
    assert torch.allclose(loaded(obs), first)  # same seeded noise after load
    assert loaded.act(obs[0].numpy()).shape == (5, 4)
    mlp = PolicyBundle(6, 4, 5, [32])
    assert not mlp.is_flow and mlp(obs).shape == (2, 5, 4)
