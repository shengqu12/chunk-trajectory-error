"""Shared data, model-bundle and rollout utilities for stages 02-08 (ManiSkill)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from chunk_trajectory_error.contracts import ChunkSpec
from chunk_trajectory_error.error_net import HistoryConditionedErrorNetwork
from chunk_trajectory_error.policies.flow_chunk import FlowChunkPolicy
from chunk_trajectory_error.policies.mlp_bc import MLPChunkPolicy, behavior_cloning_loss
from chunk_trajectory_error.replay import position_errors, trajectory_cost_m2
from chunk_trajectory_error.scheduler import ChunkScheduler

# Small MLPs: one thread is fastest and keeps parallel stage processes from fighting over cores.
torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def chunk_spec(config: dict[str, Any], delay: int | None = None) -> ChunkSpec:
    chunk = config["chunk"]
    spec = ChunkSpec(
        horizon=chunk["horizon"],
        execute_steps=chunk["execute_steps"],
        history_steps=chunk["history_steps"],
        evaluation_steps=chunk["evaluation_steps"],
        delay=chunk["train_delay"] if delay is None else delay,
    )
    spec.validate()
    return spec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def out_dir(config: dict[str, Any]) -> Path:
    path = Path(config["experiment"]["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- #
# demonstrations and features
# --------------------------------------------------------------------------- #
@dataclass
class DemoSet:
    episodes: dict[int, Any]  # episode_id -> DemoEpisode
    splits: dict[str, list[int]]
    tcp_slice: slice
    robot_slice: slice
    gripper_dims: list[int]

    def ids(self, split: str) -> list[int]:
        return self.splits[split]

    def tcp(self, episode_id: int) -> np.ndarray:
        """End-effector XYZ at every state index, [T + 1, 3] (metres)."""
        return self.episodes[episode_id].obs[:, self.tcp_slice].astype(np.float32)

    def history(self, episode_id: int, t: int, steps: int) -> np.ndarray:
        """Past EE positions relative to the current one: p[t-L:t] - p[t]  ->  [L, 3]."""
        positions = self.tcp(episode_id)
        if t < steps:
            raise ValueError("not enough history")
        return positions[t - steps : t] - positions[t]

    def robot_state(self, episode_id: int, t: int) -> np.ndarray:
        return self.episodes[episode_id].obs[t, self.robot_slice].astype(np.float32)

    def observation(self, episode_id: int, t: int) -> np.ndarray:
        return self.episodes[episode_id].obs[t].astype(np.float32)


def load_demoset(config: dict[str, Any]) -> DemoSet:
    from chunk_trajectory_error.backends.maniskill import check_action_state_boundary, load_demo_episodes

    data_cfg, obs_cfg = config["data"], config["observation"]
    split_record = json.loads(Path(data_cfg["split_manifest"]).read_text(encoding="utf-8"))
    wanted = sorted(sum(split_record["splits"].values(), []))
    episodes = {ep.episode_id: ep for ep in load_demo_episodes(data_cfg["replayed_trajectory_path"], wanted)}
    for episode in episodes.values():
        check_action_state_boundary(episode.num_actions, episode.num_states)
        if episode.obs is None or episode.obs.shape[0] != episode.num_actions + 1:
            raise ValueError(f"episode {episode.episode_id} lacks T+1 state observations")
    return DemoSet(
        episodes=episodes,
        splits=split_record["splits"],
        tcp_slice=slice(*obs_cfg["tcp_position_slice"]),
        robot_slice=slice(*obs_cfg["robot_state_slice"]),
        gripper_dims=list(config["action"]["gripper_dims"]),
    )


def hold_padded_chunk(actions: np.ndarray, t: int, horizon: int, gripper_dims: list[int]) -> np.ndarray:
    """actions[t : t + H]; past the episode end, pad with a hold command
    (zero delta on continuous channels, last gripper command)."""
    chunk = actions[t : t + horizon]
    if len(chunk) < horizon:
        hold = np.zeros_like(actions[-1])
        hold[gripper_dims] = actions[-1][gripper_dims]
        chunk = np.concatenate([chunk, np.repeat(hold[None], horizon - len(chunk), axis=0)])
    return chunk.astype(np.float32)


def anchor_times(num_actions: int, spec: ChunkSpec, count: int) -> list[int]:
    first, last = spec.history_steps, num_actions - spec.delay - spec.evaluation_steps
    if last < first:
        return []
    return sorted({int(round(value)) for value in np.linspace(first, last, count)})


def build_policy_dataset(demos: DemoSet, split: str, spec: ChunkSpec) -> dict[str, torch.Tensor]:
    """One sample per (episode, t).  The error term is only valid where the
    history is complete and K real future steps exist (same range as the anchors)."""
    obs, chunks, hist, robot, valid, index = [], [], [], [], [], []
    for episode_id in demos.ids(split):
        episode = demos.episodes[episode_id]
        last_anchor = episode.num_actions - spec.delay - spec.evaluation_steps
        for t in range(episode.num_actions):
            obs.append(demos.observation(episode_id, t))
            chunks.append(hold_padded_chunk(episode.actions, t, spec.horizon, demos.gripper_dims))
            ok = spec.history_steps <= t <= last_anchor
            hist.append(demos.history(episode_id, t, spec.history_steps) if ok else np.zeros((spec.history_steps, 3), np.float32))
            robot.append(demos.robot_state(episode_id, t))
            valid.append(ok)
            index.append((episode_id, t))
    return {
        "obs": torch.tensor(np.stack(obs)),
        "chunk": torch.tensor(np.stack(chunks)),
        "history": torch.tensor(np.stack(hist)),
        "robot": torch.tensor(np.stack(robot)),
        "error_valid": torch.tensor(np.array(valid)),
        "index": torch.tensor(np.array(index)),
    }


# --------------------------------------------------------------------------- #
# model bundles (network + the statistics needed to run it)
# --------------------------------------------------------------------------- #
class PolicyBundle(nn.Module):
    """Chunk policy with train-set observation normalisation.  This is the only artefact
    the deployment path needs.

    ``policy_type`` selects the network: ``mlp_chunk`` (deterministic regression, the
    original experiment) or ``flow_chunk`` (conditional flow matching, sampled with a fixed
    number of Euler steps).  Checkpoints written before ``policy_type`` existed load as MLPs.
    For the flow policy, ``forward``/``act`` draw the initial noise from the bundle's own
    generator (seeded by ``sample_seed`` on construction and on load), so an evaluation run
    is reproducible without touching the global RNG.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, horizon: int, hidden_dims: list[int],
        policy_type: str = "mlp_chunk", flow: dict[str, Any] | None = None, sample_seed: int = 17,
    ) -> None:
        super().__init__()
        if policy_type not in ("mlp_chunk", "flow_chunk"):
            raise ValueError(f"unknown policy_type {policy_type!r}")
        self.meta = {
            "obs_dim": obs_dim, "action_dim": action_dim, "horizon": horizon, "hidden_dims": list(hidden_dims),
            "policy_type": policy_type, "flow": dict(flow or {}), "sample_seed": int(sample_seed),
        }
        self.policy_type = policy_type
        if policy_type == "mlp_chunk":
            self.policy = MLPChunkPolicy(obs_dim, action_dim, horizon, hidden_dims)
        else:
            self.policy = FlowChunkPolicy(obs_dim, action_dim, horizon, hidden_dims, **self.meta["flow"])
        self.generator = torch.Generator().manual_seed(int(sample_seed))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    @property
    def is_flow(self) -> bool:
        return self.policy_type == "flow_chunk"

    def reseed(self, seed: int | None = None) -> None:
        self.generator.manual_seed(int(self.meta["sample_seed"] if seed is None else seed))

    def set_normalizer(self, observations: torch.Tensor) -> None:
        self.obs_mean.copy_(observations.mean(0))
        self.obs_std.copy_(observations.std(0).clamp_min(1e-3))

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.obs_mean) / self.obs_std

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Action chunk [batch, H, action_dim].  For the flow policy this is one sample
        (noise from the bundle generator), differentiable w.r.t. the parameters."""
        if self.is_flow:
            return self.policy.sample(self.normalize(observation), generator=self.generator)
        return self.policy(self.normalize(observation))

    def sample(self, observation: torch.Tensor, noise: torch.Tensor | None = None, steps: int | None = None) -> torch.Tensor:
        """Flow policy only: sample with explicit noise/steps (used by audits and tests)."""
        if not self.is_flow:
            raise RuntimeError("sample() is only defined for the flow policy")
        return self.policy.sample(self.normalize(observation), noise=noise, steps=steps, generator=self.generator)

    def imitation_loss(self, observation: torch.Tensor, expert_chunk: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """The policy's own imitation objective: MSE for the MLP, the conditional
        flow-matching loss for the flow policy (tau and noise from ``generator``)."""
        if self.is_flow:
            return self.policy.flow_matching_loss(self.normalize(observation), expert_chunk, generator=generator)
        return behavior_cloning_loss(self(observation), expert_chunk)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Chunk [H, action_dim] in executor units, clipped to the normalised action box."""
        chunk = self(torch.as_tensor(observation, dtype=torch.float32)[None])[0]
        return chunk.clamp(-1.0, 1.0).numpy()

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "policy", "meta": self.meta, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @staticmethod
    def load(path: str | Path) -> "PolicyBundle":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        bundle = PolicyBundle(**blob["meta"])
        bundle.load_state_dict(blob["state_dict"])
        bundle.extra = blob.get("extra", {})
        return bundle.eval()


class ErrorBundle(nn.Module):
    """Error network plus input scaling and the label scale (fit on train only).

    ``forward`` returns the cost in label-scale units; multiply by ``j_scale`` for m^2.
    ``use_history=False`` feeds zeros instead of the EE history (no-history ablation).
    """

    def __init__(
        self, history_steps: int, state_dim: int, evaluation_steps: int, action_dim: int, hidden_dim: int,
        use_history: bool = True, state_source: str = "robot", target: str = "J", j_ref_m2: float = 1.0,
    ) -> None:
        super().__init__()
        if target not in ("J", "sqrtJ"):
            raise ValueError("target must be 'J' or 'sqrtJ'")
        self.meta = {
            "history_steps": history_steps, "state_dim": state_dim, "evaluation_steps": evaluation_steps,
            "action_dim": action_dim, "hidden_dim": hidden_dim, "use_history": use_history, "state_source": state_source,
            "target": target, "j_ref_m2": float(j_ref_m2),
        }
        self.target = target
        self.net = HistoryConditionedErrorNetwork(history_steps, state_dim, evaluation_steps, action_dim, hidden_dim)
        self.use_history = use_history
        self.state_source = state_source
        self.register_buffer("history_scale", torch.ones(()))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("j_scale", torch.ones(()))

    def set_normalizer(self, history: torch.Tensor, state: torch.Tensor, j_m2: torch.Tensor) -> None:
        self.history_scale.copy_(history.std().clamp_min(1e-4))
        self.state_mean.copy_(state.mean(0))
        self.state_std.copy_(state.std(0).clamp_min(1e-3))
        label = j_m2 if self.target == "J" else j_m2.sqrt()
        self.j_scale.copy_(label.mean().clamp_min(1e-12))
        self.meta["j_ref_m2"] = float(j_m2.mean().clamp_min(1e-12))

    def label(self, j_m2: torch.Tensor) -> torch.Tensor:
        """Regression target in label-scale units."""
        return (j_m2 if self.target == "J" else j_m2.sqrt()) / self.j_scale

    def cost_m2(self, history: torch.Tensor, state: torch.Tensor, actions: torch.Tensor, delay: int | torch.Tensor = 0) -> torch.Tensor:
        """Predicted J in m^2 (differentiable w.r.t. the actions)."""
        output = self(history, state, actions, delay) * self.j_scale
        return output if self.target == "J" else output**2

    def normalized_cost(self, history: torch.Tensor, state: torch.Tensor, actions: torch.Tensor, delay: int | torch.Tensor = 0) -> torch.Tensor:
        """Predicted J divided by the train-mean J, so that policy-loss weights are comparable across targets."""
        return self.cost_m2(history, state, actions, delay) / self.meta["j_ref_m2"]

    def forward(self, history: torch.Tensor, state: torch.Tensor, actions: torch.Tensor, delay: int | torch.Tensor = 0) -> torch.Tensor:
        scaled_history = history / self.history_scale if self.use_history else torch.zeros_like(history)
        return self.net(scaled_history, (state - self.state_mean) / self.state_std, actions, delay)

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "error", "meta": self.meta, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @staticmethod
    def load(path: str | Path) -> "ErrorBundle":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        bundle = ErrorBundle(**blob["meta"])
        bundle.load_state_dict(blob["state_dict"])
        if "j_ref_m2" not in blob["meta"]:  # checkpoints written before the sqrt-target option: label scale = mean J
            bundle.meta["j_ref_m2"] = float(bundle.j_scale)
        bundle.extra = blob.get("extra", {})
        return bundle.eval()


def select_state(demos: DemoSet, episode_id: int, t: int, source: str) -> np.ndarray:
    return demos.observation(episode_id, t) if source == "full_obs" else demos.robot_state(episode_id, t)


# --------------------------------------------------------------------------- #
# branch rollouts and labels
# --------------------------------------------------------------------------- #
def rollout_branch(backend: Any, snapshot: Any, actions: np.ndarray) -> tuple[np.ndarray, str]:
    """Restore the snapshot and execute a fixed-length action tail.  Returns EE
    positions after every step, [K, 3].  A failure of the task never truncates the
    rollout; only a simulator exception does, and it is reported in the status."""
    backend.restore(snapshot)
    positions = np.full((len(actions), 3), np.nan, dtype=np.float64)
    try:
        for index, action in enumerate(actions):
            backend.step(action)
            positions[index] = backend.ee_position()
    except Exception as error:  # recorded, never silently dropped
        return positions, f"aborted:{type(error).__name__}"
    return positions, "ok"


def branch_cost(candidate: np.ndarray, expert: np.ndarray) -> float:
    return trajectory_cost_m2(position_errors(candidate.tolist(), expert.tolist()))


def low_frequency_perturbation(
    rng: np.random.Generator, steps: int, action_dim: int, gripper_dims: list[int], amplitude_range: tuple[float, float]
) -> np.ndarray:
    """Constant bias plus a linear ramp on the continuous channels (same-direction
    drift); log-uniform amplitude; gripper channels untouched."""
    low, high = amplitude_range
    amplitude = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    continuous = [dim for dim in range(action_dim) if dim not in gripper_dims]
    bias = rng.uniform(-1.0, 1.0, size=len(continuous)) * amplitude
    ramp = rng.uniform(-0.5, 0.5, size=len(continuous)) * amplitude
    phase = np.linspace(-1.0, 1.0, steps)[:, None]
    delta = np.zeros((steps, action_dim), dtype=np.float32)
    delta[:, continuous] = bias[None] + ramp[None] * phase
    return delta


def anchor_snapshot(episode: Any, t: int, delay: int) -> Any:
    """Branch start for anchor ``t``: the recorded expert state that precedes
    ``actions[t + delay]``.  Loading it from the demonstration file keeps the snapshot
    exactly consistent with the recorded observation/history features; the replay audit
    (``hdf5_state_restore`` and ``snapshot_branch_repeatability``) certifies this path."""
    return episode.state_at(t + delay)


def policy_true_cost(
    backend: Any, demos: DemoSet, bundle: PolicyBundle, split: str, spec: ChunkSpec, anchors_per_episode: int,
    episode_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Simulator-recomputed J of the policy's own chunk on expert anchors of a split."""
    rows = []
    r, K = spec.delay, spec.evaluation_steps
    ids = demos.ids(split)[:episode_limit] if episode_limit else demos.ids(split)
    for episode_id in ids:
        episode = demos.episodes[episode_id]
        times = anchor_times(episode.num_actions, spec, anchors_per_episode)
        backend.reset(seed=episode.seed, init_state=episode.state_at(0))
        for t in times:
            snapshot = anchor_snapshot(episode, t, r)
            expert, _ = rollout_branch(backend, snapshot, episode.actions[t + r : t + r + K])
            chunk = bundle.act(demos.observation(episode_id, t))
            candidate, status = rollout_branch(backend, snapshot, chunk[r : r + K])
            rows.append({"episode_id": episode_id, "t": t, "J_m2": branch_cost(candidate, expert) if status == "ok" else np.nan, "status": status})
    return rows


def load_branches(path: str | Path) -> dict[str, np.ndarray]:
    """Flatten ``anchor_id/candidate_id`` groups into arrays."""
    anchor_ids, episode_ids, times, candidate_ids, types, actions, costs, status = [], [], [], [], [], [], [], []
    import h5py

    with h5py.File(path, "r") as store:
        for anchor_id in store:
            episode_id, t = int(anchor_id[2:6]), int(anchor_id[8:12])
            for candidate_id in store[anchor_id]:
                entry = store[anchor_id][candidate_id]
                anchor_ids.append(anchor_id)
                episode_ids.append(episode_id)
                times.append(t)
                candidate_ids.append(int(candidate_id))
                types.append(entry.attrs["candidate_type"])
                actions.append(entry["action_cmd"][()])
                costs.append(float(entry.attrs["J_m2"]))
                status.append(entry.attrs["rollout_status"])
    return {
        "anchor_id": np.array(anchor_ids), "episode_id": np.array(episode_ids), "t": np.array(times),
        "candidate_id": np.array(candidate_ids), "candidate_type": np.array(types),
        "action_cmd": np.stack(actions).astype(np.float32), "J_m2": np.array(costs), "rollout_status": np.array(status),
    }



# --------------------------------------------------------------------------- #
# closed-loop evaluation
# --------------------------------------------------------------------------- #
def closed_loop_episode(
    backend: Any, bundle: PolicyBundle, spec: ChunkSpec, delay: int, seed: int, init_state: Any | None,
    max_steps: int, gripper_dims: list[int], frames: list | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode with the shared scheduler.  Returns the episode record and one
    record per *executed* handover boundary."""
    scheduler = ChunkScheduler(spec.horizon, spec.execute_steps, delay)
    backend.reset(seed=seed, init_state=init_state)
    continuous = [dim for dim in range(bundle.meta["action_dim"]) if dim not in gripper_dims]
    previous_command, previous_position = None, np.asarray(backend.ee_position())
    boundaries, within_jumps, speeds, path = [], [], [], 0.0
    success, steps = False, 0
    if frames is not None:
        frames.append(backend.render_rgb())
    for k in range(max_steps):
        observation = backend.observe()["state"]
        command, record = scheduler.step(k, lambda: bundle.act(observation))
        command = np.asarray(command, dtype=np.float32)
        backend.step(command)
        position = np.asarray(backend.ee_position())
        displacement = float(np.linalg.norm(position - previous_position))
        path += displacement
        speeds.append(displacement)
        if previous_command is not None:
            jump_xyz = float(np.linalg.norm(command[continuous] - previous_command[continuous]))
            jump_grip = float(np.abs(command[gripper_dims] - previous_command[gripper_dims]).max())
            if record.is_boundary:
                boundaries.append(
                    {
                        "step": k, "chunk_id": record.chunk_id, "observed_at": record.observed_at,
                        "jump_xyz": jump_xyz, "jump_gripper": jump_grip,
                        "ee_speed_before_m": speeds[-2] if len(speeds) > 1 else 0.0, "ee_speed_after_m": displacement,
                    }
                )
            else:
                within_jumps.append(jump_xyz)
        previous_command, previous_position = command, position
        steps = k + 1
        if frames is not None:
            frames.append(backend.render_rgb())
        if backend.is_success():
            success = True
            break
    episode = {
        "seed": seed, "delay": delay, "success": bool(success), "steps": steps, "ee_path_m": path,
        "mean_ee_step_m": float(np.mean(speeds)), "num_boundaries": len(boundaries),
        "median_within_chunk_jump_xyz": float(np.median(within_jumps)) if within_jumps else np.nan,
    }
    return episode, boundaries


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap CI; resamples whole groups (episodes) when given."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if groups is None:
        samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        sums = np.array([values[groups == g].sum() for g in unique])
        counts = np.array([(groups == g).sum() for g in unique])
        picks = rng.integers(0, len(unique), size=(n_boot, len(unique)))
        samples = sums[picks].sum(1) / counts[picks].sum(1)
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
