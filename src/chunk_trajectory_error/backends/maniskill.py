"""ManiSkill 3 adapter and demonstration reader.

Audited against mani_skill 3.0.1 / sapien 3.0.3 on ``physx_cpu`` with one
environment.  The heavy imports (torch, gymnasium, mani_skill) happen inside
``ManiSkillBackend`` so that the HDF5 reader and its tests only need h5py/numpy.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chunk_trajectory_error.contracts import Observation, SimulationBackend


# --------------------------------------------------------------------------- #
# Demonstration reader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoEpisode:
    """One demonstration.  ``env_states[i]`` is the state *before* ``actions[i]``."""

    episode_id: int
    traj_key: str
    seed: int | None
    control_mode: str
    actions: np.ndarray  # [T, action_dim]
    env_states: dict[str, Any]  # nested dict of arrays with leading dim T + 1
    obs: np.ndarray | None  # [T + 1, obs_dim] when the file stores state observations
    success: bool | None
    meta: dict[str, Any]

    @property
    def num_actions(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_states(self) -> int:
        return leading_length(self.env_states)

    def state_at(self, index: int) -> dict[str, Any]:
        """State dict (unbatched numpy leaves) observed before ``actions[index]``."""
        if not 0 <= index < self.num_states:
            raise IndexError(f"state index {index} outside [0, {self.num_states})")
        return _index_tree(self.env_states, index)


def _read_tree(node: Any) -> Any:
    import h5py

    if isinstance(node, h5py.Dataset):
        return node[()]
    return {key: _read_tree(node[key]) for key in node.keys()}


def _index_tree(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {key: _index_tree(value, index) for key, value in tree.items()}
    return np.array(tree[index], copy=True)


def leading_length(tree: Any) -> int:
    """Common leading length of every leaf; raises if leaves disagree."""
    lengths = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
        else:
            lengths.add(int(np.asarray(node).shape[0]))

    visit(tree)
    if len(lengths) != 1:
        raise ValueError(f"state leaves have inconsistent leading lengths: {sorted(lengths)}")
    return lengths.pop()


def check_action_state_boundary(num_actions: int, num_states: int) -> None:
    """T actions must come with exactly T + 1 states (state i precedes action i)."""
    if num_actions <= 0:
        raise ValueError("episode has no actions")
    if num_states != num_actions + 1:
        raise ValueError(
            f"expected T+1 states for T actions, got T={num_actions} actions and {num_states} states"
        )


def load_demo_metadata(h5_path: str | Path) -> dict[str, Any]:
    json_path = Path(h5_path).with_suffix(".json")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_demo_episodes(h5_path: str | Path, episode_ids: list[int] | None = None) -> list[DemoEpisode]:
    """Read ManiSkill ``traj_*`` groups together with the sibling JSON metadata."""
    import h5py

    meta = load_demo_metadata(h5_path)
    wanted = None if episode_ids is None else set(episode_ids)
    episodes: list[DemoEpisode] = []
    with h5py.File(h5_path, "r") as handle:
        for entry in meta["episodes"]:
            episode_id = int(entry["episode_id"])
            if wanted is not None and episode_id not in wanted:
                continue
            key = f"traj_{episode_id}"
            if key not in handle:
                raise KeyError(f"{key} listed in JSON but missing from {h5_path}")
            group = handle[key]
            seed = entry.get("episode_seed")
            if seed is None:
                seed = (entry.get("reset_kwargs") or {}).get("seed")
            if isinstance(seed, list):
                seed = seed[0]
            episodes.append(
                DemoEpisode(
                    episode_id=episode_id,
                    traj_key=key,
                    seed=None if seed is None else int(seed),
                    control_mode=str(entry.get("control_mode")),
                    actions=np.asarray(group["actions"][()], dtype=np.float32),
                    env_states=_read_tree(group["env_states"]),
                    obs=np.asarray(group["obs"][()]) if "obs" in group and hasattr(group["obs"], "shape") else None,
                    success=entry.get("success"),
                    meta=dict(entry),
                )
            )
    return episodes


# --------------------------------------------------------------------------- #
# Simulator adapter
# --------------------------------------------------------------------------- #
class ManiSkillBackend(SimulationBackend):
    """Single-environment ManiSkill adapter (``physx_cpu``).

    ``snapshot`` / ``restore`` use ``get_state_dict`` / ``set_state_dict`` and also
    carry the controller state.  ``set_state_dict`` does not restore the episode
    step counter or PD drive targets; the first ``step`` after a restore rewrites
    the drive targets, and the replay audit is what certifies this is sufficient.
    """

    def __init__(self, config: dict[str, Any], render: bool = False) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers the tasks)
        import torch

        self._torch = torch
        self.config = config
        env_cfg = config["environment"]
        if env_cfg.get("sim_backend") != "physx_cpu":
            raise ValueError("this adapter is audited for sim_backend=physx_cpu only")
        self.env = gym.make(
            env_cfg["task"],
            num_envs=1,
            obs_mode=env_cfg["obs_mode"],
            control_mode=env_cfg["control_mode"],
            sim_backend=env_cfg["sim_backend"],
            render_mode="rgb_array" if render else None,
            max_episode_steps=env_cfg.get("max_episode_steps"),
        )
        self.base = self.env.unwrapped
        self._obs: Any = None

    # -- helpers ------------------------------------------------------------ #
    def _to_batched_tensor_tree(self, tree: Any) -> Any:
        if isinstance(tree, dict):
            return {key: self._to_batched_tensor_tree(value) for key, value in tree.items()}
        tensor = self._torch.as_tensor(np.asarray(tree), dtype=self._torch.float32)
        expected_ndim = 2
        return tensor[None] if tensor.ndim < expected_ndim else tensor

    def state_dict_numpy(self) -> dict[str, Any]:
        """Current simulator state as unbatched numpy arrays (same layout as the HDF5)."""

        def convert(tree: Any) -> Any:
            if isinstance(tree, dict):
                return {key: convert(value) for key, value in tree.items()}
            return tree.detach().cpu().numpy()[0].copy()

        return convert(self.base.get_state_dict())

    # -- contract ----------------------------------------------------------- #
    def reset(self, seed: int | None = None, init_state: Any | None = None) -> Observation:
        obs, _ = self.env.reset(seed=seed)
        if init_state is not None:
            self.base.set_state_dict(self._to_batched_tensor_tree(init_state))
            obs = self.base.get_obs()
        self._obs = obs
        return self.observe()

    def observe(self) -> Observation:
        obs = self.base.get_obs()
        return {"state": obs.detach().cpu().numpy()[0].copy()}

    def snapshot(self) -> Any:
        sim_state = {
            key: value
            for key, value in self.base.get_state_dict().items()
            if key in ("actors", "articulations")
        }
        sim_state = copy.deepcopy(sim_state)
        return {
            "sim_state": sim_state,
            "controller_state": copy.deepcopy(self.base.agent.get_controller_state()),
        }

    def restore(self, snapshot: Any) -> None:
        if "sim_state" in snapshot:
            sim_state, controller_state = snapshot["sim_state"], snapshot.get("controller_state")
        else:  # a raw state dict, e.g. one row read from the demonstration HDF5
            sim_state, controller_state = snapshot, None
        sim_state = {k: v for k, v in sim_state.items() if k in ("actors", "articulations")}
        self.base.set_state_dict(self._to_batched_tensor_tree(sim_state))
        if controller_state:
            self.base.agent.set_controller_state(controller_state)

    def step(self, env_action: Any) -> Observation:
        action = self._torch.as_tensor(np.asarray(env_action, dtype=np.float32))[None]
        self._obs, _, _, _, _ = self.env.step(action)
        return self.observe()

    def ee_position(self) -> tuple[float, float, float]:
        position = self.base.agent.tcp.pose.p[0].detach().cpu().numpy()
        return (float(position[0]), float(position[1]), float(position[2]))

    def is_success(self) -> bool:
        return bool(self.base.evaluate()["success"][0].item())

    # -- extras used by the audits ----------------------------------------- #
    def render_rgb(self) -> np.ndarray:
        frame = self.env.render()
        return frame[0].detach().cpu().numpy().astype(np.uint8)

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.base.spec.id if self.base.spec else self.config["environment"]["task"],
            "control_mode": self.base.control_mode,
            "obs_mode": self.base.obs_mode,
            "sim_backend": self.config["environment"]["sim_backend"],
            "physx_device": str(self.base.device),
            "num_envs": int(self.base.num_envs),
            "control_freq_hz": int(self.base.control_freq),
            "sim_freq_hz": int(self.base.sim_freq),
            "single_action_space": repr(self.env.unwrapped.single_action_space),
            "single_observation_shape": list(self.env.unwrapped.single_observation_space.shape),
            "controller_class": type(self.base.agent.controller).__name__,
            "controller_state_keys": sorted(self.base.agent.get_controller_state().keys()),
            "robot": type(self.base.agent).__name__,
        }

    def close(self) -> None:
        self.env.close()
