"""Phase B helpers of the sim-to-sim pilot: forward-only target logging and source-replay gap labels.

Hardware-faithful rule: ``collect_target_episode`` only resets the target (by seed) and runs it
forward; it never restores it.  ``gap_labels_for_episode`` restores the *source* simulator to the
logged state at t and executes the K logged *commanded* actions without delay:

    J_gap(t) = mean_{j=1..K} || p_target[t+j] - p_source[t+j] ||^2      (end effector, m^2)

with p_target[t+j] the logged position after the target applied its command at step t+j-1.
The cube deviation is stored the same way.  ``target_restore`` (used only by the sim-to-sim
diagnostics of ``13_audit_gap.py``) is the one place that restores a target.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from chunk_trajectory_error.pipeline import low_frequency_perturbation
from chunk_trajectory_error.scheduler import ChunkScheduler
from chunk_trajectory_error.shifts import LatencyBackend


def _inner(backend: Any) -> Any:
    return backend.inner if isinstance(backend, LatencyBackend) else backend


def cube_position(backend: Any) -> np.ndarray:
    inner = _inner(backend)
    if hasattr(inner, "cube_position"):  # test doubles
        return np.asarray(inner.cube_position(), np.float64).copy()
    base = inner.base
    return base.cube.pose.p[0].detach().cpu().numpy().astype(np.float64).copy()


def pending_commands(backend: Any, next_command: np.ndarray) -> np.ndarray:
    """Commands in flight before ``next_command`` is sent (part of the target state), [d, A]."""
    if isinstance(backend, LatencyBackend):
        return backend.in_flight(next_command)
    return np.zeros((0, len(next_command)), np.float32)


def collect_target_episode(
    target: Any, act: Callable[[np.ndarray], np.ndarray], horizon: int, execute_steps: int, seed: int, max_steps: int,
    gripper_dims: list[int], perturb_rng: np.random.Generator | None, amplitude_range: tuple[float, float],
    step_budget: int, state_fn: Callable[[], dict] | None = None,
) -> dict[str, Any]:
    """Run one target episode forward (reset by seed only, no restores) and log every step.

    ``act(obs) -> chunk [H, A]``.  With ``perturb_rng`` every executed chunk gets a bounded
    low-frequency perturbation (gripper untouched) before clipping to the action box.  The
    episode ends at success, at ``max_steps`` or when ``step_budget`` steps have been logged.
    """
    scheduler = ChunkScheduler(horizon, execute_steps, 0)
    target.reset(seed=seed)
    inner = _inner(target)
    state_fn = state_fn or inner.state_dict_numpy
    log: dict[str, list] = {k: [] for k in ("state", "obs", "pending", "commanded", "applied", "ee_before", "ee_after",
                                            "cube_before", "cube_after", "chunk_step")}
    chunks, success = [], False
    limit = min(max_steps, step_budget)
    for k in range(limit):
        observation = target.observe()["state"]
        log["state"].append(state_fn())
        log["obs"].append(observation.astype(np.float32))
        log["ee_before"].append(np.asarray(target.ee_position(), np.float64))
        log["cube_before"].append(cube_position(target))

        def predict() -> np.ndarray:
            chunk = np.asarray(act(observation), np.float32)
            if perturb_rng is not None:
                delta = low_frequency_perturbation(perturb_rng, chunk.shape[0], chunk.shape[1], gripper_dims, amplitude_range)
                chunk = np.clip(chunk + delta, -1.0, 1.0).astype(np.float32)
            chunks.append(chunk)
            return chunk

        command, record = scheduler.step(k, predict)
        command = np.asarray(command, np.float32)
        log["pending"].append(pending_commands(target, command).copy())
        target.step(command)
        applied = target.last_applied if isinstance(target, LatencyBackend) else command
        log["commanded"].append(command)
        log["applied"].append(np.asarray(applied, np.float32).copy())
        log["chunk_step"].append((record.chunk_id, record.chunk_index))
        log["ee_after"].append(np.asarray(target.ee_position(), np.float64))
        log["cube_after"].append(cube_position(target))
        if target.is_success():
            success = True
            break
    steps = len(log["commanded"])
    out = {key: value for key, value in log.items() if key != "state"}
    out = {key: np.asarray(value) for key, value in out.items()}
    out["pending"] = np.stack(log["pending"]) if steps else np.zeros((0, 0, 0), np.float32)
    out["state"] = stack_states(log["state"])
    out["chunks"] = np.stack(chunks) if chunks else np.zeros((0,), np.float32)
    out.update({"seed": int(seed), "steps": steps, "success": bool(success), "perturbed": perturb_rng is not None,
                "truncated_by_budget": bool(not success and steps == step_budget and step_budget < max_steps)})
    return out


def stack_states(states: list[dict]) -> Any:
    """List of nested state dicts -> nested dict of arrays with a leading time axis."""
    if not states:
        return {}
    first = states[0]
    if isinstance(first, dict):
        return {key: stack_states([s[key] for s in states]) for key in first}
    return np.stack([np.asarray(s) for s in states])


def index_state(states: Any, index: int) -> Any:
    if isinstance(states, dict):
        return {key: index_state(value, index) for key, value in states.items()}
    return np.array(states[index], copy=True)


def ee_at_states(episode: dict) -> np.ndarray:
    """EE position at every state index 0..T, [T + 1, 3]."""
    return np.concatenate([episode["ee_before"], episode["ee_after"][-1:]], axis=0)


def label_times(num_steps: int, history_steps: int, evaluation_steps: int) -> range:
    """t with a full history (t >= L) and K further logged steps (t + K <= T)."""
    return range(history_steps, num_steps - evaluation_steps + 1)


def history_features(ee_states: np.ndarray, t: int, steps: int) -> np.ndarray:
    """Relative EE positions over the previous L logged steps: p[t-L:t] - p[t], [L, 3]."""
    if t < steps:
        raise ValueError("not enough history")
    return (ee_states[t - steps : t] - ee_states[t]).astype(np.float32)


def replay_source(source: Any, state: Any, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Restore the source to ``state`` and execute ``actions`` (no delay).  EE and cube after every step."""
    source.restore(state)
    ee = np.full((len(actions), 3), np.nan)
    cube = np.full((len(actions), 3), np.nan)
    try:
        for j, action in enumerate(actions):
            source.step(action)
            ee[j] = source.ee_position()
            cube[j] = cube_position(source)
    except Exception as error:  # counted by the caller, never dropped silently
        return ee, cube, f"aborted:{type(error).__name__}"
    return ee, cube, "ok"


def gap_cost(p_target: np.ndarray, p_source: np.ndarray) -> float:
    return float(np.mean(np.sum((np.asarray(p_target, np.float64) - np.asarray(p_source, np.float64)) ** 2, axis=1)))


def gap_labels_for_episode(source: Any, episode: dict, episode_id: int, history_steps: int, evaluation_steps: int,
                           robot_slice: slice) -> list[dict]:
    """Source-replay labels for every valid t of one logged target episode."""
    rows = []
    ee_states = ee_at_states(episode)
    K = evaluation_steps
    for t in label_times(episode["steps"], history_steps, K):
        actions = episode["commanded"][t : t + K]
        p_source, c_source, status = replay_source(source, index_state(episode["state"], t), actions)
        ok = status == "ok"
        rows.append({
            "episode_id": episode_id, "t": t, "perturbed": episode["perturbed"], "status": status,
            "history": history_features(ee_states, t, history_steps),
            "robot": episode["obs"][t][robot_slice].astype(np.float32),
            "obs": episode["obs"][t].astype(np.float32),
            "actions": actions.astype(np.float32),
            "pending": episode["pending"][t].astype(np.float32),
            "J_gap_m2": gap_cost(episode["ee_after"][t : t + K], p_source) if ok else np.nan,
            "J_gap_cube_m2": gap_cost(episode["cube_after"][t : t + K], c_source) if ok else np.nan,
        })
    return rows


def roughness(actions: np.ndarray, continuous: list[int]) -> np.ndarray:
    """Mean squared difference of consecutive translation commands over the K steps, [N]."""
    actions = np.asarray(actions)
    diff = np.diff(actions[..., continuous], axis=-2)
    return np.mean(np.sum(diff**2, axis=-1), axis=-1)


def target_restore(target: Any, state: Any, pending: np.ndarray) -> None:
    """SIM-TO-SIM ONLY: restore a target (including its pending-command buffer).  Never used for data."""
    if isinstance(target, LatencyBackend):
        target.restore({"inner": state, "pending": np.asarray(pending, np.float32), "last_applied": None})
    else:
        target.restore(state)
