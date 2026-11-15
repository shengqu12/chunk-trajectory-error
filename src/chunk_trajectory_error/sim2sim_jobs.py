"""Picklable process-pool workers for the sim-to-sim pilot (Phase C and its diagnostics).

* ``gap_rollout_worker`` -- SIM-TO-SIM ONLY: executes action sequences from a logged state in the
  source and in the *restored* target and returns the true gap (same logic as 13_audit_gap.py).
* ``closed_loop_worker`` -- runs policies closed loop (forward only, reset by seed) in the target or
  the source and returns one record per episode.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np

from chunk_trajectory_error.config import load_config


def _shift(entry: dict):
    from chunk_trajectory_error.shifts import ShiftSpec

    return ShiftSpec(family=entry["family"], stiffness_scale=float(entry.get("s", 1.0)), delay_steps=int(entry.get("d", 0)),
                     friction_scale=float(entry.get("f", 1.0)), mass_scale=float(entry.get("m", 1.0)),
                     contact_mode=entry.get("contact_mode", "cube_only"))


def make_envs(pilot: dict, max_steps: int | None = None, which: tuple[str, ...] = ("source", "target")) -> dict:
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.shifts import make_target_backend

    base_config = load_config(pilot["base_config"])
    if max_steps is not None:
        base_config["environment"]["max_episode_steps"] = int(max_steps)
    first = int(pilot["phase_b"]["first_reset_seed"])
    envs = {}
    if "source" in which:
        envs["source"] = ManiSkillBackend(base_config)
        envs["source"].reset(seed=first)
    if "target" in which:
        envs["target"], envs["target_record"] = make_target_backend(base_config, _shift(pilot["phase_c"]["shift"]),
                                                                    list(base_config["action"]["gripper_dims"]), first_seed=first)
    return envs


def gap_rollout_worker(pilot_path: str, jobs: list[dict]) -> list[dict]:
    """SIM-TO-SIM ONLY.  job = {job, state, pending, actions}."""
    from chunk_trajectory_error.sim2sim import gap_cost, replay_source, target_restore

    envs = make_envs(load_config(pilot_path))
    source, target = envs["source"], envs["target"]
    results = []
    for job in jobs:
        p_source, _, s_status = replay_source(source, job["state"], job["actions"])
        target_restore(target, job["state"], job["pending"])  # SIM-TO-SIM ONLY
        p_target = np.full_like(p_source, np.nan)
        t_status = "ok"
        try:
            for j, action in enumerate(job["actions"]):
                target.step(action)
                p_target[j] = target.ee_position()
        except Exception as error:  # counted by the caller
            t_status = f"aborted:{type(error).__name__}"
        ok = s_status == "ok" and t_status == "ok"
        results.append({"job": job["job"], "gap_m2": gap_cost(p_target, p_source) if ok else np.nan,
                        "status": "ok" if ok else f"{s_status}|{t_status}"})
    from chunk_trajectory_error.shifts import verify_shift

    verify_shift(target, envs["target_record"])  # the shift survived every restore
    source.close()
    target.close()
    return results


def closed_loop_worker(pilot_path: str, env_name: str, max_steps: int, jobs: list[dict]) -> list[dict]:
    """job = {policy_path, tags (dict copied into every record), seeds (list[int])}.  Forward only."""
    from chunk_trajectory_error.pipeline import PolicyBundle, chunk_spec, closed_loop_episode
    from chunk_trajectory_error.shifts import verify_shift

    pilot = load_config(pilot_path)
    base_config = load_config(pilot["base_config"])
    gripper_dims = list(base_config["action"]["gripper_dims"])
    spec = chunk_spec(base_config, 0)
    envs = make_envs(pilot, max_steps, (env_name,))
    backend = envs[env_name]
    rows = []
    for job in jobs:
        bundle = PolicyBundle.load(job["policy_path"])
        for seed in job["seeds"]:
            try:
                record, _ = closed_loop_episode(backend, bundle, spec, 0, int(seed), None, max_steps, gripper_dims)
                record["status"] = "ok"
            except Exception as error:  # counted, never dropped silently
                record = {"seed": int(seed), "success": False, "steps": np.nan, "mean_ee_step_m": np.nan,
                          "status": f"aborted:{type(error).__name__}:{error}"}
            record["init_seed"] = int(seed)  # the job tags may carry a different "seed" (the policy seed)
            record.update(job["tags"])
            record["env"] = env_name
            rows.append(record)
    if env_name == "target":
        verify_shift(backend, envs["target_record"])
    backend.close()
    return rows


def run_pool(fn: Callable, argument_lists: list[tuple], workers: int) -> list[Any]:
    """Run ``fn(*args)`` for every tuple in a spawn pool; returns the results in order."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(fn, *args) for args in argument_lists]
        return [future.result() for future in futures]


def run_gap_jobs(pilot_path: str, jobs: list[dict], workers: int) -> dict[int, dict]:
    chunks = [jobs[i::workers] for i in range(workers)]
    out = {}
    for rows in run_pool(gap_rollout_worker, [(pilot_path, c) for c in chunks if c], workers):
        for row in rows:
            out[row["job"]] = row
    return out
