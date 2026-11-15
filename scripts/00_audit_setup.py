"""Setup audit: runtime versions, data provenance, metadata and shape checks.

Writes ``<output_dir>/manifest.json``.  Exits non-zero if any gate fails.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from chunk_trajectory_error.config import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def nvidia_driver() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
        return {"nvidia_smi": "ok", "gpus": out.splitlines()}
    except FileNotFoundError:
        return {"nvidia_smi": "not found", "gpus": []}
    except Exception as error:  # report, never guess
        return {"nvidia_smi": f"failed: {error}", "gpus": []}


def runtime_report() -> dict[str, Any]:
    import torch

    report = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "os_system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {
            name: package_version(name)
            for name in ("mani_skill", "sapien", "gymnasium", "numpy", "h5py", "mplib", "imageio", "pytest")
        },
    }
    report.update(nvidia_driver())
    return report


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = ArgumentParser(description="Audit task, demo, controller, and runtime configuration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backend_name = config.get("environment", {}).get("backend")
    if backend_name != "maniskill":
        raise NotImplementedError(f"Setup audit not implemented for {backend_name}")

    from chunk_trajectory_error.backends.maniskill import (
        ManiSkillBackend,
        check_action_state_boundary,
        load_demo_episodes,
        load_demo_metadata,
    )

    env_cfg, data_cfg = config["environment"], config["data"]
    raw_h5 = Path(data_cfg["trajectory_path"])
    replay_h5 = Path(data_cfg["replayed_trajectory_path"])
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- files ------------------------------------------------------------- #
    files = {}
    for label, path in (("raw_h5", raw_h5), ("replayed_h5", replay_h5)):
        for kind, candidate in ((label, path), (label.replace("_h5", "_json"), path.with_suffix(".json"))):
            exists = candidate.is_file()
            check(f"file_exists:{kind}", exists, str(candidate))
            if exists:
                files[kind] = file_record(candidate)
    if not all(item["passed"] for item in checks):
        return finish(config, {"runtime": runtime_report(), "files": files, "checks": checks})

    # --- metadata ---------------------------------------------------------- #
    raw_meta, replay_meta = load_demo_metadata(raw_h5), load_demo_metadata(replay_h5)
    replay_env = replay_meta["env_info"]
    check("replayed.env_id", replay_env["env_id"] == env_cfg["task"], replay_env["env_id"])
    for key, cfg_key in (("control_mode", "control_mode"), ("obs_mode", "obs_mode"), ("sim_backend", "sim_backend")):
        value = replay_env["env_kwargs"].get(key)
        check(f"replayed.{key}", value == env_cfg[cfg_key], {"file": value, "config": env_cfg[cfg_key]})
    check("raw.env_id", raw_meta["env_info"]["env_id"] == env_cfg["task"], raw_meta["env_info"]["env_id"])

    # --- episodes: T actions / T+1 states --------------------------------- #
    episodes = load_demo_episodes(replay_h5)
    raw_episodes = {ep.episode_id: ep for ep in load_demo_episodes(raw_h5)}
    boundary_rows, boundary_ok = [], True
    for episode in episodes:
        try:
            check_action_state_boundary(episode.num_actions, episode.num_states)
            status = "ok"
        except ValueError as error:
            status, boundary_ok = str(error), False
        obs_len = None if episode.obs is None else int(episode.obs.shape[0])
        if obs_len is not None and obs_len != episode.num_actions + 1:
            status, boundary_ok = f"obs length {obs_len} != T+1", False
        raw = raw_episodes.get(episode.episode_id)
        boundary_rows.append(
            {
                "episode_id": episode.episode_id,
                "seed": episode.seed,
                "T_actions": episode.num_actions,
                "num_states": episode.num_states,
                "num_obs": obs_len,
                "raw_T_actions": None if raw is None else raw.num_actions,
                "raw_num_states": None if raw is None else raw.num_states,
                "control_mode": episode.control_mode,
                "success": episode.success,
                "status": status,
            }
        )
    check("T_actions_vs_T_plus_1_states(all episodes)", boundary_ok, f"{len(episodes)} episodes checked")
    check(
        "episode_control_mode",
        all(ep.control_mode == env_cfg["control_mode"] for ep in episodes),
        sorted({ep.control_mode for ep in episodes}),
    )
    check(
        "conversion_count",
        len(episodes) == len(raw_episodes),
        {"raw": len(raw_episodes), "replayed": len(episodes), "replayed_success": sum(bool(e.success) for e in episodes)},
    )
    chunk = config["chunk"]
    needed = chunk["history_steps"] + chunk["train_delay"] + chunk["evaluation_steps"]
    usable = [ep.episode_id for ep in episodes if ep.num_actions >= needed]
    check("episodes_long_enough_for_one_anchor(L+r+K)", len(usable) > 0, {"needed_steps": needed, "usable": len(usable)})

    # --- live environment -------------------------------------------------- #
    backend = ManiSkillBackend(config)
    sample = episodes[0]
    observation = backend.reset(seed=sample.seed, init_state=sample.state_at(0))
    env_info = backend.describe()
    live_state = backend.state_dict_numpy()
    file_state = sample.state_at(0)
    state_shapes = {
        group: {name: list(array.shape) for name, array in entries.items()}
        for group, entries in live_state.items()
        if group in ("actors", "articulations")
    }
    file_shapes = {
        group: {name: list(array.shape) for name, array in entries.items()} for group, entries in file_state.items()
    }
    check("state_tree_layout_matches_file", state_shapes == file_shapes, {"live": state_shapes, "file": file_shapes})
    action_dim = int(sample.actions.shape[1])
    live_action_dim = int(backend.base.single_action_space.shape[0])
    check("action_dim_matches_env", action_dim == live_action_dim, {"file": action_dim, "env": live_action_dim})
    obs_dim = int(observation["state"].shape[0])
    if sample.obs is not None:
        check("obs_dim_matches_env", int(sample.obs.shape[1]) == obs_dim, {"file": int(sample.obs.shape[1]), "env": obs_dim})
    check("control_mode_live", env_info["control_mode"] == env_cfg["control_mode"], env_info["control_mode"])
    check(
        "action_within_normalized_bounds",
        bool(abs(sample.actions).max() <= 1.0 + 1e-6),
        {"abs_max": float(abs(sample.actions).max())},
    )
    sample_shapes = {
        "episode_id": sample.episode_id,
        "actions": list(sample.actions.shape),
        "obs": None if sample.obs is None else list(sample.obs.shape),
        "env_states": file_shapes,
        "policy_observation_dim": obs_dim,
        "ee_position_xyz_m": list(backend.ee_position()),
        "policy_output_planned": [chunk["horizon"], action_dim],
        "error_net_inputs_planned": {
            "ee_history": [chunk["history_steps"], 3],
            "action_tail": [chunk["evaluation_steps"], action_dim],
        },
    }
    backend.close()

    return finish(
        config,
        {
            "runtime": runtime_report(),
            "environment": env_info,
            "files": files,
            "data_provenance": {
                "raw_source_type": raw_meta.get("source_type"),
                "raw_source_desc": raw_meta.get("source_desc"),
                "raw_env_kwargs": raw_meta["env_info"]["env_kwargs"],
                "replayed_env_kwargs": replay_env["env_kwargs"],
                "note": data_cfg.get("provenance_note"),
            },
            "episodes": boundary_rows,
            "sample_shapes": sample_shapes,
            "checks": checks,
        },
    )


def finish(config: dict[str, Any], body: dict[str, Any]) -> int:
    passed = all(item["passed"] for item in body["checks"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "all_checks_passed": passed,
        **body,
    }
    out_dir = Path(config["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    for item in body["checks"]:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}")
    print(f"manifest -> {out_path}")
    print("SETUP AUDIT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
