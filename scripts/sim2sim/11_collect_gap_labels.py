"""Phase B, step 1: forward-only target rollouts and source-replay gap labels.

Target = source physics + one shift (Phase A choice), applied once.  The frozen ``pi_bc``
policy runs closed loop (r = 0) from the seeds of the split's demonstration episodes (reset by
seed only); every second episode adds a bounded low-frequency perturbation to each executed
chunk.  The target is never restored.  Collection stops exactly at the step budget.

Labels: for every logged step t with t >= L and K further logged steps, the *source* simulator is
restored to the logged state at t and executes the K logged commanded actions without delay;
J_gap = mean_j ||p_target[t+j] - p_source[t+j]||^2 (end effector) and the same for the cube.

Outputs (``runs/sim2sim_pilot/<shift>/``): ``target_log_<split>.h5``, ``labels_<split>.npz``,
``collection_<split>.json``; the run is recorded under ``phase_b`` in ``manifest.json``.

``--shift identity`` runs the same procedure in an unshifted "target" (diagnostic noise floor of
the source restore; not part of any budget).
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.shifts import ShiftSpec

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


SPLIT_OFFSET = {"train": 0, "val": 1}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shift_from_entry(entry: dict) -> ShiftSpec:
    return load_script("10_shift_scan.py").shift_from_entry(entry)


def write_h5(path: Path, episodes: list[dict], attrs: dict) -> None:
    import h5py

    def put(group, name, value):
        if isinstance(value, dict):
            sub = group.create_group(name)
            for key, item in value.items():
                put(sub, key, item)
        else:
            group.create_dataset(name, data=np.asarray(value), compression="gzip", compression_opts=4)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        store.attrs.update({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in attrs.items()})
        for episode in episodes:
            group = store.create_group(f"ep_{episode['episode_id']:04d}")
            for key in ("seed", "steps", "success", "perturbed", "truncated_by_budget", "episode_id"):
                group.attrs[key] = episode[key]
            group.attrs["chunk_step_columns"] = "chunk_id,chunk_index"
            for key in ("obs", "pending", "commanded", "applied", "ee_before", "ee_after", "cube_before", "cube_after",
                        "chunk_step", "chunks"):
                put(group, key, episode[key])
            put(group, "env_states", episode["state"])


def label_worker(base_config_path: str, episodes: list[dict], spec_tuple: tuple[int, int], robot_slice: tuple[int, int],
                 first_seed: int) -> list[dict]:
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.sim2sim import gap_labels_for_episode

    config = load_config(base_config_path)
    source = ManiSkillBackend(config)
    source.reset(seed=first_seed)  # the env must be reset once before any restore
    rows = []
    for episode in episodes:
        rows += gap_labels_for_episode(source, episode, episode["episode_id"], spec_tuple[0], spec_tuple[1], slice(*robot_slice))
    source.close()
    return rows


def quantiles(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {"n": int(len(values)), "mean": float(values.mean()), "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(values.max()),
            "exact_zeros": int((values == 0).sum())}


def update_manifest(run_root: Path, key: str, payload: dict) -> None:
    path = run_root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    manifest.setdefault("phase_b", {})[key] = payload
    path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")


def main() -> int:
    parser = ArgumentParser(description="Phase B: forward-only target logging and source-replay gap labels")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--shift", choices=("phase_b", "identity"), default="phase_b")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from chunk_trajectory_error.pipeline import PolicyBundle, chunk_spec, load_demoset
    from chunk_trajectory_error.shifts import make_target_backend
    from chunk_trajectory_error.sim2sim import collect_target_episode

    pilot = load_config(args.config)
    phase = pilot["phase_b"]
    base_config = load_config(pilot["base_config"])
    spec = chunk_spec(base_config, 0)
    demos = load_demoset(base_config)
    if args.shift == "identity":
        shift, budget, tag = ShiftSpec("nominal"), int(phase["noise_floor"]["steps"]), "identity"
    else:
        shift, budget, tag = shift_from_entry(phase["shift"]), int(phase["budgets"][f"{args.split}_steps"]), shift_from_entry(phase["shift"]).name
    run_root = Path(pilot["output_dir"])
    out = run_root / tag
    out.mkdir(parents=True, exist_ok=True)
    seed = int(phase["seed"]) + 1000 * SPLIT_OFFSET[args.split]
    rng = np.random.default_rng(seed)
    amplitude = tuple(float(x) for x in base_config["branches"]["perturbation_amplitude"])
    bundle = PolicyBundle.load(phase["policy_checkpoint"])
    max_steps = int(base_config["environment"]["max_episode_steps"])

    started = time.time()
    target, shift_record = make_target_backend(base_config, shift, demos.gripper_dims, first_seed=int(phase["first_reset_seed"]))
    episodes, used = [], 0
    for order, episode_id in enumerate(demos.ids(args.split)):
        if used >= budget:
            break
        perturbed = order % 2 == 1  # every second episode is perturbed
        episode = collect_target_episode(
            target, bundle.act, spec.horizon, spec.execute_steps, demos.episodes[episode_id].seed, max_steps,
            demos.gripper_dims, rng if perturbed else None, amplitude, budget - used,
        )
        episode["episode_id"] = int(episode_id)
        np.testing.assert_allclose(episode["ee_before"][1:], episode["ee_after"][:-1], atol=0)  # logging consistency
        episodes.append(episode)
        used += episode["steps"]
    from chunk_trajectory_error.shifts import verify_shift

    shift_record["after_collection"] = verify_shift(target, shift_record)
    target.close()
    collect_seconds = time.time() - started
    if used != budget:
        raise RuntimeError(f"split exhausted: {used} of {budget} steps")
    write_h5(out / f"target_log_{args.split}.h5", episodes,
             {"shift": shift.name, "split": args.split, "budget_steps": budget, "seed": seed,
              "policy": phase["policy_checkpoint"], "target_restores": 0, "perturbation_amplitude": list(amplitude)})

    started_labels = time.time()
    chunks = [episodes[i :: args.workers] for i in range(args.workers)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(label_worker, pilot["base_config"], chunk, (spec.history_steps, spec.evaluation_steps),
                               tuple(base_config["observation"]["robot_state_slice"]), int(phase["first_reset_seed"]))
                   for chunk in chunks if chunk]
        for future in futures:
            rows += future.result()
    rows.sort(key=lambda r: (r["episode_id"], r["t"]))
    label_seconds = time.time() - started_labels
    arrays = {key: np.stack([r[key] for r in rows]) for key in ("history", "robot", "obs", "actions", "pending")}
    arrays.update({key: np.array([r[key] for r in rows]) for key in ("episode_id", "t", "perturbed", "status", "J_gap_m2", "J_gap_cube_m2")})
    np.savez_compressed(out / f"labels_{args.split}.npz", **arrays)

    ok = arrays["status"] == "ok"
    summary = {
        "shift": shift.name, "split": args.split, "seed": seed, "budget_steps": budget, "target_steps_used": int(used),
        "episodes": len(episodes), "episodes_perturbed": int(sum(e["perturbed"] for e in episodes)),
        "episodes_success": int(sum(e["success"] for e in episodes)),
        "success_unperturbed": [int(sum(e["success"] for e in episodes if not e["perturbed"])), int(sum(not e["perturbed"] for e in episodes))],
        "success_perturbed": [int(sum(e["success"] for e in episodes if e["perturbed"])), int(sum(e["perturbed"] for e in episodes))],
        "episodes_truncated_by_budget": int(sum(e["truncated_by_budget"] for e in episodes)),
        "episode_ids": [int(e["episode_id"]) for e in episodes],
        "label_samples": int(len(rows)), "label_samples_ok": int(ok.sum()), "label_samples_aborted": int((~ok).sum()),
        "steps_without_label": int(used - len(rows)),
        "source_steps_for_labels": int(len(rows) * spec.evaluation_steps),
        "target_restores": 0,
        "J_gap_m2": quantiles(arrays["J_gap_m2"]), "J_gap_cube_m2": quantiles(arrays["J_gap_cube_m2"]),
        "sqrtJ_gap_mm": quantiles(np.sqrt(arrays["J_gap_m2"]) * 1e3),
        "J_gap_m2_unperturbed": quantiles(arrays["J_gap_m2"][~arrays["perturbed"]]),
        "J_gap_m2_perturbed": quantiles(arrays["J_gap_m2"][arrays["perturbed"]]),
        "pending_logged": {"shape_per_step": list(episodes[0]["pending"].shape[1:]), "note": "commands sent but not yet applied, before the step"},
        "controller_state_keys": [], "seconds_collection": collect_seconds, "seconds_labels": label_seconds,
    }
    (out / f"collection_{args.split}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    audit = load_script("00_audit_setup.py")
    files = [out / f"target_log_{args.split}.h5", out / f"labels_{args.split}.npz", Path(phase["policy_checkpoint"]), Path(args.config),
             Path(__file__), ROOT / "src/chunk_trajectory_error/sim2sim.py", ROOT / "src/chunk_trajectory_error/shifts.py"]
    update_manifest(run_root, f"11_{tag}_{args.split}", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": {k: v for k, v in summary.items() if k != "episode_ids"},
        "shift_record": shift_record, "files": [audit.file_record(p) for p in files], "runtime": audit.runtime_report(),
    })
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_ids"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
