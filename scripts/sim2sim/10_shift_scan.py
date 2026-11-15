"""Phase A of the sim-to-sim pilot: find physics shifts that create a success gap (gate 1).

The frozen ``pi_bc`` policy is run closed loop (shared ChunkScheduler, r = 0) from the same
150 initial states as ``07_eval_closed_loop.py`` (100 test episodes + 50 fresh seeds) in the
nominal environment and in each shifted target.  Every condition runs in its own process
with a freshly created environment; the shift is applied once after the first reset and
verified after the next reset and again after the last episode.  The target is only run
forward (episode resets only, no mid-episode restores).

Outputs (``runs/sim2sim_pilot/``): ``shift_scan.csv``, ``shift_scan_episodes.parquet``,
``manifest.json``, ``logs/10_shift_scan.log``.

Gate 1: per family, the mildest level whose paired success drop vs nominal is at least
``gate1.min_drop_pp`` points with a bootstrap 95% interval that excludes zero.
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
import pandas as pd

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.shifts import ShiftSpec

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shift_from_entry(entry: dict) -> ShiftSpec:
    family = entry["family"]
    return ShiftSpec(
        family=family,
        stiffness_scale=float(entry.get("s", 1.0)),
        delay_steps=int(entry.get("d", 0)),
        friction_scale=float(entry.get("f", 1.0)),
        mass_scale=float(entry.get("m", 1.0)),
        contact_mode=entry.get("contact_mode", "cube_only"),
    )


def initial_states(base_config: dict, demos) -> list[tuple[str, int, object]]:
    """Exactly the list built by 07_eval_closed_loop.py."""
    inits = [("test_episode", demos.episodes[i].seed, demos.episodes[i].state_at(0)) for i in demos.ids("test")]
    fresh = base_config["evaluation"]["fresh_seeds"]
    inits += [("fresh_seed", fresh["start"] + i, None) for i in range(fresh["count"])]
    return inits


def run_condition(pilot_path: str, entry: dict) -> dict:
    """Worker: one fresh target environment, shift applied once, 150 episodes."""
    from chunk_trajectory_error.pipeline import PolicyBundle, chunk_spec, closed_loop_episode, load_demoset
    from chunk_trajectory_error.shifts import make_target_backend, verify_shift

    pilot = load_config(pilot_path)
    base_config = load_config(pilot["base_config"])
    phase = pilot["phase_a"]
    shift = shift_from_entry(entry)
    demos = load_demoset(base_config)
    bundle = PolicyBundle.load(phase["policy_checkpoint"])
    spec = chunk_spec(base_config, phase["delay"])
    max_steps = base_config["environment"]["max_episode_steps"]
    started = time.time()
    backend, record = make_target_backend(base_config, shift, demos.gripper_dims, first_seed=phase["first_reset_seed"])
    rows = []
    for init_index, (source, seed, init_state) in enumerate(initial_states(base_config, demos)):
        try:
            episode, _ = closed_loop_episode(backend, bundle, spec, phase["delay"], seed, init_state, max_steps, demos.gripper_dims)
            episode["status"] = "ok"
        except Exception as error:  # counted, never dropped silently
            episode = {"seed": seed, "delay": phase["delay"], "success": False, "steps": np.nan, "mean_ee_step_m": np.nan,
                       "status": f"aborted:{type(error).__name__}:{error}"}
        episode.update({"condition": shift.name, "family": shift.family, "init_index": init_index, "init_source": source})
        rows.append(episode)
    record["after_last_episode"] = verify_shift(backend, record)
    backend.close()
    target_steps = int(np.nansum([row["steps"] for row in rows]))
    return {"shift": shift.name, "entry": entry, "record": record, "rows": rows, "seconds": time.time() - started,
            "target_steps": target_steps}


def summarize(results: dict[str, dict], order: list[str], diagnostic: set[str], gate_cfg: dict, boot: dict) -> tuple[pd.DataFrame, dict]:
    from chunk_trajectory_error.pipeline import bootstrap_ci

    wilson = load_script("08_report.py").wilson
    nominal = pd.DataFrame(results["nominal"]["rows"]).set_index("init_index").sort_index()
    table = []
    for name in order:
        result = results[name]
        frame = pd.DataFrame(result["rows"]).set_index("init_index").sort_index()
        if not frame.index.equals(nominal.index) or not (frame["seed"] == nominal["seed"]).all():
            raise RuntimeError(f"{name}: initial states are not paired with nominal")
        success = frame["success"].astype(float).to_numpy()
        n, wins = len(success), int(success.sum())
        low, high = wilson(wins, n)
        row = {
            "condition": name, "family": result["entry"]["family"],
            "s": result["entry"].get("s"), "d": result["entry"].get("d"),
            "f": result["entry"].get("f"), "m": result["entry"].get("m"),
            "contact_mode": result["entry"].get("contact_mode", "cube_only") if result["entry"]["family"] == "contact" else None,
            "role": "diagnostic" if name in diagnostic else ("reference" if name == "nominal" else "scan"),
            "n": n, "successes": wins, "success_rate": wins / n, "wilson_low": low, "wilson_high": high,
            "mean_episode_steps": float(frame["steps"].mean()),
            "mean_episode_steps_success": float(frame.loc[frame["success"], "steps"].mean()) if wins else np.nan,
            "mean_ee_step_mm": float(frame["mean_ee_step_m"].mean() * 1e3),
            "aborted": int((frame["status"] != "ok").sum()),
        }
        if name == "nominal":
            row.update({"paired_diff_pp": 0.0, "paired_ci_low_pp": 0.0, "paired_ci_high_pp": 0.0,
                        "only_shift_succeeds": 0, "only_nominal_succeeds": 0})
        else:
            diff = success - nominal["success"].astype(float).to_numpy()
            mean, ci_low, ci_high = bootstrap_ci(diff, n_boot=boot["n_boot"], seed=boot["seed"])
            row.update({"paired_diff_pp": 100 * mean, "paired_ci_low_pp": 100 * ci_low, "paired_ci_high_pp": 100 * ci_high,
                        "only_shift_succeeds": int((diff > 0).sum()), "only_nominal_succeeds": int((diff < 0).sum())})
        row["meets_gate1"] = bool(name != "nominal" and row["paired_diff_pp"] <= -gate_cfg["min_drop_pp"] and row["paired_ci_high_pp"] < 0)
        table.append(row)
    frame = pd.DataFrame(table)

    def choose(rows: pd.DataFrame) -> dict:
        passing = rows[rows["meets_gate1"]]
        return {"levels_in_order": rows["condition"].tolist(),
                "chosen": None if passing.empty else passing.iloc[0]["condition"],
                "passes": not passing.empty}

    gate = {"rule": f"per family, mildest level with paired success drop >= {gate_cfg['min_drop_pp']:g} pp and bootstrap 95% CI "
                    "of the paired difference entirely below 0; a family with no such level is dropped",
            "families": {}}
    scan = frame[frame["role"] == "scan"]
    for family in gate_cfg["families"]:
        gate["families"][family] = choose(scan[scan["family"] == family])
    diag = frame[frame["role"] == "diagnostic"]
    if not diag.empty:
        gate["diagnostic_contact_shared_in_place"] = choose(diag)
    gate["any_family_passes"] = any(v["passes"] for v in gate["families"].values())
    frame["chosen_gate1"] = frame["condition"].isin([v["chosen"] for v in gate["families"].values() if v["chosen"]])
    return frame, gate


def main() -> int:
    parser = ArgumentParser(description="Phase A shift scan (gate 1)")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    pilot = load_config(args.config)
    phase = pilot["phase_a"]
    run_dir = Path(pilot["output_dir"])
    for protected in ("runs/maniskill_pickcube", "runs/maniskill_pickcube_official"):
        if run_dir.resolve() == Path(protected).resolve():
            raise RuntimeError(f"refusing to write into {protected}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    entries = list(phase["conditions"]) + list(phase.get("diagnostic_conditions", []))
    names = [shift_from_entry(e).name for e in entries]
    if len(set(names)) != len(names) or names[0] != "nominal":
        raise ValueError("condition names must be unique and start with nominal")
    diagnostic = {shift_from_entry(e).name for e in phase.get("diagnostic_conditions", [])}

    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers or len(entries), mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_condition, args.config, entry): name for entry, name in zip(entries, names)}
        for future, name in futures.items():
            results[name] = future.result()
            r = results[name]
            wins = sum(row["success"] for row in r["rows"])
            print(f"{name}: {wins}/{len(r['rows'])} success, {r['target_steps']} target steps, {r['seconds']:.0f}s", flush=True)

    table, gate = summarize(results, names, diagnostic, pilot["gate1"], phase["bootstrap"])
    table.to_csv(run_dir / "shift_scan.csv", index=False, float_format="%.6g")
    pd.DataFrame([row for name in names for row in results[name]["rows"]]).to_parquet(run_dir / "shift_scan_episodes.parquet", index=False)

    audit = load_script("00_audit_setup.py")
    base_config = load_config(pilot["base_config"])
    files = [Path(args.config), Path(pilot["base_config"]), Path(phase["policy_checkpoint"]),
             Path(base_config["data"]["replayed_trajectory_path"]), Path(base_config["data"]["split_manifest"]),
             Path(__file__), ROOT / "src/chunk_trajectory_error/shifts.py"]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "A",
        "spec": "docs/SIM2SIM_PILOT_SPEC.md",
        "runtime": audit.runtime_report(),
        "files": [audit.file_record(path) for path in files],
        "policy": {"checkpoint": phase["policy_checkpoint"], "extra": _policy_extra(phase["policy_checkpoint"])},
        "chunk": base_config["chunk"], "scheduler_delay": phase["delay"],
        "environment": base_config["environment"],
        "initial_states": {"test_episodes": len(_test_ids(base_config)), "fresh_seeds": base_config["evaluation"]["fresh_seeds"],
                           "same_as": "scripts/07_eval_closed_loop.py"},
        "seeds": {"experiment_seed": base_config["experiment"]["seed"], "first_reset_seed": phase["first_reset_seed"],
                  "bootstrap": phase["bootstrap"]},
        "shift_application": "once per condition, on a freshly created environment after its first reset; verified after the next "
                             "reset and after the last episode; one process per condition",
        "conditions": {name: {"entry": results[name]["entry"], "role": "diagnostic" if name in diagnostic else "scan",
                              "seconds": results[name]["seconds"], "target_steps": results[name]["target_steps"],
                              "aborted_episodes": sum(row["status"] != "ok" for row in results[name]["rows"]),
                              "shift_record": results[name]["record"]} for name in names},
        "target_steps_total": int(sum(r["target_steps"] for r in results.values())),
        "target_restores": 0,
        "gate1": gate,
        "wall_seconds": time.time() - started,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    columns = ["condition", "role", "successes", "n", "wilson_low", "wilson_high", "paired_diff_pp", "paired_ci_low_pp",
               "paired_ci_high_pp", "mean_episode_steps", "aborted", "meets_gate1"]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(gate, indent=2))
    print("GATE 1:", "PASS" if gate["any_family_passes"] else "FAIL (no family passes)")
    return 0


def _policy_extra(path: str) -> dict:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("extra", {})


def _test_ids(base_config: dict) -> list:
    return json.loads(Path(base_config["data"]["split_manifest"]).read_text())["splits"]["test"]


if __name__ == "__main__":
    raise SystemExit(main())
