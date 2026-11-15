"""Phase C, step 2: final evaluation of the selected policies.

* Closed loop, r = 0, from the 150 final seeds (200000-200149, reset by seed only, never used for
  selection) in the target and in the source; one run with a 400-step cap per episode gives both
  success@200 (primary: success within 200 steps) and success@400 (secondary).
* Consistency check: pi_bc in the 400-cap target on the 50 Phase A fresh seeds must reproduce the
  Phase A latency_d1 outcomes within 200 steps exactly.
* SIM-TO-SIM ONLY: true gap of each policy's own chunk (first K actions at the logged observation) on
  500 held-out logged target states (Phase B val logs), source and target restored from the log.

Outputs (``runs/sim2sim_pilot/<shift>/phase_c/``): ``eval_episodes.parquet``, ``true_gap_sim2sim_only.parquet``,
``eval_summary.json``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import PolicyBundle, chunk_spec

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


METHODS = ("zero_shot", "gap_demo", "gap_logged", "finetune", "smooth")  # latency default; phase_c.methods overrides


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = ArgumentParser(description="Phase C final evaluation")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--workers", type=int, default=22)
    args = parser.parse_args()
    from chunk_trajectory_error.sim2sim_jobs import closed_loop_worker, run_gap_jobs, run_pool

    pilot = load_config(args.config)
    phase = pilot["phase_c"]
    shift_name = load_script("10_shift_scan.py").shift_from_entry(phase["shift"]).name
    root = Path(pilot["output_dir"]) / shift_name
    out = root / "phase_c"
    selection = json.loads((out / "selection.json").read_text())
    final = phase["final_eval"]
    seeds = list(range(final["seeds"]["start"], final["seeds"]["start"] + final["seeds"]["count"]))
    max_steps = int(final["max_steps"])
    started = time.time()

    policies = [(int(s), m, selection["seeds"][str(s)][m]["checkpoint"]) for s in phase["seeds"] for m in phase.get("methods", METHODS)]
    halves = [seeds[: len(seeds) // 2], seeds[len(seeds) // 2 :]]
    jobs = [(args.config, env, max_steps, [{"policy_path": path, "seeds": half, "tags": {"seed": s, "method": m}}])
            for env in final["envs"] for (s, m, path) in policies for half in halves]
    fresh = load_config(pilot["base_config"])["evaluation"]["fresh_seeds"]
    check_seeds = list(range(fresh["start"], fresh["start"] + fresh["count"]))
    jobs.append((args.config, "target", max_steps, [{"policy_path": pilot["phase_b"]["policy_checkpoint"], "seeds": check_seeds,
                                                     "tags": {"seed": -1, "method": "consistency_pi_bc"}}]))
    rows = [row for result in run_pool(closed_loop_worker, jobs, args.workers) for row in result]
    frame = pd.DataFrame(rows)
    frame["success_400"] = frame["success"].astype(bool)
    frame["success_200"] = frame["success_400"] & (frame["steps"] <= final["primary_cap"])
    check = frame[frame["method"] == "consistency_pi_bc"].sort_values("init_seed")
    frame = frame[frame["method"] != "consistency_pi_bc"]
    frame.to_parquet(out / "eval_episodes.parquet", index=False)

    phase_a = pd.read_parquet(Path(pilot["output_dir"]) / "shift_scan_episodes.parquet")
    phase_a = phase_a[(phase_a["condition"] == shift_name) & (phase_a["init_source"] == "fresh_seed")].sort_values("seed")
    consistency = {
        "episodes": int(len(check)),
        "phase_a_success": int(phase_a["success"].sum()), "rerun_success_200": int(check["success_200"].sum()),
        "same_outcome_per_episode": bool((check["success_200"].to_numpy() == phase_a["success"].to_numpy()).all()),
        "same_steps_where_success": bool((check.loc[check["success_200"], "steps"].to_numpy() == phase_a.loc[phase_a["success"], "steps"].to_numpy()).all())
        if check["success_200"].sum() == phase_a["success"].sum() else False,
    }
    print("consistency:", consistency, flush=True)

    # SIM-TO-SIM ONLY: true gap of each policy's own chunk on held-out logged states
    labels = dict(np.load(root / "labels_val.npz"))
    ok = np.flatnonzero(labels["status"] == "ok")
    rng = np.random.default_rng(int(phase["true_gap"]["seed"]))
    chosen = np.sort(rng.choice(ok, size=int(phase["true_gap"]["samples"]), replace=False))
    states = load_script("13_audit_gap.py").read_states(root / "target_log_val.h5",
                                                         [(int(labels["episode_id"][i]), int(labels["t"][i])) for i in chosen])
    spec = chunk_spec(load_config(pilot["base_config"]), 0)
    gap_jobs, meta = [], []
    for s, m, path in policies:
        bundle = PolicyBundle.load(path)
        for i in chosen:
            actions = bundle.act(labels["obs"][i])[spec.delay : spec.delay + spec.evaluation_steps].astype(np.float32)
            key = (int(labels["episode_id"][i]), int(labels["t"][i]))
            gap_jobs.append({"job": len(gap_jobs), "state": states[key], "pending": labels["pending"][i], "actions": actions})
            meta.append({"seed": s, "method": m, "episode_id": key[0], "t": key[1], "perturbed": bool(labels["perturbed"][i])})
    results = run_gap_jobs(args.config, gap_jobs, args.workers)
    gaps = pd.DataFrame([m | {"true_gap_m2": results[n]["gap_m2"], "status": results[n]["status"]} for n, m in enumerate(meta)])
    gaps.to_parquet(out / "true_gap_sim2sim_only.parquet", index=False)

    summary = {
        "final_seeds": [seeds[0], seeds[-1]], "envs": final["envs"], "max_steps": max_steps,
        "episodes": int(len(frame)), "aborted_episodes": int((frame["status"] != "ok").sum()),
        "target_steps_final_eval": int(frame.loc[frame["env"] == "target", "steps"].sum()),
        "source_steps_final_eval": int(frame.loc[frame["env"] == "source", "steps"].sum()),
        "consistency_check": consistency,
        "true_gap": {"sim_to_sim_only": True, "samples": int(len(chosen)), "episodes": int(len(set(labels["episode_id"][chosen]))),
                     "aborted": int((gaps["status"] != "ok").sum()), "target_restores": len(gap_jobs)},
        "seconds": time.time() - started,
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    manifest_path = Path(pilot["output_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault(pilot.get("manifest_section_c", "phase_c"), {})["15_eval"] = summary | {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(frame.groupby(["env", "method"])[["success_200", "success_400"]].mean().unstack(0).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
