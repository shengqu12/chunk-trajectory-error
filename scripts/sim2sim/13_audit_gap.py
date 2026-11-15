"""Phase B, step 3 -- SIM-TO-SIM ONLY DIAGNOSTICS (these restore the target; never usable on hardware).

On 300 validation samples of the policy's own unperturbed chunks (pi_bc at the logged target
observation, first K actions, r = 0), a step along -dG/dA (gripper masked, L2 norm over the K x 3
continuous entries) is compared with an equal-norm random step.  Base, gradient and random chunks
are executed from the logged state in both the source (no delay) and the target (latency wrapper,
pending-command buffer restored from the log) and the true gap is recomputed.  The step size is
chosen on half of the samples (split by episode; rule of 05: lowest mean change of the true gap
under the gradient step) and tested once on the other half.

Gate 3: on the test half, the 95% episode-clustered bootstrap interval of dJ_grad is below 0 and
that of (dJ_grad - dJ_rand) is below 0.

Outputs (``runs/sim2sim_pilot/<shift>/``): ``gap_audit_sim2sim_only.csv``, ``gap_audit_sim2sim_only.json``.
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
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle, bootstrap_ci, chunk_spec

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


BANNER = "SIM-TO-SIM ONLY: target restores are used below; nothing here is a hardware-faithful measurement"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_states(path: Path, wanted: list[tuple[int, int]]) -> dict:
    import h5py

    def take(node, t):
        if isinstance(node, h5py.Dataset):
            return np.array(node[t])
        return {key: take(node[key], t) for key in node}

    with h5py.File(path, "r") as store:
        return {(e, t): take(store[f"ep_{e:04d}"]["env_states"], t) for e, t in wanted}


def rollout_worker(pilot_path: str, jobs: list[dict]) -> list[dict]:
    """Execute action sequences from logged states in the source and in the restored target."""
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend
    from chunk_trajectory_error.shifts import make_target_backend
    from chunk_trajectory_error.sim2sim import gap_cost, replay_source, target_restore

    pilot = load_config(pilot_path)
    base_config = load_config(pilot["base_config"])
    shift = load_script("10_shift_scan.py").shift_from_entry(pilot["phase_b"]["shift"])
    source = ManiSkillBackend(base_config)
    source.reset(seed=int(pilot["phase_b"]["first_reset_seed"]))
    target, record = make_target_backend(base_config, shift, list(base_config["action"]["gripper_dims"]),
                                    first_seed=int(pilot["phase_b"]["first_reset_seed"]))
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
        except Exception as error:
            t_status = f"aborted:{type(error).__name__}"
        ok = s_status == "ok" and t_status == "ok"
        results.append({"job": job["job"], "gap_m2": gap_cost(p_target, p_source) if ok else np.nan, "status": "ok" if ok else f"{s_status}|{t_status}"})
    from chunk_trajectory_error.shifts import verify_shift

    verify_shift(target, record)  # the shift survived every restore
    source.close()
    target.close()
    return results


def run_jobs(pilot_path: str, jobs: list[dict], workers: int) -> dict[int, dict]:
    chunks = [jobs[i::workers] for i in range(workers)]
    out = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for future in [pool.submit(rollout_worker, pilot_path, c) for c in chunks if c]:
            for row in future.result():
                out[row["job"]] = row
    return out


def summarise(frame: pd.DataFrame) -> dict:
    d_grad = (frame["J_grad_m2"] - frame["J0_m2"]).to_numpy()
    d_rand = (frame["J_rand_m2"] - frame["J0_m2"]).to_numpy()
    groups = frame["episode_id"].to_numpy()
    mg, lg, hg = bootstrap_ci(d_grad, groups)
    mr, lr, hr = bootstrap_ci(d_rand, groups)
    mp_, lp, hp = bootstrap_ci(d_grad - d_rand, groups)
    return {
        "samples": int(len(frame)), "episodes": int(frame["episode_id"].nunique()), "mean_J0_m2": float(frame["J0_m2"].mean()),
        "delta_J_grad_m2": {"mean": mg, "ci95": [lg, hg]}, "delta_J_rand_m2": {"mean": mr, "ci95": [lr, hr]},
        "paired_grad_minus_rand_m2": {"mean": mp_, "ci95": [lp, hp]},
        "relative_change_grad": float(d_grad.sum() / frame["J0_m2"].sum()), "relative_change_rand": float(d_rand.sum() / frame["J0_m2"].sum()),
        "fraction_grad_lowers_J": float((d_grad < 0).mean()), "fraction_grad_beats_rand": float((d_grad < d_rand).mean()),
        "mean_effective_norm_grad": float(frame["effective_norm_grad"].mean()), "mean_effective_norm_rand": float(frame["effective_norm_rand"].mean()),
    }


def main() -> int:
    parser = ArgumentParser(description="SIM-TO-SIM ONLY: gradient-direction audit of the gap network (gate 3)")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    print(BANNER, flush=True)
    pilot = load_config(args.config)
    phase, audit_cfg = pilot["phase_b"], pilot["phase_b"]["audit"]
    base_config = load_config(pilot["base_config"])
    spec = chunk_spec(base_config, 0)
    r, K = spec.delay, spec.evaluation_steps
    shift_name = load_script("10_shift_scan.py").shift_from_entry(phase["shift"]).name
    run_root = Path(pilot["output_dir"])
    out = run_root / shift_name
    continuous = [d for d in range(4) if d not in base_config["action"]["gripper_dims"]]
    started = time.time()

    labels = dict(np.load(out / "labels_val.npz"))
    eligible = np.flatnonzero((~labels["perturbed"]) & (labels["status"] == "ok"))
    rng = np.random.default_rng(int(audit_cfg["seed"]))
    chosen = np.sort(rng.choice(eligible, size=int(audit_cfg["samples"]), replace=False))
    episodes = sorted(set(labels["episode_id"][chosen].tolist()))
    selection_episodes = set(episodes[0::2])
    states = read_states(out / "target_log_val.h5", [(int(labels["episode_id"][i]), int(labels["t"][i])) for i in chosen])

    policy = PolicyBundle.load(phase["policy_checkpoint"])
    networks = {target: ErrorBundle.load(out / "checkpoints" / f"gap_{target}.pt") for target in ("J", "sqrtJ")}
    for network in networks.values():
        for parameter in network.parameters():
            parameter.requires_grad_(False)

    samples = []
    for i in chosen:
        base = policy.act(labels["obs"][i])[r : r + K].astype(np.float32)  # the policy's own chunk, unperturbed
        noise = np.zeros_like(base)
        noise[:, continuous] = rng.standard_normal((K, len(continuous)))
        noise /= np.linalg.norm(noise)
        entry = {"index": int(i), "episode_id": int(labels["episode_id"][i]), "t": int(labels["t"][i]),
                 "half": "selection" if int(labels["episode_id"][i]) in selection_episodes else "test",
                 "base": base, "noise": noise, "label_J_m2": float(labels["J_gap_m2"][i]), "directions": {}, "pred": {}}
        for target, network in networks.items():
            action = torch.tensor(base)[None].requires_grad_(True)
            predicted = network.cost_m2(torch.tensor(labels["history"][i])[None], torch.tensor(labels["robot"][i])[None], action, r)
            (gradient,) = torch.autograd.grad(predicted.sum(), action)
            direction = np.zeros_like(base)
            direction[:, continuous] = gradient[0].numpy()[:, continuous]
            norm = float(np.linalg.norm(direction))
            entry["directions"][target] = direction / norm if norm > 0 else None
            entry["pred"][target] = float(predicted.detach())
        samples.append(entry)
    skipped_zero_gradient = {t: sum(s["directions"][t] is None for s in samples) for t in networks}

    def make_jobs(etas: dict[str, list[float]], half: str) -> tuple[list[dict], list[dict]]:
        jobs, meta = [], []
        for s in samples:
            if s["half"] != half:
                continue
            key = (s["episode_id"], s["t"])
            base_job = len(jobs)
            jobs.append({"job": base_job, "state": states[key], "pending": labels["pending"][s["index"]], "actions": s["base"]})
            for target, eta_list in etas.items():
                if s["directions"][target] is None:
                    continue
                for eta in eta_list:
                    a_grad = np.clip(s["base"] - eta * s["directions"][target], -1.0, 1.0).astype(np.float32)
                    a_rand = np.clip(s["base"] + eta * s["noise"], -1.0, 1.0).astype(np.float32)
                    g, q = len(jobs), len(jobs) + 1
                    jobs.append({"job": g, "state": states[key], "pending": labels["pending"][s["index"]], "actions": a_grad})
                    jobs.append({"job": q, "state": states[key], "pending": labels["pending"][s["index"]], "actions": a_rand})
                    meta.append({"sample": s, "network": target, "eta": eta, "jobs": (base_job, g, q),
                                 "effective_norm_grad": float(np.linalg.norm(a_grad - s["base"])),
                                 "effective_norm_rand": float(np.linalg.norm(a_rand - s["base"]))})
        return jobs, meta

    def to_frame(meta, results, half) -> tuple[pd.DataFrame, int]:
        rows, aborted = [], 0
        for m in meta:
            b, g, q = (results[j] for j in m["jobs"])
            if any(x["status"] != "ok" for x in (b, g, q)):
                aborted += 1
                continue
            s = m["sample"]
            rows.append({"half": half, "network": m["network"], "eta": m["eta"], "episode_id": s["episode_id"], "t": s["t"],
                         "label_J_m2": s["label_J_m2"], "G_pred_m2": s["pred"][m["network"]], "J0_m2": b["gap_m2"],
                         "J_grad_m2": g["gap_m2"], "J_rand_m2": q["gap_m2"],
                         "effective_norm_grad": m["effective_norm_grad"], "effective_norm_rand": m["effective_norm_rand"]})
        return pd.DataFrame(rows), aborted

    step_sizes = [float(x) for x in audit_cfg["step_sizes"]]
    jobs, meta = make_jobs({t: step_sizes for t in networks}, "selection")
    selection, aborted_sel = to_frame(meta, run_jobs(args.config, jobs, args.workers), "selection")
    target_restores = len(jobs)
    selection_summary, chosen_eta = {}, {}
    for target in networks:
        sub = selection[selection["network"] == target]
        selection_summary[target] = {str(eta): summarise(sub[sub["eta"] == eta]) for eta in step_sizes}
        chosen_eta[target] = min(step_sizes, key=lambda eta: selection_summary[target][str(eta)]["delta_J_grad_m2"]["mean"])
    print("step size chosen on the selection half:", chosen_eta, flush=True)
    jobs, meta = make_jobs({t: [chosen_eta[t]] for t in networks}, "test")
    test, aborted_test = to_frame(meta, run_jobs(args.config, jobs, args.workers), "test")
    target_restores += len(jobs)

    # restore-fidelity check: the logged commanded actions replayed in the restored target and in the source
    # should reproduce the forward-run label up to the restore noise
    fidelity_jobs = [{"job": n, "state": states[(s["episode_id"], s["t"])], "pending": labels["pending"][s["index"]],
                      "actions": labels["actions"][s["index"]]} for n, s in enumerate(samples)]
    fidelity_results = run_jobs(args.config, fidelity_jobs, args.workers)
    target_restores += len(fidelity_jobs)
    restored = np.array([fidelity_results[n]["gap_m2"] for n in range(len(samples))])
    forward = np.array([s["label_J_m2"] for s in samples])

    frame = pd.concat([selection, test])
    frame.to_csv(out / "gap_audit_sim2sim_only.csv", index=False)
    test_summary, gates = {}, {}
    for target in networks:
        row = summarise(test[test["network"] == target])
        row["step_size"] = chosen_eta[target]
        test_summary[target] = row
        gates[target] = bool(row["delta_J_grad_m2"]["ci95"][1] < 0 and row["paired_grad_minus_rand_m2"]["ci95"][1] < 0)
    base_rows = frame.drop_duplicates(["episode_id", "t"])
    summary = {
        "sim_to_sim_only": True, "banner": BANNER,
        "samples": int(len(chosen)), "sample_source": "val split, unperturbed episodes, pi_bc chunk at the logged target observation",
        "episodes": len(episodes), "selection_episodes": sorted(selection_episodes),
        "samples_by_half": {"selection": int(sum(s["half"] == "selection" for s in samples)), "test": int(sum(s["half"] == "test" for s in samples))},
        "step_sizes": step_sizes, "step_size_rule": "lowest mean dJ_grad on the selection half (as 05)", "chosen_step_size": chosen_eta,
        "selection": selection_summary, "test": test_summary,
        "gate3": {"rule": "on the test half at the chosen step: episode-clustered bootstrap 95% CI of dJ_grad below 0 AND of (dJ_grad - dJ_rand) below 0",
                  "by_network": gates},
        "skipped": {"zero_gradient": skipped_zero_gradient, "aborted_comparisons_selection": aborted_sel, "aborted_comparisons_test": aborted_test},
        "target_restores": int(target_restores),
        "restore_fidelity_logged_actions": {
            "note": "same logged commanded actions: target restored from the log (incl. pending buffer) vs the forward-run target label",
            "samples": int(len(samples)), "aborted": int(sum(fidelity_results[n]["status"] != "ok" for n in range(len(samples)))),
            "median_abs_diff_m2": float(np.nanmedian(np.abs(restored - forward))),
            "p90_abs_diff_m2": float(np.nanpercentile(np.abs(restored - forward), 90)),
            "median_forward_label_m2": float(np.median(forward)), "median_restored_m2": float(np.nanmedian(restored)),
            "spearman_restored_vs_forward": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(restored, forward, nan_policy="omit").statistic),
        },
        "seconds": time.time() - started,
    }
    (out / "gap_audit_sim2sim_only.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("phase_b", {})[f"13_{shift_name}_sim2sim_only"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sim_to_sim_only": True, "target_restores": int(target_restores),
        "gate3": summary["gate3"], "chosen_step_size": chosen_eta, "skipped": summary["skipped"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("chosen_step_size", "test", "gate3", "skipped", "target_restores", "restore_fidelity_logged_actions")}, indent=2, default=float))
    for target, ok in gates.items():
        print(f"GATE 3 ({target}):", "PASS" if ok else "FAIL")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
