"""Phase C pre-training diagnostic -- SIM-TO-SIM ONLY, NON-GATING (target restores are used).

Is G's gradient usable on source demonstration states (where gap-demo applies it)?  On 300 samples
from held-out train demonstrations (train-split episodes whose seeds were not used for the Phase B
target logs), with pi_bc's chunk at the demonstration observation (first K actions, r = 0):
  * true gap: source and target restored to the demonstration state, target pending commands = the
    previous d demonstration actions (latency d; no pending buffer for other shifts); both execute the chunk; J = mean_j ||p_target - p_source||^2;
  * G's prediction on the demonstration history / robot state / chunk;
  * gradient-vs-random step test at every step size in ``step_sizes`` (no selection, all reported).

Outputs (``runs/sim2sim_pilot/<shift>/``): ``gap_demo_diagnostic_sim2sim_only.json`` / ``.csv``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle, build_policy_dataset, chunk_spec, load_demoset

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


BANNER = "SIM-TO-SIM ONLY, NON-GATING: target restores are used below"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = ArgumentParser(description=BANNER)
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    print(BANNER, flush=True)
    audit13 = load_script("13_audit_gap.py")
    pilot = load_config(args.config)
    diag = pilot["phase_c"]["demo_diagnostic"]
    base_config = load_config(pilot["base_config"])
    spec = chunk_spec(base_config, 0)
    r, K = spec.delay, spec.evaluation_steps
    shift_name = load_script("10_shift_scan.py").shift_from_entry(pilot["phase_b"]["shift"]).name
    run_root = Path(pilot["output_dir"])
    out = run_root / shift_name
    continuous = [d for d in range(4) if d not in base_config["action"]["gripper_dims"]]
    started = time.time()

    demos = load_demoset(base_config)
    logged = set(json.loads((out / "collection_train.json").read_text())["episode_ids"])
    held_out = [e for e in demos.ids("train") if e not in logged]
    data = build_policy_dataset(demos, "train", spec)
    index = data["index"].numpy()
    pool = np.flatnonzero(data["error_valid"].numpy() & np.isin(index[:, 0], held_out))
    rng = np.random.default_rng(int(diag["seed"]))
    chosen = np.sort(rng.choice(pool, size=int(diag["samples"]), replace=False))

    policy = PolicyBundle.load(pilot["phase_b"]["policy_checkpoint"])
    gap = ErrorBundle.load(pilot["phase_c"]["gap_checkpoint"])
    for parameter in gap.parameters():
        parameter.requires_grad_(False)
    step_sizes = [float(x) for x in diag["step_sizes"]]
    delay = int(pilot["phase_c"]["shift"].get("d", 0)) if pilot["phase_c"]["shift"]["family"] == "latency" else 0

    samples, jobs = [], []
    for i in chosen:
        episode_id, t = int(index[i, 0]), int(index[i, 1])
        episode = demos.episodes[episode_id]
        base = policy.act(data["obs"][i].numpy())[r : r + K].astype(np.float32)
        action = torch.tensor(base)[None].requires_grad_(True)
        predicted = gap.cost_m2(data["history"][i][None], data["robot"][i][None], action, r)
        (gradient,) = torch.autograd.grad(predicted.sum(), action)
        direction = np.zeros_like(base)
        direction[:, continuous] = gradient[0].numpy()[:, continuous]
        norm = float(np.linalg.norm(direction))
        noise = np.zeros_like(base)
        noise[:, continuous] = rng.standard_normal((K, len(continuous)))
        noise /= np.linalg.norm(noise)
        state = episode.state_at(t + r)
        # latency d: the previous d demonstration actions are in flight; other shifts have no pending buffer
        pending = (episode.actions[t - delay : t] if delay > 0 else episode.actions[:0]).astype(np.float32)
        entry = {"episode_id": episode_id, "t": t, "G_pred_m2": float(predicted.detach()), "grad_norm": norm, "jobs": {}}
        entry["jobs"]["base"] = len(jobs)
        jobs.append({"job": len(jobs), "state": state, "pending": pending, "actions": base})
        if norm > 0:
            for eta in step_sizes:
                for kind, actions in (("grad", base - eta * direction / norm), ("rand", base + eta * noise)):
                    actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
                    entry["jobs"][(kind, eta)] = len(jobs)
                    entry[f"norm_{kind}_{eta}"] = float(np.linalg.norm(actions - base))
                    jobs.append({"job": len(jobs), "state": state, "pending": pending, "actions": actions})
        samples.append(entry)
    from chunk_trajectory_error.sim2sim_jobs import run_gap_jobs

    results = run_gap_jobs(args.config, jobs, args.workers)

    rows, aborted, zero_gradient = [], 0, sum(s["grad_norm"] == 0 for s in samples)
    for s in samples:
        b = results[s["jobs"]["base"]]
        if b["status"] != "ok":
            aborted += 1
            continue
        for eta in step_sizes:
            if ("grad", eta) not in s["jobs"]:
                continue
            g, q = results[s["jobs"][("grad", eta)]], results[s["jobs"][("rand", eta)]]
            if g["status"] != "ok" or q["status"] != "ok":
                aborted += 1
                continue
            rows.append({"episode_id": s["episode_id"], "t": s["t"], "eta": eta, "G_pred_m2": s["G_pred_m2"], "J0_m2": b["gap_m2"],
                         "J_grad_m2": g["gap_m2"], "J_rand_m2": q["gap_m2"],
                         "effective_norm_grad": s[f"norm_grad_{eta}"], "effective_norm_rand": s[f"norm_rand_{eta}"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "gap_demo_diagnostic_sim2sim_only.csv", index=False)
    base_rows = frame.drop_duplicates(["episode_id", "t"])
    true, pred = base_rows["J0_m2"].to_numpy(), base_rows["G_pred_m2"].to_numpy()
    labels_val = dict(np.load(out / "labels_val.npz"))["J_gap_m2"]
    fit_mean = float(gap.meta["j_ref_m2"])

    def dist(values):
        values = np.asarray(values, np.float64)
        return {"p10": float(np.percentile(values, 10)), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "mean": float(values.mean())}

    summary = {
        "sim_to_sim_only": True, "gating": False, "banner": BANNER,
        "samples": int(len(base_rows)), "episodes": int(base_rows["episode_id"].nunique()),
        "held_out_train_episodes": len(held_out), "policy": pilot["phase_b"]["policy_checkpoint"], "gap_checkpoint": pilot["phase_c"]["gap_checkpoint"],
        "true_gap_demo_states_m2": dist(true), "G_pred_demo_states_m2": dist(pred), "labels_logged_val_m2": dist(labels_val),
        "prediction_vs_truth": {
            "spearman": float(spearmanr(pred, true).statistic), "pearson": float(pearsonr(pred, true).statistic),
            "MAE_m2": float(np.abs(pred - true).mean()), "constant_fit_mean_MAE_m2": float(np.abs(fit_mean - true).mean()),
            "constant_demo_mean_MAE_m2": float(np.abs(true.mean() - true).mean()),
            "mean_pred_over_mean_true": float(pred.mean() / true.mean()),
            "median_pred_over_true": float(np.median(pred / np.maximum(true, 1e-15))),
        },
        "step_test": {str(eta): audit13.summarise(frame[frame["eta"] == eta]) for eta in step_sizes},
        "skipped": {"zero_gradient": int(zero_gradient), "aborted_comparisons": int(aborted)},
        "target_restores": len(jobs), "seconds": time.time() - started,
    }
    (out / "gap_demo_diagnostic_sim2sim_only.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault(pilot.get("manifest_section_c", "phase_c"), {})["13_gap_demo_diagnostic_sim2sim_only"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gating": False, "target_restores": len(jobs),
        "samples": summary["samples"], "skipped": summary["skipped"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("banner",)}, indent=2, default=float))
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
