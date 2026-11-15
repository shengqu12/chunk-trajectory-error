"""Phase C, step 1: per-seed base policies, fine-tuned candidates and their selection.

``--stage train --seed S``
  * base policy: ``scripts/02_prepare_policy.py`` run unchanged with a per-seed copy of the base
    config (``experiment.seed = S``, ``experiment.output_dir`` = this seed's directory);
  * from it, with the 06 recipe (``06_train_policy.train_one``: same steps, learning rate, batch size
    and demonstration batch sequence):
      zero_shot   imitation only (lambda = 0)
      gap_demo    imitation + lambda * G on the same demonstration samples      (06 variant error_sqrt)
      gap_logged  imitation + lambda * G on logged target train states (policy chunk at the logged
                  observation, logged history / robot state); equal-size logged batches, own generator
      finetune    imitation on source demonstrations + the successful logged train episodes;
                  checkpoints every ``finetune_checkpoint_every`` steps are the candidates
      smooth      imitation + beta * action smoothness                            (06 variant smooth)
      imitation_selected (actuation, amendment 14) the zero-shot run with checkpoints every
                  ``imitation_checkpoint_every`` steps as candidates; the last one is zero-shot itself
      magnitude   (actuation, amendment 15) imitation + beta * mean squared translation command
  ``phase_c.methods`` lists the methods (default: the latency set above without the new two).
``--stage select``
  every candidate of every seed runs closed loop in the target from the 100 val-split seeds
  (200-step cap); per seed and method the candidate with the most successes is selected (ties ->
  smaller weight / earlier checkpoint).  These target steps are counted separately.

Outputs: ``runs/sim2sim_pilot/<shift>/phase_c/seed_<S>/`` (checkpoints, ``training.json``) and
``phase_c/selection.json``, ``phase_c/selection_episodes.parquet``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.objectives import set_trainable
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, build_policy_dataset, chunk_spec, hold_padded_chunk, load_demoset, seed_everything,
)
from chunk_trajectory_error.policies.mlp_bc import behavior_cloning_loss

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


def phase_dir(pilot: dict) -> Path:
    shift = load_script("10_shift_scan.py").shift_from_entry(pilot["phase_c"]["shift"]).name
    return Path(pilot["output_dir"]) / shift


def grad_norm(loss, parameters) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g**2).sum() for g in grads if g is not None)))


def train_gap_logged(config, spec, train, val, pi0_path, weight, gap, logged, logged_batch, logged_offset):
    """06's train_one with the auxiliary term moved to logged target states (amendment 9b)."""
    cfg = config["policy_finetune"]
    seed = config["experiment"]["seed"]
    seed_everything(seed)
    bundle = PolicyBundle.load(pi0_path)
    bundle.train()
    parameters = list(bundle.policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed + 6)  # identical demonstration batches to every other variant
    logged_generator = torch.Generator().manual_seed(seed + logged_offset)
    r, K = spec.delay, spec.evaluation_steps
    log = []
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=generator)
        chunk = bundle(train["obs"][batch])
        imitation = behavior_cloning_loss(chunk, train["chunk"][batch])
        lb = torch.randint(0, len(logged["obs"]), (logged_batch,), generator=logged_generator)
        tail = bundle(logged["obs"][lb])[:, r : r + K].clamp(-1.0, 1.0)  # executor clips commands to the action box
        auxiliary = gap.normalized_cost(logged["history"][lb], logged["robot"][lb], tail, r).mean()
        total = imitation + weight * auxiliary
        if step % 500 == 0 or step == 1:
            log.append({"step": step, "imitation": float(imitation.detach()), "auxiliary": float(auxiliary.detach()),
                        "grad_norm_imitation": grad_norm(imitation, parameters),
                        "grad_norm_weighted_auxiliary": grad_norm(weight * auxiliary, parameters)})
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    bundle.eval()
    with torch.no_grad():
        val_bc = float(behavior_cloning_loss(bundle(val["obs"]), val["chunk"]))
    return bundle, {"weight": weight, "val_bc_loss": val_bc, "log": log}


def train_aux(config, spec, train, val, pi0_path, weight, aux_fn, continuous):
    """06's train_one with a chunk-only auxiliary term ``aux_fn(chunk)`` (same seeds, batches, logging)."""
    cfg = config["policy_finetune"]
    seed_everything(config["experiment"]["seed"])
    bundle = PolicyBundle.load(pi0_path)
    bundle.train()
    parameters = list(bundle.policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 6)  # identical batches for all variants
    log = []
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=generator)
        chunk = bundle(train["obs"][batch])
        imitation = behavior_cloning_loss(chunk, train["chunk"][batch])
        auxiliary = aux_fn(chunk) if weight > 0 else chunk.sum() * 0.0
        total = imitation + weight * auxiliary
        if step % 500 == 0 or step == 1:
            entry = {"step": step, "imitation": float(imitation.detach()), "auxiliary": float(auxiliary.detach()),
                     "grad_norm_imitation": grad_norm(imitation, parameters)}
            entry["grad_norm_weighted_auxiliary"] = grad_norm(weight * auxiliary, parameters) if weight > 0 and auxiliary.requires_grad else 0.0
            log.append(entry)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    bundle.eval()
    with torch.no_grad():
        val_chunk = bundle(val["obs"])
        val_bc = float(behavior_cloning_loss(val_chunk, val["chunk"]))
        val_smooth = float(((val_chunk[:, 1:, continuous] - val_chunk[:, :-1, continuous]) ** 2).mean())
        val_magnitude = float((val_chunk[:, :, continuous] ** 2).mean())
    return bundle, {"weight": weight, "val_bc_loss": val_bc, "val_chunk_roughness": val_smooth,
                    "val_chunk_magnitude": val_magnitude, "log": log}


def magnitude_penalty(continuous):
    """Amendment 15: mean squared translation command of the chunk (all H steps, unclipped; same scope as smoothness)."""
    return lambda chunk: (chunk[:, :, continuous] ** 2).mean()


def smoothness_penalty(continuous):
    """The 06 smoothness term, used only to check that ``train_aux`` reproduces ``train_one`` bitwise."""
    return lambda chunk: ((chunk[:, 1:, continuous] - chunk[:, :-1, continuous]) ** 2).mean()


def train_imitation_checkpoints(config, spec, train, val, pi0_path, every, save):
    """06's train_one for variant bc (lambda = 0), saving a checkpoint every ``every`` steps (amendment 14)."""
    cfg = config["policy_finetune"]
    seed_everything(config["experiment"]["seed"])
    bundle = PolicyBundle.load(pi0_path)
    bundle.train()
    optimizer = torch.optim.Adam(list(bundle.policy.parameters()), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 6)
    infos = []
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=generator)
        chunk = bundle(train["obs"][batch])
        imitation = behavior_cloning_loss(chunk, train["chunk"][batch])
        total = imitation + 0.0 * (chunk.sum() * 0.0)  # exactly train_one's bc objective
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        if step % every == 0:
            bundle.eval()
            with torch.no_grad():
                val_bc = float(behavior_cloning_loss(bundle(val["obs"]), val["chunk"]))
            infos.append({"step": step, "train_loss": float(imitation.detach()), "val_bc_loss": val_bc})
            save(step, bundle, infos[-1])
            bundle.train()
    return infos


def same_weights(a: PolicyBundle, b: PolicyBundle) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def train_finetune(config, spec, train, val, pi0_path, extra, every, save):
    """06 recipe, imitation only, on demonstrations + logged successful target episodes (uniform over the union)."""
    cfg = config["policy_finetune"]
    seed = config["experiment"]["seed"]
    seed_everything(seed)
    obs = torch.cat([train["obs"], extra["obs"]])
    chunks = torch.cat([train["chunk"], extra["chunk"]])
    bundle = PolicyBundle.load(pi0_path)
    bundle.train()
    optimizer = torch.optim.Adam(list(bundle.policy.parameters()), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed + 6)
    infos = []
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(obs), (cfg["batch_size"],), generator=generator)
        loss = behavior_cloning_loss(bundle(obs[batch]), chunks[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % every == 0:
            bundle.eval()
            with torch.no_grad():
                val_bc = float(behavior_cloning_loss(bundle(val["obs"]), val["chunk"]))
            path = save(step, bundle)
            infos.append({"step": step, "train_loss": float(loss.detach()), "val_bc_loss": val_bc, "checkpoint": str(path)})
            bundle.train()
    return infos


def logged_success_episodes(path: Path, horizon: int, gripper_dims: list[int]) -> tuple[dict, dict]:
    import h5py

    obs, chunks, info = [], [], []
    with h5py.File(path, "r") as store:
        for key in store:
            group = store[key]
            if not bool(group.attrs["success"]):
                continue
            commanded = group["commanded"][()]
            for t in range(len(commanded)):
                obs.append(group["obs"][t])
                chunks.append(hold_padded_chunk(commanded, t, horizon, gripper_dims))
            info.append({"episode_id": int(group.attrs["episode_id"]), "steps": int(group.attrs["steps"]), "perturbed": bool(group.attrs["perturbed"])})
    return ({"obs": torch.tensor(np.stack(obs), dtype=torch.float32), "chunk": torch.tensor(np.stack(chunks))},
            {"episodes": len(info), "samples": len(obs), "perturbed_episodes": sum(i["perturbed"] for i in info), "by_episode": info})


def stage_train(pilot_path: str, pilot: dict, seed: int) -> None:
    phase = pilot["phase_c"]
    root = phase_dir(pilot)
    seed_dir = root / "phase_c" / f"seed_{seed}"
    (seed_dir / "logs").mkdir(parents=True, exist_ok=True)
    base_config = load_config(pilot["base_config"])
    base_config["experiment"]["seed"] = int(seed)
    base_config["experiment"]["output_dir"] = str(seed_dir)
    config_path = seed_dir / "base_config.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    started = time.time()
    pi0_path = seed_dir / "checkpoints" / "pi0.pt"
    with (seed_dir / "logs" / "02_prepare_policy.log").open("w") as log_file:
        subprocess.run([sys.executable, str(PIPELINE_SCRIPTS / "02_prepare_policy.py"), "--config", str(config_path)],
                       check=True, stdout=log_file, stderr=subprocess.STDOUT)

    f06 = load_script("06_train_policy.py")
    spec = chunk_spec(base_config, 0)
    demos = load_demoset(base_config)
    train, val = build_policy_dataset(demos, "train", spec), build_policy_dataset(demos, "val", spec)
    continuous = [d for d in range(train["chunk"].shape[2]) if d not in demos.gripper_dims]
    gap = ErrorBundle.load(phase["gap_checkpoint"])
    set_trainable(gap, False)
    labels = dict(np.load(root / "labels_train.npz"))
    ok = labels["status"] == "ok"
    logged = {key: torch.tensor(labels[key][ok]) for key in ("obs", "history", "robot")}
    extra, extra_info = logged_success_episodes(root / "target_log_train.h5", spec.horizon, demos.gripper_dims)

    candidates = []

    def save(method, tag, bundle, info):
        path = seed_dir / "checkpoints" / method / f"{tag}.pt"
        bundle.save(path, {"method": method, "tag": tag, "seed": seed, "start": str(pi0_path)})
        entry = {"method": method, "tag": tag, "checkpoint": str(path)} | {k: v for k, v in info.items() if k != "log"}
        entry["last_log"] = info.get("log", [{}])[-1] if info.get("log") else {}
        candidates.append(entry)
        print(f"seed {seed} {method} {tag}: val BC {info.get('val_bc_loss', float('nan')):.5f}", flush=True)

    methods = list(phase.get("methods", METHODS))
    checks = {}
    zero_shot, info = f06.train_one(base_config, spec, train, val, pi0_path, "bc", 0.0, None, continuous)
    save("zero_shot", "w0", zero_shot, info)
    if "imitation_selected" in methods:
        every = int(phase["imitation_checkpoint_every"])

        def save_im(step, bundle, info):
            if step == int(base_config["policy_finetune"]["steps"]) and same_weights(bundle, zero_shot):
                checks["imitation_last_checkpoint_equals_zero_shot"] = True
                path = Path(candidates[0]["checkpoint"])  # the zero-shot file itself; shares its selection rollouts
            else:
                if step == int(base_config["policy_finetune"]["steps"]):
                    checks["imitation_last_checkpoint_equals_zero_shot"] = False
                path = seed_dir / "checkpoints" / "imitation_selected" / f"step{step}.pt"
                bundle.save(path, {"method": "imitation_selected", "tag": f"step{step}", "seed": seed, "start": str(pi0_path)})
            candidates.append({"method": "imitation_selected", "tag": f"step{step}", "weight": step, "step": step,
                               "train_loss": info["train_loss"], "val_bc_loss": info["val_bc_loss"], "checkpoint": str(path)})
            print(f"seed {seed} imitation_selected step{step}: val BC {info['val_bc_loss']:.5f}", flush=True)

        train_imitation_checkpoints(base_config, spec, train, val, pi0_path, every, save_im)
    if "gap_demo" in methods:
        for weight in phase["lambdas"]:
            bundle, info = f06.train_one(base_config, spec, train, val, pi0_path, "error_sqrt", float(weight), gap, continuous)
            save("gap_demo", f"w{weight:g}", bundle, info)
    if "gap_logged" in methods:
        for weight in phase["lambdas"]:
            bundle, info = train_gap_logged(base_config, spec, train, val, pi0_path, float(weight), gap, logged,
                                            int(phase["logged_batch_size"]), int(phase["logged_batch_seed_offset"]))
            save("gap_logged", f"w{weight:g}", bundle, info)
    if "smooth" in methods:
        for weight in phase["smooth_betas"]:
            bundle, info = f06.train_one(base_config, spec, train, val, pi0_path, "smooth", float(weight), None, continuous)
            save("smooth", f"w{weight:g}", bundle, info)
            if "magnitude" in methods and "train_aux_reproduces_train_one_smooth" not in checks:
                mirror, _ = train_aux(base_config, spec, train, val, pi0_path, float(weight), smoothness_penalty(continuous), continuous)
                checks["train_aux_reproduces_train_one_smooth"] = {"beta": float(weight), "bitwise_equal": same_weights(mirror, bundle)}
                print(f"seed {seed} train_aux check: {checks['train_aux_reproduces_train_one_smooth']}", flush=True)
    if "magnitude" in methods:
        for weight in phase["magnitude_betas"]:
            bundle, info = train_aux(base_config, spec, train, val, pi0_path, float(weight), magnitude_penalty(continuous), continuous)
            save("magnitude", f"w{weight:g}", bundle, info)

    def save_ft(step, bundle):
        path = seed_dir / "checkpoints" / "finetune" / f"step{step}.pt"
        bundle.save(path, {"method": "finetune", "tag": f"step{step}", "seed": seed, "start": str(pi0_path)})
        return path

    for info in (train_finetune(base_config, spec, train, val, pi0_path, extra, int(phase["finetune_checkpoint_every"]), save_ft)
                 if "finetune" in methods else []):
        candidates.append({"method": "finetune", "tag": f"step{info['step']}", "weight": info["step"]} | info)
        print(f"seed {seed} finetune step{info['step']}: val BC {info['val_bc_loss']:.5f}", flush=True)

    record = {"seed": seed, "config": str(config_path), "pi0": str(pi0_path),
              "pi0_baseline": json.loads((seed_dir / "pi0_baseline.json").read_text()) | {"curve": None},
              "finetune_data": {k: v for k, v in extra_info.items()}, "logged_states_for_gap_logged": int(ok.sum()),
              "candidates": candidates, "checks": checks, "seconds": time.time() - started}
    (seed_dir / "training.json").write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")


def stage_select(pilot_path: str, pilot: dict, workers: int) -> None:
    import pandas as pd

    from chunk_trajectory_error.sim2sim_jobs import closed_loop_worker, run_pool

    phase = pilot["phase_c"]
    root = phase_dir(pilot) / "phase_c"
    base_config = load_config(pilot["base_config"])
    demos = load_demoset(base_config)
    val_seeds = [int(demos.episodes[e].seed) for e in demos.ids("val")]
    final = range(phase["final_eval"]["seeds"]["start"], phase["final_eval"]["seeds"]["start"] + phase["final_eval"]["seeds"]["count"])
    if set(val_seeds) & set(final):
        raise RuntimeError("selection seeds overlap the final evaluation seeds")
    methods = list(phase.get("methods", METHODS))
    jobs, table_rows = [], []
    for seed in phase["seeds"]:
        training = json.loads((root / f"seed_{seed}" / "training.json").read_text())
        paths = list(dict.fromkeys(c["checkpoint"] for c in training["candidates"] if c["method"] in methods))
        for c in training["candidates"]:
            if c["method"] in methods:
                table_rows.append({"seed": seed, "checkpoint": c["checkpoint"], "method": c["method"], "tag": c["tag"], "weight": float(c["weight"])})
        for path in paths:
            jobs.append((pilot_path, "target", int(phase["selection"]["max_steps"]),
                         [{"policy_path": path, "seeds": val_seeds, "tags": {"seed": seed, "checkpoint": path}}]))
    started = time.time()
    runs = pd.DataFrame([row for result in run_pool(closed_loop_worker, jobs, workers) for row in result])
    candidates = pd.DataFrame(table_rows)
    frame = runs.merge(candidates, on=["seed", "checkpoint"], how="inner", validate="many_to_many")
    shared = candidates.groupby(["seed", "checkpoint"]).size()
    frame["shared_rollout"] = [shared[(s, c)] > 1 for s, c in zip(frame["seed"], frame["checkpoint"])]
    if len(frame) != len(candidates) * len(val_seeds):
        raise RuntimeError("selection rollouts do not cover every candidate")
    frame.to_parquet(root / "selection_episodes.parquet", index=False)
    selection = {"rule": phase["selection"]["rule"], "initial_states": "100 val-split seeds (reset by seed)", "env": "target",
                 "max_steps": phase["selection"]["max_steps"], "target_steps_used": int(runs["steps"].sum()),
                 "episodes": int(len(runs)), "distinct_checkpoints": int(len(jobs)), "candidates": int(len(candidates)),
                 "aborted": int((runs["status"] != "ok").sum()), "seeds": {}, "seconds": time.time() - started}
    for seed in phase["seeds"]:
        selection["seeds"][str(seed)] = {}
        for method in methods:
            sub = frame[(frame["seed"] == seed) & (frame["method"] == method)]
            table = sub.groupby(["tag", "weight", "checkpoint"])["success"].sum().reset_index().sort_values("weight")
            best = table.sort_values(["success", "weight"], ascending=[False, True]).iloc[0]
            selection["seeds"][str(seed)][method] = {
                "selected": best["tag"], "selected_weight": float(best["weight"]), "val_successes": int(best["success"]),
                "candidates": {row.tag: int(row.success) for row in table.itertuples()},
                "checkpoint": str(best["checkpoint"]),
            }
    (root / "selection.json").write_text(json.dumps(selection, indent=2, default=float), encoding="utf-8")
    manifest_path = Path(pilot["output_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault(pilot.get("manifest_section_c", "phase_c"), {})["14_selection"] = {k: v for k, v in selection.items() if k != "seeds"} | {
        "selected": {s: {m: v["selected"] for m, v in d.items()} for s, d in selection["seeds"].items()}}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(json.dumps({s: {m: (v["selected"], v["val_successes"], v["candidates"]) for m, v in d.items()} for s, d in selection["seeds"].items()}, indent=1))
    print("selection target steps:", selection["target_steps_used"], "aborted:", selection["aborted"])


def main() -> int:
    parser = ArgumentParser(description="Phase C: base policies, fine-tuned candidates, selection on val seeds in the target")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--stage", choices=("train", "select"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=22)
    args = parser.parse_args()
    pilot = load_config(args.config)
    if args.stage == "train":
        if args.seed not in pilot["phase_c"]["seeds"]:
            raise ValueError("--seed must be one of phase_c.seeds")
        stage_train(args.config, pilot, args.seed)
    else:
        stage_select(args.config, pilot, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
