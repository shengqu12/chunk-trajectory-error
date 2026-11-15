"""Phase B, step 2: train the gap network G (ErrorBundle + the 04 training loop) on the gap labels.

Two targets are trained: ``J`` (regression on J_gap) and ``sqrtJ`` (regression on sqrt(J_gap)).
Inputs: relative EE history over the previous L logged steps, robot state (``robot_state_slice``),
the K commanded actions, delay r = 0.  Normalisation and label scale come from the fitting episodes.

The 04 loop selects its best checkpoint on validation; here every ``checkpoint_holdout_every``-th
*train* episode is held out for that purpose instead, so the validation episodes used for gate 2
are never seen during training or checkpoint selection.

Reports on validation (gate 2: MAE below the constant-mean predictor and global Spearman > 0.5),
the roughness and action-magnitude comparisons (magnitude added for the actuation run) (rank correlation of roughness with the true gap and with G, and a
one-feature linear predictor fit on train), and the off-distribution check (G on source
demonstration states with demonstration actions, next to the label distribution).

Outputs (``runs/sim2sim_pilot/<shift>/``): ``checkpoints/gap_<target>.pt``, ``gap_report.json``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.error_net import error_network_loss
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, build_policy_dataset, chunk_spec, load_demoset, seed_everything,
)
from chunk_trajectory_error.sim2sim import roughness


def magnitude(actions: np.ndarray, continuous: list[int]) -> np.ndarray:
    """Mean over the K steps of the squared translation command ||a[j, xyz]||^2, [N]."""
    return np.mean(np.sum(np.asarray(actions)[..., continuous] ** 2, axis=-1), axis=-1)

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_labels(path: Path) -> tuple[dict, dict]:
    raw = dict(np.load(path))
    ok = raw["status"] == "ok"
    data = {key: value[ok] for key, value in raw.items()}
    return data, {"samples": int(len(ok)), "used": int(ok.sum()), "dropped_not_ok": int((~ok).sum())}


def as_tensors(data: dict) -> dict:
    return {"history": torch.tensor(data["history"]), "state": torch.tensor(data["robot"]),
            "actions": torch.tensor(data["actions"]), "J": torch.tensor(data["J_gap_m2"], dtype=torch.float32)}


def subset(data: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in data.items()}


def train_gap(fit: dict, holdout: dict, spec, cfg: dict, hidden: int, target: str, seed: int) -> tuple[ErrorBundle, list]:
    """The 04 training loop (same loss, optimiser, batches, evaluation cadence, best-checkpoint rule)."""
    seed_everything(seed)
    fit_t, hold_t = as_tensors(fit), as_tensors(holdout)
    bundle = ErrorBundle(spec.history_steps, fit_t["state"].shape[1], spec.evaluation_steps, fit_t["actions"].shape[2], hidden,
                         use_history=True, state_source="robot", target=target)
    bundle.set_normalizer(fit_t["history"], fit_t["state"], fit_t["J"])
    optimizer = torch.optim.Adam(bundle.net.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed)
    best, best_state, curve = float("inf"), None, []
    bundle.train()
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(fit_t["J"]), (cfg["batch_size"],), generator=generator)
        prediction = bundle(fit_t["history"][batch], fit_t["state"][batch], fit_t["actions"][batch], spec.delay)
        loss = error_network_loss(prediction, bundle.label(fit_t["J"][batch]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            with torch.no_grad():
                hold_loss = float(error_network_loss(bundle(hold_t["history"], hold_t["state"], hold_t["actions"], spec.delay),
                                                     bundle.label(hold_t["J"])))
            curve.append({"step": step, "train_loss": float(loss.detach()), "holdout_loss": hold_loss})
            print(f"[{target}] step {step}: train {float(loss.detach()):.4f}  holdout {hold_loss:.4f}  (label-scale units)", flush=True)
            if hold_loss < best:
                best, best_state = hold_loss, {k: v.clone() for k, v in bundle.state_dict().items()}
    bundle.load_state_dict(best_state)
    return bundle.eval(), curve


def predict(bundle: ErrorBundle, history, state, actions, delay: int) -> np.ndarray:
    with torch.no_grad():
        return bundle.cost_m2(torch.as_tensor(history), torch.as_tensor(state), torch.as_tensor(actions), delay).numpy().astype(np.float64)


def dist(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    return {"n": int(len(values)), "p10": float(np.percentile(values, 10)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean": float(values.mean())}


def metrics(predicted: np.ndarray, true: np.ndarray, constant: float) -> dict:
    order = np.argsort(true, kind="stable")
    low = order[: max(1, len(true) // 10)]
    return {
        "samples": int(len(true)),
        "MAE_m2": float(np.abs(predicted - true).mean()),
        "constant_mean_MAE_m2": float(np.abs(constant - true).mean()),
        "MAE_sqrtJ_mm": float(np.abs(np.sqrt(np.maximum(predicted, 0)) - np.sqrt(true)).mean() * 1e3),
        "pearson": float(pearsonr(predicted, true).statistic),
        "spearman": float(spearmanr(predicted, true).statistic),
        "lowest_decile_true_gap": {"n": int(len(low)), "mean_true_m2": float(true[low].mean()), "mean_pred_m2": float(predicted[low].mean())},
        "mean_true_m2": float(true.mean()), "mean_pred_m2": float(predicted.mean()),
    }


def main() -> int:
    parser = ArgumentParser(description="Phase B: train the gap network")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    args = parser.parse_args()
    pilot = load_config(args.config)
    phase = pilot["phase_b"]
    base_config = load_config(pilot["base_config"])
    spec = chunk_spec(base_config, 0)
    shift_name = load_script("10_shift_scan.py").shift_from_entry(phase["shift"]).name
    run_root = Path(pilot["output_dir"])
    out = run_root / shift_name
    seed = int(base_config["experiment"]["seed"])
    cfg = base_config["error_training"]
    hidden = int(base_config["models"]["error_network"]["hidden_dim"])
    continuous = [d for d in range(4) if d not in base_config["action"]["gripper_dims"]]

    train, train_info = load_labels(out / "labels_train.npz")
    val, val_info = load_labels(out / "labels_val.npz")
    episodes = list(dict.fromkeys(train["episode_id"].tolist()))  # collection order
    every = int(phase["gap_training"]["checkpoint_holdout_every"])
    holdout_episodes = set(episodes[every - 1 :: every])
    hold_mask = np.isin(train["episode_id"], list(holdout_episodes))
    fit, holdout = subset(train, ~hold_mask), subset(train, hold_mask)
    constant = float(fit["J_gap_m2"].mean())

    started = time.time()
    report = {
        "shift": shift_name, "label_files": {"train": train_info, "val": val_info},
        "split": {"fit_episodes": len(episodes) - len(holdout_episodes), "fit_samples": int((~hold_mask).sum()),
                  "checkpoint_holdout_episodes": sorted(int(e) for e in holdout_episodes), "checkpoint_holdout_samples": int(hold_mask.sum()),
                  "val_episodes": int(len(np.unique(val["episode_id"]))), "val_samples": int(len(val["J_gap_m2"]))},
        "training": {k: cfg[k] for k in ("steps", "batch_size", "lr", "eval_every")} | {"seed": seed, "hidden_dim": hidden,
                     "state_source": "robot", "use_history": True, "delay": spec.delay},
        "constant_predictor_m2": constant, "networks": {},
    }
    rough_train, rough_val = roughness(train["actions"], continuous), roughness(val["actions"], continuous)
    mag_train, mag_val = magnitude(train["actions"], continuous), magnitude(val["actions"], continuous)
    true_val = val["J_gap_m2"].astype(np.float64)

    demos = load_demoset(base_config)
    demo = build_policy_dataset(demos, "train", spec)
    valid = demo["error_valid"].numpy()
    demo_history, demo_state = demo["history"][valid].numpy(), demo["robot"][valid].numpy()
    demo_actions = demo["chunk"][valid][:, spec.delay : spec.delay + spec.evaluation_steps].numpy()
    policy = PolicyBundle.load(phase["policy_checkpoint"])
    with torch.no_grad():
        policy_actions = policy(demo["obs"][valid]).clamp(-1.0, 1.0)[:, spec.delay : spec.delay + spec.evaluation_steps].numpy()

    gate_cfg = pilot["gate2"]
    for target in ("J", "sqrtJ"):
        bundle, curve = train_gap(fit, holdout, spec, cfg, hidden, target, seed)
        path = out / "checkpoints" / f"gap_{target}.pt"
        bundle.save(path, {"shift": shift_name, "target": target, "fit_samples": int((~hold_mask).sum()), "seed": seed,
                           "label_scale": float(bundle.j_scale)})
        pred_val = predict(bundle, val["history"], val["robot"], val["actions"], spec.delay)
        entry = {"checkpoint": str(path), "curve": curve, "val": metrics(pred_val, true_val, constant)}
        for name, mask in (("val_unperturbed", ~val["perturbed"]), ("val_perturbed", val["perturbed"])):
            entry[name] = metrics(pred_val[mask], true_val[mask], constant)
        v = entry["val"]
        entry["gate2"] = {"MAE_below_constant": v["MAE_m2"] < v["constant_mean_MAE_m2"],
                          "spearman_above": v["spearman"] > gate_cfg["rank_corr_min"]}
        entry["gate2"]["pass"] = bool(all(entry["gate2"].values()))
        entry["roughness"] = {"spearman_prediction_vs_roughness": float(spearmanr(pred_val, rough_val).statistic)}
        entry["magnitude"] = {"spearman_prediction_vs_magnitude": float(spearmanr(pred_val, mag_val).statistic)}
        entry["off_distribution"] = {
            "G_on_logged_val_states_m2": dist(pred_val),
            "G_on_demo_states_demo_actions_m2": dist(predict(bundle, demo_history, demo_state, demo_actions, spec.delay)),
            "G_on_demo_states_pi_bc_chunk_m2": dist(predict(bundle, demo_history, demo_state, policy_actions, spec.delay)),
        }
        report["networks"][target] = entry

    # one-feature predictors: linear fit of the label on roughness, fit on all train labels
    one_feature = {}
    for label_kind in ("J", "sqrtJ"):
        y_train = train["J_gap_m2"] if label_kind == "J" else np.sqrt(train["J_gap_m2"])
        slope, intercept = np.polyfit(rough_train, y_train, 1)
        fitted = intercept + slope * rough_val
        pred = fitted if label_kind == "J" else np.maximum(fitted, 0.0) ** 2
        one_feature[f"linear_{label_kind}_on_roughness"] = {"slope": float(slope), "intercept": float(intercept)} | metrics(pred, true_val, constant)
    report["roughness"] = {
        "definition": "mean over the K-1 consecutive pairs of ||a[j+1, xyz] - a[j, xyz]||^2 of the K commanded actions",
        "spearman_true_gap_vs_roughness_val": float(spearmanr(true_val, rough_val).statistic),
        "pearson_true_gap_vs_roughness_val": float(pearsonr(true_val, rough_val).statistic),
        "one_feature_predictors": one_feature,
        "roughness_logged_val": dist(rough_val), "roughness_logged_train": dist(rough_train),
        "roughness_demo_actions": dist(roughness(demo_actions, continuous)),
        "roughness_pi_bc_chunk_on_demo_states": dist(roughness(policy_actions, continuous)),
    }
    one_feature_mag = {}
    for label_kind in ("J", "sqrtJ"):
        y_train = train["J_gap_m2"] if label_kind == "J" else np.sqrt(train["J_gap_m2"])
        slope, intercept = np.polyfit(mag_train, y_train, 1)
        fitted = intercept + slope * mag_val
        pred = fitted if label_kind == "J" else np.maximum(fitted, 0.0) ** 2
        one_feature_mag[f"linear_{label_kind}_on_magnitude"] = {"slope": float(slope), "intercept": float(intercept)} | metrics(pred, true_val, constant)
    report["magnitude"] = {
        "definition": "mean over the K commanded actions of ||a[j, xyz]||^2 (squared translation command)",
        "spearman_true_gap_vs_magnitude_val": float(spearmanr(true_val, mag_val).statistic),
        "pearson_true_gap_vs_magnitude_val": float(pearsonr(true_val, mag_val).statistic),
        "one_feature_predictors": one_feature_mag,
        "magnitude_logged_val": dist(mag_val), "magnitude_logged_train": dist(mag_train),
        "magnitude_demo_actions": dist(magnitude(demo_actions, continuous)),
        "magnitude_pi_bc_chunk_on_demo_states": dist(magnitude(policy_actions, continuous)),
    }
    report["off_distribution"] = {
        "note": "G is applied in Phase C to source demonstration states; it was trained on logged target states",
        "label_distribution_train_m2": dist(train["J_gap_m2"]), "label_distribution_val_m2": dist(true_val),
        "demo_samples": int(valid.sum()), "demo_episodes": len(demos.ids("train")),
        "history_norm_logged_val_mm": dist(np.linalg.norm(val["history"].reshape(len(val["history"]), -1), axis=1) * 1e3),
        "history_norm_demo_mm": dist(np.linalg.norm(demo_history.reshape(len(demo_history), -1), axis=1) * 1e3),
    }
    report["gate2"] = {"rule": f"on validation: MAE below the constant-mean predictor and global Spearman > {gate_cfg['rank_corr_min']}",
                       "by_network": {t: report["networks"][t]["gate2"]["pass"] for t in report["networks"]}}
    report["seconds"] = time.time() - started
    (out / "gap_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    audit = load_script("00_audit_setup.py")
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("phase_b", {})[f"12_{shift_name}"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gate2": report["gate2"], "split": report["split"],
        "training": report["training"],
        "files": [audit.file_record(p) for p in (out / "checkpoints/gap_J.pt", out / "checkpoints/gap_sqrtJ.pt", out / "gap_report.json", Path(__file__))],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    for target, entry in report["networks"].items():
        v = entry["val"]
        print(f"{target}: MAE {v['MAE_m2']:.3e} vs const {v['constant_mean_MAE_m2']:.3e}, pearson {v['pearson']:.3f}, "
              f"spearman {v['spearman']:.3f}, lowest decile true/pred {v['lowest_decile_true_gap']['mean_true_m2']:.2e}/"
              f"{v['lowest_decile_true_gap']['mean_pred_m2']:.2e}  GATE 2: {'PASS' if entry['gate2']['pass'] else 'FAIL'}")
    print(json.dumps(report["roughness"], indent=2, default=float))
    print(json.dumps(report["magnitude"], indent=2, default=float))
    print(json.dumps({t: e["off_distribution"] for t, e in report["networks"].items()} | report["off_distribution"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
