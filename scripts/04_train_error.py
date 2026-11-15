"""Train the trajectory-error network on simulator labels.

Variants: ``history`` (default), ``--no-history`` (history input zeroed) and
``--state-source full_obs`` (object/goal information added; extra ablation).
Normalisation statistics and the label scale are fit on the train split only.

Outputs: ``checkpoints/error_round<R>[_<tag>].pt`` and ``error_report_<tag>.json`` with
held-out MAE, within-anchor ranking and constant / no-input baselines.
"""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import torch
from scipy.stats import spearmanr

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.error_net import error_network_loss
from chunk_trajectory_error.pipeline import (
    ErrorBundle, chunk_spec, load_branches, load_demoset, out_dir, seed_everything, select_state, write_json,
)


def tensors(demos, table, spec, state_source):
    history = np.stack([demos.history(e, t, spec.history_steps) for e, t in zip(table["episode_id"], table["t"])])
    state = np.stack([select_state(demos, e, t, state_source) for e, t in zip(table["episode_id"], table["t"])])
    keep = table["rollout_status"] == "ok"
    return {
        "history": torch.tensor(history[keep]), "state": torch.tensor(state[keep]),
        "actions": torch.tensor(table["action_cmd"][keep]), "J": torch.tensor(table["J_m2"][keep], dtype=torch.float32),
        "anchor_id": table["anchor_id"][keep], "candidate_type": table["candidate_type"][keep],
    }


def evaluate(bundle: ErrorBundle, data: dict, delay: int, train_mean_j: float) -> dict:
    with torch.no_grad():
        predicted = bundle.cost_m2(data["history"], data["state"], data["actions"], delay).numpy()
    target = data["J"].numpy()
    correlations, top1 = [], []
    for anchor in np.unique(data["anchor_id"]):
        mask = data["anchor_id"] == anchor
        if mask.sum() >= 3 and np.ptp(target[mask]) > 0:
            correlations.append(spearmanr(predicted[mask], target[mask]).statistic)
            top1.append(int(np.argmin(predicted[mask]) == np.argmin(target[mask])))
    per_type = {}
    for candidate_type in np.unique(data["candidate_type"]):
        mask = data["candidate_type"] == candidate_type
        per_type[str(candidate_type)] = {
            "count": int(mask.sum()), "MAE_m2": float(np.abs(predicted[mask] - target[mask]).mean()),
            "mean_true_J_m2": float(target[mask].mean()), "mean_pred_J_m2": float(predicted[mask].mean()),
        }
    return {
        "samples": int(len(target)),
        "MAE_m2": float(np.abs(predicted - target).mean()),
        "RMSE_mm2": float(np.sqrt(((predicted - target) ** 2).mean()) * 1e6),
        "MAE_sqrtJ_mm": float(np.abs(np.sqrt(np.maximum(predicted, 0)) - np.sqrt(target)).mean() * 1e3),
        "constant_baseline_MAE_m2": float(np.abs(train_mean_j - target).mean()),
        "constant_median_baseline_MAE_m2": float(np.abs(np.median(target) - target).mean()),
        "pearson": float(np.corrcoef(predicted, target)[0, 1]),
        "spearman_global": float(spearmanr(predicted, target).statistic),
        "within_anchor_spearman_mean": float(np.nanmean(correlations)),
        "within_anchor_best_candidate_match": float(np.mean(top1)),
        "anchors_ranked": len(correlations),
        "by_candidate_type": per_type,
    }


def main() -> int:
    parser = ArgumentParser(description="Train the history-conditioned trajectory-error network")
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--state-source", choices=("robot", "full_obs"), default="robot")
    parser.add_argument("--target", choices=("J", "sqrtJ"), default="J",
                        help="regression target; 'sqrtJ' is an exploratory deviation from the specified loss on J")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    spec, cfg = chunk_spec(config), config["error_training"]
    run_dir = out_dir(config)
    demos = load_demoset(config)
    data = {split: tensors(demos, load_branches(run_dir / f"branches_{split}.h5"), spec, args.state_source) for split in ("train", "val", "test")}
    # Refresh rounds aggregate: train on the round-0 branches plus every later round's branches (collected
    # with the policies fine-tuned against earlier critics).  Round-0 val/test stay the fixed benchmark.
    rounds_used = [0]
    for extra_round in range(1, args.round + 1):
        extra = tensors(demos, load_branches(run_dir / f"branches_train_r{extra_round}.h5"), spec, args.state_source)
        for key in data["train"]:
            joiner = torch.cat if isinstance(extra[key], torch.Tensor) else np.concatenate
            data["train"][key] = joiner([data["train"][key], extra[key]])
        rounds_used.append(extra_round)
    refresh_val = run_dir / f"branches_val_r{args.round}.h5"
    if args.round > 0 and refresh_val.is_file():
        data["val_refresh"] = tensors(demos, load_branches(refresh_val), spec, args.state_source)
    train = data["train"]

    bundle = ErrorBundle(
        spec.history_steps, train["state"].shape[1], spec.evaluation_steps, train["actions"].shape[2],
        config["models"]["error_network"]["hidden_dim"], use_history=not args.no_history, state_source=args.state_source, target=args.target,
    )
    bundle.set_normalizer(train["history"], train["state"], train["J"])
    scale = float(bundle.j_scale)
    optimizer = torch.optim.Adam(bundle.net.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    best, best_state, curve = float("inf"), None, []
    bundle.train()
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(train["J"]), (cfg["batch_size"],), generator=generator)
        prediction = bundle(train["history"][batch], train["state"][batch], train["actions"][batch], spec.delay)
        loss = error_network_loss(prediction, bundle.label(train["J"][batch]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            with torch.no_grad():
                val_prediction = bundle(data["val"]["history"], data["val"]["state"], data["val"]["actions"], spec.delay)
                val_loss = float(error_network_loss(val_prediction, bundle.label(data["val"]["J"])))
            curve.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss})
            print(f"step {step}: train {float(loss.detach()):.4f}  val {val_loss:.4f}  (label-scale units)")
            if val_loss < best:
                best, best_state = val_loss, {k: v.clone() for k, v in bundle.state_dict().items()}
    bundle.load_state_dict(best_state)
    bundle.eval()

    tag = ("nohist" if args.no_history else "history") + ("" if args.state_source == "robot" else "_fullobs")
    tag += "" if args.target == "J" else "_sqrt"
    ckpt = run_dir / "checkpoints" / f"error_round{args.round}_{tag}.pt"
    bundle.save(ckpt, {"round": args.round, "tag": tag, "target": args.target, "label_scale": scale, "best_val_loss_scaled": best})
    train_mean = float(train["J"].mean())
    report = {
        "round": args.round, "train_rounds": rounds_used,
        "checkpoint": str(ckpt), "tag": tag, "target": args.target, "label_scale": scale, "train_samples": int(len(train["J"])),
        "curve": curve,
        "val": evaluate(bundle, data["val"], spec.delay, train_mean),
        "test": evaluate(bundle, data["test"], spec.delay, train_mean),
    }
    if "val_refresh" in data:
        report["val_refresh_policy_distribution"] = evaluate(bundle, data["val_refresh"], spec.delay, train_mean)
    write_json(run_dir / f"error_report_{tag}{'' if args.round == 0 else f'_r{args.round}'}.json", report)
    for split in ("val", "test"):
        print(split, {k: v for k, v in report[split].items() if k != "by_candidate_type"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
