"""Fine-tune the policy from pi0 with the same data, steps and batches for every variant.

Variants
  bc             L_imit only (lambda = 0)
  error          L_imit + lambda * E_history
  nohist         L_imit + lambda * E_no_history            (ablation)
  error_fullobs  L_imit + lambda * E_history_fullobs       (extra ablation)
  smooth         L_imit + beta * mean ||a_{h+1} - a_h||^2  (continuous channels only)

The weight is selected on the validation split by the simulator-recomputed J of the
policy's own chunk; the test split is not touched here.

``--policy flow`` runs the same variants for the flow-matching policy: it starts from
``checkpoints/pi0_flow.pt``, the imitation term is the conditional flow-matching loss, and the
auxiliary term (critic or smoothness) is applied to a chunk sampled with the Euler integrator,
so its gradient reaches the parameters through every sampling step.  Names get a ``_flow``
suffix (``bc_flow``, ``error_flow``, ``smooth_flow``).

Outputs: ``checkpoints/pi_<name>/w<weight>.pt``, ``checkpoints/pi_<name>/pi.pt``
(selected) and ``policy_training_<name>.json``.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser

import numpy as np
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.objectives import set_trainable
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, build_policy_dataset, chunk_spec, load_demoset, out_dir, policy_true_cost,
    seed_everything, write_json,
)
from chunk_trajectory_error.policies.mlp_bc import behavior_cloning_loss

ERROR_TAGS = {
    "error": "history", "nohist": "nohist", "error_fullobs": "history_fullobs",
    "error_sqrt": "history_sqrt", "error_fullobs_sqrt": "history_fullobs_sqrt",  # exploratory, outside the specified loss
}


def grad_norm(loss: torch.Tensor, parameters: list) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g**2).sum() for g in grads if g is not None)))


def train_one(config, spec, train, val, pi0_path, variant, weight, error, continuous) -> tuple[PolicyBundle, dict]:
    cfg = config["policy_finetune"]
    seed_everything(config["experiment"]["seed"])
    bundle = PolicyBundle.load(pi0_path)
    bundle.train()
    parameters = list(bundle.policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 6)  # identical batches for all variants
    noise_generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 7)  # flow loss tau / x0, identical for all variants
    r, K = spec.delay, spec.evaluation_steps
    log = []
    for step in range(1, cfg["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=generator)
        if bundle.is_flow:
            imitation = bundle.imitation_loss(train["obs"][batch], train["chunk"][batch], generator=noise_generator)
            # the auxiliary terms act on a sampled chunk; sampling is skipped when no term is active
            chunk = bundle(train["obs"][batch]) if weight > 0 else None
            auxiliary = imitation * 0.0
        else:
            chunk = bundle(train["obs"][batch])
            imitation = behavior_cloning_loss(chunk, train["chunk"][batch])
            auxiliary = chunk.sum() * 0.0
        if variant in ERROR_TAGS and weight > 0:
            valid = train["error_valid"][batch]
            if valid.any():
                state = train["obs"][batch] if error.meta["state_source"] == "full_obs" else train["robot"][batch]
                tail = chunk[valid][:, r : r + K].clamp(-1.0, 1.0)  # executor clips commands to the action box
                # frozen critic: no parameter gradients, but d(error)/d(actions) is kept
                auxiliary = error.normalized_cost(train["history"][batch][valid], state[valid], tail, r).mean()
        elif variant == "smooth" and weight > 0:
            auxiliary = ((chunk[:, 1:, continuous] - chunk[:, :-1, continuous]) ** 2).mean()
        total = imitation + weight * auxiliary
        if step % 500 == 0 or step == 1:
            entry = {"step": step, "imitation": float(imitation.detach()), "auxiliary": float(auxiliary.detach()), "grad_norm_imitation": grad_norm(imitation, parameters)}
            entry["grad_norm_weighted_auxiliary"] = grad_norm(weight * auxiliary, parameters) if weight > 0 and auxiliary.requires_grad else 0.0
            log.append(entry)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    bundle.eval()
    info = {"weight": weight}
    with torch.no_grad():
        if bundle.is_flow:
            bundle.reseed()
            val_chunk = bundle(val["obs"]).clamp(-1.0, 1.0)
            val_generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 2)
            val_tau = torch.rand(len(val["obs"]), generator=val_generator)
            val_noise = torch.randn(val["chunk"].shape, generator=val_generator)
            info["val_flow_matching_loss"] = float(bundle.policy.flow_matching_loss(bundle.normalize(val["obs"]), val["chunk"], tau=val_tau, noise=val_noise))
            bundle.reseed()
        else:
            val_chunk = bundle(val["obs"])
        val_bc = float(behavior_cloning_loss(val_chunk, val["chunk"]))  # for the flow policy: MSE of one sampled chunk
        val_smooth = float(((val_chunk[:, 1:, continuous] - val_chunk[:, :-1, continuous]) ** 2).mean())
    info.update({"val_bc_loss": val_bc, "val_chunk_roughness": val_smooth, "log": log})
    return bundle, info


def main() -> int:
    parser = ArgumentParser(description="Fine-tune pi0 with or without the frozen surrogate (same budget for all variants)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("bc", "error", "nohist", "error_fullobs", "smooth", "error_sqrt", "error_fullobs_sqrt"), required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="starting policy (default: checkpoints/pi0.pt)")
    parser.add_argument("--name", default=None, help="output name (default: variant, or variant_r<round> for refresh rounds)")
    parser.add_argument("--weights", default=None,
                        help="comma-separated weights to add; results already recorded for this variant are kept and merged")
    parser.add_argument("--policy", choices=("mlp", "flow"), default="mlp", help="policy class (flow: start from pi0_flow, names get _flow)")
    args = parser.parse_args()
    config = load_config(args.config)
    spec, cfg = chunk_spec(config), config["policy_finetune"]
    run_dir = out_dir(config)
    suffix = "_flow" if args.policy == "flow" else ""
    name = args.name or (f"{args.variant}{suffix}" if args.round == 0 else f"{args.variant}_r{args.round}{suffix}")
    demos = load_demoset(config)
    train, val = build_policy_dataset(demos, "train", spec), build_policy_dataset(demos, "val", spec)
    pi0_path = args.checkpoint or run_dir / "checkpoints" / f"pi0{suffix}.pt"
    continuous = [d for d in range(train["chunk"].shape[2]) if d not in demos.gripper_dims]

    error = None
    if args.variant in ERROR_TAGS:
        error = ErrorBundle.load(run_dir / "checkpoints" / f"error_round{args.round}_{ERROR_TAGS[args.variant]}.pt")
        set_trainable(error, False)
        weights = [w for w in cfg["lambdas"] if w > 0]
    elif args.variant == "smooth":
        weights = list(cfg["smooth_betas"])
    else:
        weights = [0.0]

    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    record_path = run_dir / f"policy_training_{name}.json"
    previous = []
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if record_path.is_file():
            previous = [c for c in json.loads(record_path.read_text())["candidates"] if c["weight"] not in weights]
    backend = ManiSkillBackend(config)
    anchors_per_episode = config["branches"]["anchors_per_episode"]["val"]
    candidates = [(info, None) for info in previous]
    for weight in weights:
        bundle, info = train_one(config, spec, train, val, pi0_path, args.variant, float(weight), error, continuous)
        costs = np.array([row["J_m2"] for row in policy_true_cost(backend, demos, bundle, "val", spec, anchors_per_episode)])
        info["val_true_J_m2_mean"] = float(np.nanmean(costs))
        info["val_true_sqrtJ_mm_mean"] = float(np.nanmean(np.sqrt(costs)) * 1e3)
        path = run_dir / "checkpoints" / f"pi_{name}" / f"w{weight:g}.pt"
        bundle.save(path, {"variant": args.variant, "weight": float(weight), "start": str(pi0_path), "steps": cfg["steps"]})
        info["checkpoint"] = str(path)
        candidates.append((info, bundle))
        print(f"{name} weight={weight:g}: val true J {info['val_true_J_m2_mean']:.3e} m^2 "
              f"(sqrtJ {info['val_true_sqrtJ_mm_mean']:.3f} mm), val BC {info['val_bc_loss']:.5f}")
    backend.close()

    candidates.sort(key=lambda item: item[0]["weight"])
    chosen_info, chosen_bundle = min(candidates, key=lambda item: item[0]["val_true_J_m2_mean"])
    if chosen_bundle is None:
        chosen_bundle = PolicyBundle.load(chosen_info["checkpoint"])
    selected = run_dir / "checkpoints" / f"pi_{name}" / "pi.pt"
    chosen_bundle.save(selected, {"variant": args.variant, "weight": chosen_info["weight"], "start": str(pi0_path), "steps": cfg["steps"]})
    write_json(
        record_path,
        {
            "name": name, "variant": args.variant, "policy": args.policy, "round": args.round, "start_checkpoint": str(pi0_path), "steps": cfg["steps"], "lr": cfg["lr"],
            "batch_size": cfg["batch_size"], "selection_rule": "minimum simulator-recomputed J of the policy chunk on val anchors",
            "selected_weight": chosen_info["weight"], "selected_checkpoint": str(selected),
            "candidates": [info for info, _ in candidates],
        },
    )
    print("selected weight:", chosen_info["weight"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
