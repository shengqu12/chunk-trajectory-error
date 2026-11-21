"""Gradient audit for the flow-matching policy: does the frozen critic's gradient, taken
through the Euler sampler, move the *policy parameters* in a direction that lowers the
simulator-recomputed J?

Stage 05 audits d(E)/d(actions) on stored candidate chunks and is policy-independent.  For
the flow policy the open question is the extra path through the sampler, so this stage:

1. checks that d(E(sample(s)))/d(theta) is finite and non-zero on every policy parameter
   while the critic receives no parameter gradient;
2. starting from pi0_flow, takes Adam steps on the critic term alone (no imitation loss)
   and, at a few checkpoints, recomputes J in the simulator on validation anchors.  A control
   run takes steps of the same size in random directions.

Gate: at some checkpoint with <= ``--gate-steps`` updates, the paired change in J from
pi0_flow is below zero (95% CI) and is below the random-direction change (95% CI).

Outputs: ``flow_gradient_audit.json``.
"""

from __future__ import annotations

import copy
from argparse import ArgumentParser

import numpy as np
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.objectives import set_trainable
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, bootstrap_ci, build_policy_dataset, chunk_spec, load_demoset, out_dir, policy_true_cost,
    seed_everything, write_json,
)


def critic_term(bundle: PolicyBundle, error: ErrorBundle, data: dict, batch: torch.Tensor, r: int, K: int) -> torch.Tensor:
    valid = data["error_valid"][batch]
    chunk = bundle(data["obs"][batch])
    state = data["obs"][batch] if error.meta["state_source"] == "full_obs" else data["robot"][batch]
    tail = chunk[valid][:, r : r + K].clamp(-1.0, 1.0)
    return error.normalized_cost(data["history"][batch][valid], state[valid], tail, r).mean()


def val_cost(backend, demos, bundle, spec, anchors_per_episode, episode_limit) -> np.ndarray:
    bundle.eval()
    bundle.reseed()
    rows = policy_true_cost(backend, demos, bundle, "val", spec, anchors_per_episode, episode_limit=episode_limit)
    return np.array([[row["episode_id"], row["J_m2"]] for row in rows], dtype=np.float64)


def main() -> int:
    parser = ArgumentParser(description="Audit the critic gradient through the flow sampler")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", default="10,25,50,100,200", help="update counts at which J is recomputed")
    parser.add_argument("--gate-steps", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=40, help="validation episodes used for the simulator J")
    parser.add_argument("--critic", default="error_round0_history.pt")
    parser.add_argument("--lr", type=float, default=None, help="Adam step size (default: policy_finetune.lr)")
    parser.add_argument("--tag", default=None, help="suffix for the output file (flow_gradient_audit_<tag>.json)")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    spec, cfg = chunk_spec(config), dict(config["policy_finetune"])
    if args.lr:
        cfg["lr"] = args.lr
    run_dir = out_dir(config)
    demos = load_demoset(config)
    train = build_policy_dataset(demos, "train", spec)
    error = ErrorBundle.load(run_dir / "checkpoints" / args.critic)
    set_trainable(error, False)
    r, K = spec.delay, spec.evaluation_steps
    pi0 = PolicyBundle.load(run_dir / "checkpoints" / "pi0_flow.pt")
    if not pi0.is_flow:
        raise SystemExit("pi0_flow.pt is not a flow policy")

    # --- 1. parameter gradient through the sampler -------------------------- #
    probe = copy.deepcopy(pi0)
    probe.train()
    probe.reseed()
    generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 11)
    batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=generator)
    term = critic_term(probe, error, train, batch, r, K)
    grads = torch.autograd.grad(term, list(probe.policy.parameters()), allow_unused=True)
    per_param = [float(g.norm()) if g is not None else 0.0 for g in grads]
    gradient_check = {
        "critic_term": float(term.detach()),
        "policy_parameters": len(per_param),
        "parameters_with_gradient": int(sum(n > 0 for n in per_param)),
        "all_finite": bool(all(np.isfinite(n) for n in per_param)),
        "total_grad_norm": float(np.sqrt(sum(n**2 for n in per_param))),
        "critic_parameters_require_grad": int(sum(p.requires_grad for p in error.parameters())),
        "pass": bool(all(n > 0 and np.isfinite(n) for n in per_param)) and not any(p.requires_grad for p in error.parameters()),
    }
    print("gradient check:", gradient_check, flush=True)

    # --- 2. descent on the critic term vs random directions ----------------- #
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config)
    anchors = config["branches"]["anchors_per_episode"]["val"]
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",")})
    base = val_cost(backend, demos, pi0, spec, anchors, args.val_episodes)
    groups = base[:, 0]
    results = {"baseline_mean_J_m2": float(np.nanmean(base[:, 1])), "anchors": int(len(base)), "runs": {}}

    for mode in ("gradient", "random"):
        bundle = copy.deepcopy(pi0)
        bundle.train()
        bundle.reseed()
        params = list(bundle.policy.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])
        batch_gen = torch.Generator().manual_seed(config["experiment"]["seed"] + 12)  # identical batches in both runs
        noise_gen = torch.Generator().manual_seed(config["experiment"]["seed"] + 13)
        rows = []
        for step in range(1, max(checkpoints) + 1):
            batch = torch.randint(0, len(train["obs"]), (cfg["batch_size"],), generator=batch_gen)
            term = critic_term(bundle, error, train, batch, r, K)
            optimizer.zero_grad()
            term.backward()
            if mode == "random":  # same global norm, random direction
                norm = float(torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None)))
                randoms = [torch.randn(p.shape, generator=noise_gen) for p in params]
                rnorm = float(torch.sqrt(sum((g**2).sum() for g in randoms)))
                for p, g in zip(params, randoms):
                    p.grad = g * (norm / max(rnorm, 1e-12))
            optimizer.step()
            if step in checkpoints:
                with torch.no_grad():
                    bundle.eval()
                    bundle.reseed()
                    probe_batch = torch.arange(0, len(train["obs"]), max(1, len(train["obs"]) // 2048))
                    predicted = float(critic_term(bundle, error, train, probe_batch, r, K))
                cost = val_cost(backend, demos, bundle, spec, anchors, args.val_episodes)
                bundle.train()
                delta = cost[:, 1] - base[:, 1]
                mean, lo, hi = bootstrap_ci(delta, groups)
                rows.append({
                    "updates": step, "predicted_normalized_E_train": predicted, "mean_J_m2": float(np.nanmean(cost[:, 1])),
                    "delta_J_m2": {"mean": mean, "ci95": [lo, hi]}, "relative": float(mean / np.nanmean(base[:, 1])),
                    "_delta": delta.tolist(),
                })
                print(f"{mode} {step:4d} updates: predicted E {predicted:.4f}  mean J {np.nanmean(cost[:, 1]):.3e}  "
                      f"dJ {mean:+.3e} [{lo:+.2e}, {hi:+.2e}]", flush=True)
        results["runs"][mode] = rows
    backend.close()

    # paired gradient - random at matching update counts
    comparison = []
    for g_row, r_row in zip(results["runs"]["gradient"], results["runs"]["random"]):
        diff = np.array(g_row["_delta"]) - np.array(r_row["_delta"])
        mean, lo, hi = bootstrap_ci(diff, groups)
        comparison.append({"updates": g_row["updates"], "gradient_minus_random_J_m2": {"mean": mean, "ci95": [lo, hi]}})
    for rows in results["runs"].values():
        for row in rows:
            row.pop("_delta")
    gate_rows = [
        (g, c) for g, c in zip(results["runs"]["gradient"], comparison)
        if g["updates"] <= args.gate_steps and g["delta_J_m2"]["ci95"][1] < 0 and c["gradient_minus_random_J_m2"]["ci95"][1] < 0
    ]
    results.update({
        "gradient_check": gradient_check,
        "gradient_minus_random": comparison,
        "gate": {
            "rule": f"at some checkpoint <= {args.gate_steps} updates on the critic term alone, 95% CI of dJ below 0 and 95% CI of (gradient - random) below 0",
            "pass": bool(gradient_check["pass"] and gate_rows),
            "passing_checkpoints": [g["updates"] for g, _ in gate_rows],
        },
        "settings": {"lr": cfg["lr"], "batch_size": cfg["batch_size"], "critic": args.critic, "val_episodes": args.val_episodes},
    })
    write_json(run_dir / ("flow_gradient_audit.json" if not args.tag else f"flow_gradient_audit_{args.tag}.json"), results)
    print("gate:", results["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
