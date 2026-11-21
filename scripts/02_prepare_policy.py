"""Train the initial chunk policy pi0 by behaviour cloning and record its baseline.

``--policy mlp`` (default) regresses the chunk with an MLP; ``--policy flow`` trains a
conditional flow-matching policy (``models.policy_flow``) on the same data and budget.

Outputs: ``checkpoints/pi0.pt`` / ``pi0_baseline.json`` (MLP) or ``checkpoints/pi0_flow.pt`` /
``pi0_flow_baseline.json`` (flow): val loss curve, val true J, val closed-loop success at r=0.
For the flow policy the baseline also records ``val_sample_mse``, the MSE between one sampled
chunk and the expert chunk, which is the quantity comparable to the MLP's val BC loss.
"""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import (
    PolicyBundle, build_policy_dataset, chunk_spec, closed_loop_episode, load_demoset, out_dir,
    policy_true_cost, seed_everything, write_json,
)
from chunk_trajectory_error.policies.mlp_bc import behavior_cloning_loss


def main() -> int:
    parser = ArgumentParser(description="Train pi0 (behaviour cloning) and record the initial baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-episodes", type=int, default=30)
    parser.add_argument("--policy", choices=("mlp", "flow"), default="mlp")
    parser.add_argument("--steps", type=int, default=None, help="override bc.steps (the flow policy may need a longer budget)")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None,
                        help="override bc.lr_schedule; cosine decays to lr / 100 over the run")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    spec, bc = chunk_spec(config), dict(config["bc"])
    if args.policy == "flow" and "flow" in bc:  # per-class overrides from the config (bc.flow: {steps, lr_schedule, lr})
        bc.update(bc["flow"])
    if args.steps:
        bc["steps"] = args.steps
    if args.lr_schedule:
        bc["lr_schedule"] = args.lr_schedule
    bc.setdefault("lr_schedule", "constant")
    demos = load_demoset(config)
    train, val = build_policy_dataset(demos, "train", spec), build_policy_dataset(demos, "val", spec)

    if args.policy == "flow":
        pf = config["models"]["policy_flow"]
        bundle = PolicyBundle(
            train["obs"].shape[1], train["chunk"].shape[2], spec.horizon, pf["hidden_dims"], policy_type="flow_chunk",
            flow={"time_frequencies": pf["time_frequencies"], "sample_steps": pf["sample_steps"]}, sample_seed=config["experiment"]["seed"],
        )
        suffix = "_flow"
    else:
        bundle = PolicyBundle(train["obs"].shape[1], train["chunk"].shape[2], spec.horizon, config["models"]["policy"]["hidden_dims"])
        suffix = ""
    bundle.set_normalizer(train["obs"])
    optimizer = torch.optim.Adam(bundle.policy.parameters(), lr=bc["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=bc["steps"], eta_min=bc["lr"] / 100) if bc["lr_schedule"] == "cosine" else None
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    noise_generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 1)  # tau / x0 draws of the flow loss
    # fixed validation draws so that the selection criterion is deterministic for the flow policy
    val_generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 2)
    val_tau = torch.rand(len(val["obs"]), generator=val_generator)
    val_noise = torch.randn(val["chunk"].shape, generator=val_generator)
    best, best_state, curve = float("inf"), None, []
    bundle.train()
    for step in range(1, bc["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (bc["batch_size"],), generator=generator)
        loss = bundle.imitation_loss(train["obs"][batch], train["chunk"][batch], generator=noise_generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % bc["eval_every"] == 0 or step == bc["steps"]:
            with torch.no_grad():
                if bundle.is_flow:
                    val_loss = float(bundle.policy.flow_matching_loss(bundle.normalize(val["obs"]), val["chunk"], tau=val_tau, noise=val_noise))
                else:
                    val_loss = float(behavior_cloning_loss(bundle(val["obs"]), val["chunk"]))
            curve.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss})
            print(f"step {step}: train {float(loss.detach()):.5f}  val {val_loss:.5f}", flush=True)
            if val_loss < best:
                best, best_state = val_loss, {k: v.clone() for k, v in bundle.state_dict().items()}
    bundle.load_state_dict(best_state)
    bundle.eval()

    run_dir = out_dir(config)
    ckpt = run_dir / "checkpoints" / f"pi0{suffix}.pt"
    extra = {"stage": f"pi0{suffix}", "best_val_loss": best, "train_samples": len(train["obs"]), "policy": args.policy}
    sample_mse = None
    if bundle.is_flow:
        with torch.no_grad():
            bundle.reseed()
            sample_mse = float(behavior_cloning_loss(bundle(val["obs"]).clamp(-1.0, 1.0), val["chunk"]))
        extra["val_sample_mse"] = sample_mse
    else:
        extra["best_val_bc_loss"] = best
    bundle.save(ckpt, extra)

    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config)
    cost_rows = policy_true_cost(backend, demos, bundle, "val", spec, config["branches"]["anchors_per_episode"]["val"])
    costs = np.array([row["J_m2"] for row in cost_rows], dtype=np.float64)
    successes = []
    for episode_id in demos.ids("val")[: args.baseline_episodes]:
        episode = demos.episodes[episode_id]
        record, _ = closed_loop_episode(
            backend, bundle, spec, 0, episode.seed, episode.state_at(0),
            config["environment"]["max_episode_steps"], demos.gripper_dims,
        )
        successes.append(record["success"])
    bundle.reseed()
    backend.close()
    baseline = {
        "checkpoint": str(ckpt),
        "policy": args.policy,
        "parameters": int(sum(p.numel() for p in bundle.policy.parameters())),
        "train_samples": len(train["obs"]),
        "train_episodes": len(demos.ids("train")),
        "training": {"steps": bc["steps"], "lr": bc["lr"], "lr_schedule": bc["lr_schedule"], "batch_size": bc["batch_size"],
                     "hidden_dims": list(bundle.meta["hidden_dims"])},
        "best_val_bc_loss" if not bundle.is_flow else "best_val_flow_matching_loss": best,
        "curve": curve,
        "val_true_J_m2_mean": float(np.nanmean(costs)),
        "val_true_sqrtJ_mm_mean": float(np.nanmean(np.sqrt(costs)) * 1e3),
        "val_anchors": len(costs),
        "val_closed_loop_r0": {"episodes": len(successes), "successes": int(sum(successes))},
    }
    if sample_mse is not None:
        baseline["val_sample_mse"] = sample_mse  # comparable to the MLP's val BC loss
    write_json(run_dir / f"pi0{suffix}_baseline.json", baseline)
    print({k: v for k, v in baseline.items() if k != "curve"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
