"""Train the initial chunk policy pi0 by behaviour cloning and record its baseline.

Outputs: ``checkpoints/pi0.pt``, ``pi0_baseline.json`` (val loss curve, val true J,
val closed-loop success at r=0).
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
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    spec, bc = chunk_spec(config), config["bc"]
    demos = load_demoset(config)
    train, val = build_policy_dataset(demos, "train", spec), build_policy_dataset(demos, "val", spec)

    bundle = PolicyBundle(train["obs"].shape[1], train["chunk"].shape[2], spec.horizon, config["models"]["policy"]["hidden_dims"])
    bundle.set_normalizer(train["obs"])
    optimizer = torch.optim.Adam(bundle.policy.parameters(), lr=bc["lr"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    best, best_state, curve = float("inf"), None, []
    bundle.train()
    for step in range(1, bc["steps"] + 1):
        batch = torch.randint(0, len(train["obs"]), (bc["batch_size"],), generator=generator)
        loss = behavior_cloning_loss(bundle(train["obs"][batch]), train["chunk"][batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % bc["eval_every"] == 0 or step == bc["steps"]:
            with torch.no_grad():
                val_loss = float(behavior_cloning_loss(bundle(val["obs"]), val["chunk"]))
            curve.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss})
            print(f"step {step}: train {float(loss.detach()):.5f}  val {val_loss:.5f}", flush=True)
            if val_loss < best:
                best, best_state = val_loss, {k: v.clone() for k, v in bundle.state_dict().items()}
    bundle.load_state_dict(best_state)
    bundle.eval()

    run_dir = out_dir(config)
    ckpt = run_dir / "checkpoints" / "pi0.pt"
    bundle.save(ckpt, {"stage": "pi0", "best_val_bc_loss": best, "train_samples": len(train["obs"])})

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
    backend.close()
    baseline = {
        "checkpoint": str(ckpt),
        "train_samples": len(train["obs"]),
        "train_episodes": len(demos.ids("train")),
        "best_val_bc_loss": best,
        "curve": curve,
        "val_true_J_m2_mean": float(np.nanmean(costs)),
        "val_true_sqrtJ_mm_mean": float(np.nanmean(np.sqrt(costs)) * 1e3),
        "val_anchors": len(costs),
        "val_closed_loop_r0": {"episodes": len(successes), "successes": int(sum(successes))},
    }
    write_json(run_dir / "pi0_baseline.json", baseline)
    print({k: v for k, v in baseline.items() if k != "curve"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
