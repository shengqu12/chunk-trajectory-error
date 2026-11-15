"""Collect simulator-labelled branches at expert anchors with the frozen policy.

Every anchor gets one expert branch (reference; J must be 0), the frozen policy's own
chunk, and bounded low-frequency perturbations of both.  Candidate and expert branches
start from the same restored state and run a fixed K steps.

Outputs per split: ``anchors_<split>.parquet``, ``branches_<split>.h5``,
``branch_stats_<split>.json``.  ``--sample`` prints the two-row acceptance sample
(expert replay J = 0 / bounded perturbation J > 0) for one anchor and exits.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import (
    PolicyBundle, anchor_snapshot, anchor_times, branch_cost, chunk_spec, load_branches, load_demoset,
    low_frequency_perturbation, out_dir, rollout_branch, write_json,
)


def main() -> int:
    parser = ArgumentParser(description="Generate candidates at expert anchors and label them in simulation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--checkpoint", default=None, help="frozen policy (default: checkpoints/pi0.pt)")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--sample", action="store_true", help="two-row acceptance sample for the first anchor")
    parser.add_argument("--policy-perturbed", type=int, default=None, help="override branches.policy_perturbed_per_anchor")
    parser.add_argument("--expert-perturbed", type=int, default=None, help="override branches.expert_perturbed_per_anchor")
    args = parser.parse_args()
    config = load_config(args.config)
    spec, cfg = chunk_spec(config), dict(config["branches"])
    if args.policy_perturbed is not None:
        cfg["policy_perturbed_per_anchor"] = args.policy_perturbed
    if args.expert_perturbed is not None:
        cfg["expert_perturbed_per_anchor"] = args.expert_perturbed
    suffix = "" if args.round == 0 else f"_r{args.round}"  # refresh rounds keep the round-0 files intact
    run_dir = out_dir(config)
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "pi0.pt"
    bundle = PolicyBundle.load(ckpt)
    demos = load_demoset(config)
    r, K = spec.delay, spec.evaluation_steps
    action_dim = bundle.meta["action_dim"]
    seed = config["experiment"]["seed"] + 1000 * args.round + {"train": 0, "val": 1, "test": 2}[args.split]
    rng = np.random.default_rng(seed)

    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config)
    anchors, stats = [], {"aborted": 0, "branches": 0, "clipped_candidates": 0, "tcp_check_max_m": 0.0}
    h5_path = run_dir / f"branches_{args.split}{suffix}.h5"
    episode_ids = demos.ids(args.split)[:1] if args.sample else demos.ids(args.split)
    sample_rows = []
    with h5py.File(run_dir / "branches_sample.h5" if args.sample else h5_path, "w") as store:
        store.attrs.update({"policy_ckpt": str(ckpt), "round": args.round, "seed": seed, "split": args.split, "K": K, "r": r})
        for episode_id in episode_ids:
            episode = demos.episodes[episode_id]
            backend.reset(seed=episode.seed, init_state=episode.state_at(0))
            for t in anchor_times(episode.num_actions, spec, cfg["anchors_per_episode"][args.split]):
                anchor_id = f"ep{episode_id:04d}_t{t:04d}_r{r}"
                snapshot = anchor_snapshot(episode, t, r)
                # the recorded observation must describe the restored state
                backend.restore(snapshot)
                tcp_gap = float(np.linalg.norm(np.asarray(backend.ee_position()) - demos.tcp(episode_id)[t + r]))
                stats["tcp_check_max_m"] = max(stats["tcp_check_max_m"], tcp_gap)
                if tcp_gap > 1e-4:
                    raise RuntimeError(f"{anchor_id}: observation tcp slice disagrees with the simulator by {tcp_gap} m")

                expert_actions = episode.actions[t + r : t + r + K]
                p_expert, status = rollout_branch(backend, snapshot, expert_actions)
                if status != "ok":
                    raise RuntimeError(f"{anchor_id}: expert branch {status}")
                policy_tail = bundle.act(demos.observation(episode_id, t))[r : r + K]
                candidates = [("expert", expert_actions.copy()), ("policy", policy_tail.copy())]
                for base_name, base, count in (
                    ("policy_perturbed", policy_tail, cfg["policy_perturbed_per_anchor"]),
                    ("expert_perturbed", expert_actions, cfg["expert_perturbed_per_anchor"]),
                ):
                    for _ in range(count):
                        delta = low_frequency_perturbation(rng, K, action_dim, demos.gripper_dims, tuple(cfg["perturbation_amplitude"]))
                        requested = base + delta
                        clipped = np.clip(requested, -1.0, 1.0).astype(np.float32)
                        stats["clipped_candidates"] += int(np.any(clipped != requested))
                        candidates.append((base_name, clipped))

                group = store.create_group(anchor_id)
                for candidate_id, (candidate_type, action_cmd) in enumerate(candidates):
                    # the expert candidate is an independent second rollout of the reference actions
                    p_candidate, status = rollout_branch(backend, snapshot, action_cmd)
                    cost = branch_cost(p_candidate, p_expert) if status == "ok" else np.nan
                    stats["branches"] += 1
                    stats["aborted"] += int(status != "ok")
                    entry = group.create_group(f"{candidate_id:02d}")
                    entry["action_cmd"] = action_cmd.astype(np.float32)
                    entry["p_candidate"] = p_candidate
                    entry["p_expert"] = p_expert
                    entry.attrs.update({"J_m2": cost, "rollout_status": status, "candidate_type": candidate_type})
                    wanted = candidate_type == "expert" or (
                        candidate_type == "expert_perturbed" and not any(row["candidate_type"] == candidate_type for row in sample_rows)
                    )
                    if args.sample and len(anchors) == 0 and wanted:
                        sample_rows.append(
                            {
                                "anchor_id": anchor_id, "candidate_id": candidate_id, "candidate_type": candidate_type,
                                "J_m2": cost, "sqrtJ_mm": float(np.sqrt(cost) * 1e3),
                                "max_abs_action_delta": float(np.abs(action_cmd - expert_actions).max()),
                                "rollout_status": status,
                            }
                        )
                anchors.append(
                    {
                        "anchor_id": anchor_id, "env": "maniskill", "task": config["environment"]["task"],
                        "split": args.split, "episode_id": episode_id, "t": t, "r": r,
                        "H": spec.horizon, "e": spec.execute_steps, "K": K, "L": spec.history_steps,
                        "obs_ref": f"traj_{episode_id}/obs[{t}]",
                        "snapshot_ref": f"traj_{episode_id}/env_states[{t + r}]",
                        "history_ref": f"traj_{episode_id}/obs[{t - spec.history_steps}:{t}]",
                        "num_candidates": len(candidates),
                    }
                )
    backend.close()

    if args.sample:
        print(pd.DataFrame(sample_rows).to_string(index=False))
        write_json(run_dir / "branches_sample.json", sample_rows)
        ok = sample_rows[0]["J_m2"] == 0.0 and sample_rows[1]["J_m2"] > 0.0
        print("LABEL GATE (expert J = 0, perturbed J > 0):", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    pd.DataFrame(anchors).to_parquet(run_dir / f"anchors_{args.split}{suffix}.parquet", index=False)
    table = load_branches(h5_path)
    by_type = {}
    for candidate_type in sorted(set(table["candidate_type"])):
        costs = table["J_m2"][table["candidate_type"] == candidate_type]
        by_type[candidate_type] = {
            "count": int(len(costs)),
            "J_m2_mean": float(np.nanmean(costs)), "J_m2_median": float(np.nanmedian(costs)), "J_m2_max": float(np.nanmax(costs)),
            "sqrtJ_mm_median": float(np.nanmedian(np.sqrt(costs)) * 1e3),
        }
    stats.update({"anchors": len(anchors), "episodes": len(episode_ids), "by_type": by_type, "policy_ckpt": str(ckpt), "round": args.round, "seed": seed})
    write_json(run_dir / f"branch_stats_{args.split}{suffix}.json", stats)
    print(stats)
    gate = by_type["expert"]["J_m2_max"] == 0.0 and stats["aborted"] == 0
    print("EXPERT-VS-EXPERT J == 0 ON ALL ANCHORS, NO ABORTS:", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
