"""Selection audit: can the frozen critic tell which of the *policy's own* sampled chunks
stays closest to the expert?

At every validation anchor the flow policy samples N chunks; the critic scores each tail with
the demonstration history and robot state, and the simulator recomputes the true J of each
tail from the same restored state.  Per anchor this gives the within-anchor rank correlation
between predicted and true cost, the true J of the chunk the critic picks, of a random pick,
of the chunk it would reject, and of the oracle pick.

Outputs: ``selection_audit.json`` (summary) and ``selection_audit.parquet`` (one row per chunk).
"""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, anchor_snapshot, anchor_times, bootstrap_ci, branch_cost, chunk_spec, load_demoset, out_dir,
    rollout_branch, seed_everything, write_json,
)


def main() -> int:
    parser = ArgumentParser(description="Audit critic-based selection among the policy's own samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", default="checkpoints/pi_bc_flow/pi.pt")
    parser.add_argument("--critic", default="checkpoints/error_round0_history.pt")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--delay", type=int, default=None, help="delay r used for the tail and the critic (default: training delay)")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    spec = chunk_spec(config, args.delay) if args.delay is not None else chunk_spec(config)
    run_dir = out_dir(config)
    demos = load_demoset(config)
    policy = PolicyBundle.load(run_dir / args.policy)
    error = ErrorBundle.load(run_dir / args.critic)
    if not policy.is_flow:
        raise SystemExit("the selection audit needs a stochastic (flow) policy")
    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config)
    r, K, N = spec.delay, spec.evaluation_steps, args.samples
    anchors_per_episode = config["branches"]["anchors_per_episode"][args.split]
    ids = demos.ids(args.split)[: args.episodes] if args.episodes else demos.ids(args.split)
    rows = []
    policy.reseed()
    for episode_id in ids:
        episode = demos.episodes[episode_id]
        backend.reset(seed=episode.seed, init_state=episode.state_at(0))
        for t in anchor_times(episode.num_actions, spec, anchors_per_episode):
            snapshot = anchor_snapshot(episode, t, r)
            expert, _ = rollout_branch(backend, snapshot, episode.actions[t + r : t + r + K])
            observation = demos.observation(episode_id, t)
            with torch.no_grad():
                obs = torch.as_tensor(observation, dtype=torch.float32)[None].expand(N, -1)
                chunks = policy(obs).clamp(-1.0, 1.0)
                hist = torch.as_tensor(demos.history(episode_id, t, spec.history_steps))[None].expand(N, -1, -1)
                state = observation if error.meta.get("state_source") == "full_obs" else demos.robot_state(episode_id, t)
                state = torch.as_tensor(state, dtype=torch.float32)[None].expand(N, -1)
                predicted = error.normalized_cost(hist, state, chunks[:, r : r + K], r).numpy()
            for i in range(N):
                candidate, status = rollout_branch(backend, snapshot, chunks[i, r : r + K].numpy())
                rows.append({
                    "episode_id": episode_id, "t": t, "sample": i, "predicted": float(predicted[i]),
                    "J_m2": branch_cost(candidate, expert) if status == "ok" else np.nan, "status": status,
                })
        print(f"episode {episode_id}: {len(rows)} chunks so far", flush=True)
    backend.close()

    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "selection_audit.parquet", index=False)
    ok = frame.dropna(subset=["J_m2"])
    per_anchor = []
    for (episode_id, t), group in ok.groupby(["episode_id", "t"]):
        if len(group) < 3:
            continue
        rho = spearmanr(group["predicted"], group["J_m2"]).correlation
        j = group["J_m2"].to_numpy()
        pick = int(group["predicted"].to_numpy().argmin())
        reject = int(group["predicted"].to_numpy().argmax())
        per_anchor.append({
            "episode_id": episode_id, "t": t, "spearman": float(rho) if np.isfinite(rho) else np.nan,
            "J_picked": float(j[pick]), "J_rejected": float(j[reject]), "J_random": float(j.mean()),
            "J_oracle": float(j.min()), "J_worst": float(j.max()),
            "picked_rank": int((j < j[pick]).sum()) + 1,  # 1 = oracle
            "spread_true_sqrtJ_mm": float((np.sqrt(j.max()) - np.sqrt(j.min())) * 1e3),
            "spread_predicted": float(group["predicted"].max() - group["predicted"].min()),
        })
    pa = pd.DataFrame(per_anchor)
    groups = pa["episode_id"].to_numpy()

    def ci(values):
        mean, lo, hi = bootstrap_ci(np.asarray(values, dtype=np.float64), groups)
        return {"mean": mean, "ci95": [lo, hi]}

    summary = {
        "policy": args.policy, "critic": args.critic, "split": args.split, "delay": r, "samples_per_anchor": N,
        "anchors": int(len(pa)), "chunks": int(len(ok)),
        "within_anchor_spearman": ci(pa["spearman"].fillna(0.0)),
        "mean_J_m2": {key: ci(pa[f"J_{key}"]) for key in ("picked", "random", "rejected", "oracle", "worst")},
        "picked_minus_random_J_m2": ci(pa["J_picked"] - pa["J_random"]),
        "picked_minus_oracle_J_m2": ci(pa["J_picked"] - pa["J_oracle"]),
        "rejected_minus_random_J_m2": ci(pa["J_rejected"] - pa["J_random"]),
        "picked_is_oracle": float((pa["picked_rank"] == 1).mean()),
        "picked_in_top_half": float((pa["picked_rank"] <= N // 2).mean()),
        "mean_picked_rank": float(pa["picked_rank"].mean()),
        "spread_true_sqrtJ_mm": ci(pa["spread_true_sqrtJ_mm"]),
        "spread_predicted_normalized": ci(pa["spread_predicted"]),
        "pooled_pearson_predicted_vs_true": float(np.corrcoef(ok["predicted"], ok["J_m2"])[0, 1]),
    }
    write_json(run_dir / "selection_audit.json", summary)
    print({k: v for k, v in summary.items() if k in ("within_anchor_spearman", "picked_minus_random_J_m2", "picked_is_oracle", "mean_picked_rank")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
