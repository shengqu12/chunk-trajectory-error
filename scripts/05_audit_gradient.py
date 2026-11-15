"""Gradient audit for the frozen error network.

(a) Parameter-gradient check: back-propagating the surrogate alone gives finite,
    non-zero gradients on policy parameters and none on error-network parameters.
(b) Direction check in simulation: on held-out candidates, a small step along
    ``-dE/dA`` (gripper masked) is compared with an equal-norm random step by
    re-simulating both from the same restored state.  The step size is chosen on the
    validation split only and then applied once to the test split.

Outputs: ``gradient_audit.csv`` and ``gradient_audit_summary.json``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.objectives import surrogate_regularized_policy_loss
from chunk_trajectory_error.pipeline import (
    ErrorBundle, PolicyBundle, anchor_snapshot, bootstrap_ci, branch_cost, build_policy_dataset, chunk_spec,
    load_branches, load_demoset, out_dir, rollout_branch, select_state, write_json,
)

AUDITED_TYPES = ("policy", "policy_perturbed", "expert_perturbed")


def parameter_gradient_check(policy: PolicyBundle, error: ErrorBundle, demos, spec, state_source: str) -> dict:
    data = build_policy_dataset(demos, "train", spec)
    index = torch.nonzero(data["error_valid"]).flatten()[:256]
    state = data["obs"][index] if state_source == "full_obs" else data["robot"][index]
    policy.zero_grad()
    chunk = policy(data["obs"][index])
    tail = chunk[:, spec.delay : spec.delay + spec.evaluation_steps].clamp(-1.0, 1.0)
    surrogate_regularized_policy_loss(chunk.sum() * 0.0, error, data["history"][index], state, tail, spec.delay, 1.0)  # freezes the critic
    error.normalized_cost(data["history"][index], state, tail, spec.delay).mean().backward()
    policy_grads = [p.grad for p in policy.policy.parameters()]
    norm = float(torch.sqrt(sum((g**2).sum() for g in policy_grads if g is not None)))
    result = {
        "policy_grad_norm": norm,
        "policy_grads_finite": bool(all(g is not None and torch.isfinite(g).all() for g in policy_grads)),
        "policy_grads_nonzero": bool(norm > 0),
        "error_params_without_grad": bool(all(p.grad is None for p in error.parameters())),
        "error_params_require_grad": bool(any(p.requires_grad for p in error.parameters())),
    }
    result["pass"] = (
        result["policy_grads_finite"] and result["policy_grads_nonzero"]
        and result["error_params_without_grad"] and not result["error_params_require_grad"]
    )
    policy.zero_grad()
    return result


def main() -> int:
    parser = ArgumentParser(description="Audit parameter gradients and the simulated effect of the surrogate action gradient")
    parser.add_argument("--config", required=True)
    parser.add_argument("--error-tag", default="history")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--val-episodes", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config)
    spec = chunk_spec(config)
    run_dir = out_dir(config)
    demos = load_demoset(config)
    policy = PolicyBundle.load(run_dir / "checkpoints" / "pi0.pt")
    error = ErrorBundle.load(run_dir / "checkpoints" / f"error_round{args.round}_{args.error_tag}.pt")
    state_source = error.meta["state_source"]
    r, K = spec.delay, spec.evaluation_steps
    continuous = [d for d in range(policy.meta["action_dim"]) if d not in demos.gripper_dims]
    rng = np.random.default_rng(config["experiment"]["seed"] + 5)

    param_check = parameter_gradient_check(policy, error, demos, spec, state_source)
    print("parameter gradient check:", param_check)

    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    backend = ManiSkillBackend(config)
    backend.reset(seed=config["experiment"]["seed"])  # the env must be reset once before any restore
    skipped = {"aborted_rollouts": 0, "zero_gradient": 0}

    def audit(split: str, step_sizes: list[float], episode_limit: int | None) -> list[dict]:
        table = load_branches(run_dir / f"branches_{split}.h5")
        allowed = set(demos.ids(split)[:episode_limit] if episode_limit else demos.ids(split))
        rows, seen = [], set()
        for i in range(len(table["J_m2"])):
            candidate_type, episode_id, t = str(table["candidate_type"][i]), int(table["episode_id"][i]), int(table["t"][i])
            key = (table["anchor_id"][i], candidate_type)
            if candidate_type not in AUDITED_TYPES or episode_id not in allowed or key in seen:
                continue  # one candidate of each audited type per anchor
            seen.add(key)
            episode = demos.episodes[episode_id]
            action = torch.tensor(table["action_cmd"][i])[None].requires_grad_(True)
            history = torch.tensor(demos.history(episode_id, t, spec.history_steps))[None]
            state = torch.tensor(select_state(demos, episode_id, t, state_source))[None]
            predicted = error.cost_m2(history, state, action, r)
            (gradient,) = torch.autograd.grad(predicted.sum(), action)
            gradient = gradient[0].numpy()
            direction = np.zeros_like(gradient)
            direction[:, continuous] = gradient[:, continuous]
            grad_norm = float(np.linalg.norm(direction))
            noise = np.zeros_like(gradient)
            noise[:, continuous] = rng.standard_normal((K, len(continuous)))
            noise /= np.linalg.norm(noise)
            if grad_norm == 0.0:
                skipped["zero_gradient"] += 1
                continue
            direction /= grad_norm
            base = table["action_cmd"][i]
            snapshot = anchor_snapshot(episode, t, r)
            expert, _ = rollout_branch(backend, snapshot, episode.actions[t + r : t + r + K])
            j0 = float(table["J_m2"][i])
            for eta in step_sizes:
                a_grad = np.clip(base - eta * direction, -1.0, 1.0).astype(np.float32)
                a_rand = np.clip(base + eta * noise, -1.0, 1.0).astype(np.float32)
                p_grad, s1 = rollout_branch(backend, snapshot, a_grad)
                p_rand, s2 = rollout_branch(backend, snapshot, a_rand)
                if s1 != "ok" or s2 != "ok":
                    skipped["aborted_rollouts"] += 1
                    continue
                rows.append(
                    {
                        "split": split, "anchor_id": table["anchor_id"][i], "episode_id": episode_id, "t": t,
                        "candidate_id": int(table["candidate_id"][i]), "candidate_type": candidate_type, "eta": eta,
                        "E_pred_m2": float(predicted.detach()),
                        "J0_m2": j0, "J_grad_m2": branch_cost(p_grad, expert), "J_rand_m2": branch_cost(p_rand, expert),
                        "effective_norm_grad": float(np.linalg.norm(a_grad - base)),
                        "effective_norm_rand": float(np.linalg.norm(a_rand - base)),
                        "surrogate_grad_norm": grad_norm,
                    }
                )
        return rows

    def summarise(frame: pd.DataFrame) -> dict:
        d_grad = (frame["J_grad_m2"] - frame["J0_m2"]).to_numpy()
        d_rand = (frame["J_rand_m2"] - frame["J0_m2"]).to_numpy()
        groups = frame["episode_id"].to_numpy()
        mean_g, lo_g, hi_g = bootstrap_ci(d_grad, groups)
        mean_r, lo_r, hi_r = bootstrap_ci(d_rand, groups)
        mean_p, lo_p, hi_p = bootstrap_ci(d_grad - d_rand, groups)
        return {
            "candidates": int(len(frame)), "episodes": int(frame["episode_id"].nunique()),
            "mean_J0_m2": float(frame["J0_m2"].mean()),
            "delta_J_grad_m2": {"mean": mean_g, "ci95": [lo_g, hi_g]},
            "delta_J_rand_m2": {"mean": mean_r, "ci95": [lo_r, hi_r]},
            "paired_grad_minus_rand_m2": {"mean": mean_p, "ci95": [lo_p, hi_p]},
            "relative_change_grad": float(d_grad.sum() / frame["J0_m2"].sum()),
            "relative_change_rand": float(d_rand.sum() / frame["J0_m2"].sum()),
            "fraction_grad_lowers_J": float((d_grad < 0).mean()),
            "fraction_rand_lowers_J": float((d_rand < 0).mean()),
            "fraction_grad_beats_rand": float((d_grad < d_rand).mean()),
            "mean_effective_norm_grad": float(frame["effective_norm_grad"].mean()),
            "mean_effective_norm_rand": float(frame["effective_norm_rand"].mean()),
        }

    step_sizes = [float(x) for x in config["gradient_audit"]["step_sizes"]]
    val = pd.DataFrame(audit("val", step_sizes, args.val_episodes))
    # The step size is chosen on validation separately for every candidate type, so that
    # large artificial perturbations cannot hide what happens on the policy's own samples.
    val_summary, chosen = {}, {}
    for candidate_type in AUDITED_TYPES:
        subset = val[val["candidate_type"] == candidate_type]
        val_summary[candidate_type] = {str(eta): summarise(subset[subset["eta"] == eta]) for eta in step_sizes}
        chosen[candidate_type] = min(step_sizes, key=lambda eta: val_summary[candidate_type][str(eta)]["delta_J_grad_m2"]["mean"])
    print("step sizes chosen on val:", chosen)
    test = pd.DataFrame(audit("test", sorted(set(chosen.values())), None))
    backend.close()

    pd.concat([val, test]).to_csv(run_dir / "gradient_audit.csv", index=False)
    test_summary, gates = {}, {}
    for candidate_type in AUDITED_TYPES:
        subset = test[(test["candidate_type"] == candidate_type) & (test["eta"] == chosen[candidate_type])]
        test_summary[candidate_type] = summarise(subset)
        test_summary[candidate_type]["step_size"] = chosen[candidate_type]
        row = test_summary[candidate_type]
        gates[candidate_type] = bool(row["delta_J_grad_m2"]["ci95"][1] < 0 and row["paired_grad_minus_rand_m2"]["ci95"][1] < 0)
    direction_ok = gates["policy"]
    summary = {
        "error_checkpoint": f"error_round{args.round}_{args.error_tag}.pt",
        "parameter_gradient_check": param_check,
        "skipped": skipped,
        "step_sizes_tried_on_val": step_sizes,
        "chosen_step_size": chosen,
        "val": val_summary,
        "test": test_summary,
        "direction_gate": {
            "rule": "on test, at the step size chosen on val for that candidate type: 95% CI of delta_J_grad below 0 "
                    "AND 95% CI of (grad - rand) below 0; the policy-sample row decides the gate",
            "by_candidate_type": gates,
            "pass": direction_ok,
        },
    }
    write_json(run_dir / f"gradient_audit_summary{'' if args.error_tag == 'history' else '_' + args.error_tag}.json", summary)
    for candidate_type in AUDITED_TYPES:
        print("test", candidate_type, test_summary[candidate_type])
    print("GRADIENT DIRECTION GATE (policy samples):", "PASS" if direction_ok else "FAIL")
    return 0 if param_check["pass"] and skipped["aborted_rollouts"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
