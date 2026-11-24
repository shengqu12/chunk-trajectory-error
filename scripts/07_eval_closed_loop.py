"""Closed-loop evaluation with the shared chunk scheduler.

Every policy is run from the same initial states (held-out test episodes plus fresh
seeds never used by any demonstration) at every injected delay ``r``.  The evaluation
loads policy checkpoints only: no error network, no expert history.

Outputs (per ``--tag``): ``eval_episodes_<tag>.parquet``, ``eval_boundaries_<tag>.parquet``,
``eval_true_J_<tag>.parquet`` (test-anchor J of each policy's own chunk),
``eval_latency_<tag>.json``; with ``--video``, one paired video.
"""

from __future__ import annotations

import time
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.selection import ChunkSelector
from chunk_trajectory_error.pipeline import (
    ErrorBundle,
    PolicyBundle, chunk_spec, closed_loop_episode, load_demoset, out_dir, policy_true_cost, write_json,
)

DEFAULT_POLICIES = {
    "pi0": "checkpoints/pi0.pt",
    "bc": "checkpoints/pi_bc/pi.pt",
    "error": "checkpoints/pi_error/pi.pt",
    "nohist": "checkpoints/pi_nohist/pi.pt",
    "smooth": "checkpoints/pi_smooth/pi.pt",
    "error_fullobs": "checkpoints/pi_error_fullobs/pi.pt",
    "error_w0.03": "checkpoints/pi_error/w0.03.pt",  # fixed, non-selected weight: shows the term when it is active
    "error_r1": "checkpoints/pi_error_r1/pi.pt",  # after one critic refresh round
    "error_sqrt": "checkpoints/pi_error_sqrt/pi.pt",  # exploratory sqrt(J) target
    # flow-matching policy class (02/06 --policy flow); sampled with the bundle's seeded noise
    "pi0_flow": "checkpoints/pi0_flow.pt",
    "bc_flow": "checkpoints/pi_bc_flow/pi.pt",
    "error_flow": "checkpoints/pi_error_flow/pi.pt",
    "smooth_flow": "checkpoints/pi_smooth_flow/pi.pt",
    "error_flow_w0.3": "checkpoints/pi_error_flow/w0.3.pt",  # fixed, non-selected weight: the critic term when it dominates
}
# Test-time selection: ``select_<best|worst|first><N>_flow`` samples N chunks from bc_flow at every
# decision and executes the one the frozen critic scores lowest (best) / highest (worst) / the first.
SELECT_PATTERN = re.compile(r"^select_(best|worst|first)(\d+)_flow$")
SELECT_BASE, SELECT_CRITIC = "checkpoints/pi_bc_flow/pi.pt", "checkpoints/error_round0_history.pt"


def build_bundle(name: str, run_dir, config, demos):
    match = SELECT_PATTERN.match(name)
    if not match:
        return PolicyBundle.load(run_dir / DEFAULT_POLICIES[name])
    policy = PolicyBundle.load(run_dir / SELECT_BASE)
    error = ErrorBundle.load(run_dir / SELECT_CRITIC)
    spec = chunk_spec(config)
    return ChunkSelector(policy, error, int(match.group(2)), spec.delay, spec.history_steps, demos.tcp_slice, demos.robot_slice, mode=match.group(1))


def latency(bundle: PolicyBundle, observation: np.ndarray, repeats: int) -> dict:
    torch.set_num_threads(1)
    for _ in range(200):  # warm-up
        bundle.act(observation)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        bundle.act(observation)
        samples.append((time.perf_counter() - start) * 1e3)
    return {"device": "cpu (1 thread)", "batch": 1, "repeats": repeats, "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95))}


def write_pair_video(path: Path, left: list, right: list, labels: tuple[str, str], title: str, fps: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    try:
        font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=18)
    except TypeError:
        font = small = ImageFont.load_default()
    height, width = left[0].shape[:2]
    bar, frames = 72, []
    for index in range(max(len(left), len(right))):
        canvas = Image.new("RGB", (width * 2, height + bar), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        for column, (sequence, label) in enumerate(zip((left, right), labels)):
            done = index >= len(sequence) - 1
            canvas.paste(Image.fromarray(sequence[min(index, len(sequence) - 1)]), (column * width, bar))
            draw.text((column * width + 10, 40), f"{label}   step {min(index, len(sequence) - 1)}" + ("   (episode ended)" if done else ""), font=small, fill=(140, 220, 140) if column == 0 else (140, 190, 255))
        draw.text((10, 6), title, font=font, fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def main() -> int:
    parser = ArgumentParser(description="Closed-loop success, handover jump and latency under identical scheduling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policies", default=None, help="comma-separated names from the default set (default: all that exist)")
    parser.add_argument("--delays", default=None, help="comma-separated injected delays (default: evaluation.delays)")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--max-inits", type=int, default=None)
    parser.add_argument("--video", default=None, help="NAME_A,NAME_B,DELAY: render one paired episode and exit")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = out_dir(config)
    eval_cfg = config["evaluation"]
    demos = load_demoset(config)
    names = args.policies.split(",") if args.policies else [n for n, p in DEFAULT_POLICIES.items() if (run_dir / p).is_file()]
    bundles = {name: build_bundle(name, run_dir, config, demos) for name in names}
    delays = [int(x) for x in args.delays.split(",")] if args.delays else list(eval_cfg["delays"])
    max_steps = config["environment"]["max_episode_steps"]

    inits = [("test_episode", demos.episodes[i].seed, demos.episodes[i].state_at(0)) for i in demos.ids("test")]
    fresh = eval_cfg["fresh_seeds"]
    inits += [("fresh_seed", fresh["start"] + i, None) for i in range(fresh["count"])]
    if args.max_inits:
        inits = inits[: args.max_inits]

    from chunk_trajectory_error.backends.maniskill import ManiSkillBackend

    if args.video:
        name_a, name_b, delay = args.video.split(",")
        backend = ManiSkillBackend(config, render=True)
        source, seed, init_state = inits[0]
        sequences = []
        for name in (name_a, name_b):
            frames: list = []
            video_bundle = build_bundle(name, run_dir, config, demos)
            if isinstance(video_bundle, ChunkSelector):
                video_bundle.delay = int(delay)
            record, _ = closed_loop_episode(backend, video_bundle, chunk_spec(config), int(delay), seed, init_state, max_steps, demos.gripper_dims, frames)
            sequences.append((frames, f"{name}: {'success' if record['success'] else 'no success'} in {record['steps']} steps"))
        backend.close()
        path = run_dir / f"eval_pair_{name_a}_vs_{name_b}_r{delay}_seed{seed}.mp4"
        write_pair_video(path, sequences[0][0], sequences[1][0], (sequences[0][1], sequences[1][1]), f"PickCube-v1 closed loop, seed {seed}, injected delay r = {delay}", config["audit"].get("video_fps", 10))
        print("video ->", path)
        return 0

    backend = ManiSkillBackend(config)
    episodes, boundaries = [], []
    for name, bundle in bundles.items():
        for delay in delays:
            spec = chunk_spec(config, delay)
            if isinstance(bundle, ChunkSelector):
                bundle.delay = delay  # score the tail that will actually be executed
            if bundle.is_flow:
                bundle.reseed(config["experiment"]["seed"] + delay)
            wins = 0
            for init_index, (source, seed, init_state) in enumerate(inits):
                record, edges = closed_loop_episode(backend, bundle, spec, delay, seed, init_state, max_steps, demos.gripper_dims)
                record.update({"policy": name, "init_index": init_index, "init_source": source})
                episodes.append(record)
                wins += record["success"]
                for edge in edges:
                    edge.update({"policy": name, "delay": delay, "init_index": init_index, "seed": seed, "success": record["success"]})
                    boundaries.append(edge)
            print(f"{name} r={delay}: {wins}/{len(inits)} success", flush=True)
    pd.DataFrame(episodes).to_parquet(run_dir / f"eval_episodes_{args.tag}.parquet", index=False)
    selection = {name: bundle.summary() for name, bundle in bundles.items() if isinstance(bundle, ChunkSelector)}
    if selection:
        write_json(run_dir / f"eval_selection_{args.tag}.json", selection)
    pd.DataFrame(boundaries).to_parquet(run_dir / f"eval_boundaries_{args.tag}.parquet", index=False)

    spec = chunk_spec(config)
    rows = []
    for name, bundle in bundles.items():
        if isinstance(bundle, ChunkSelector):
            bundle.delay = spec.delay
        if bundle.is_flow:
            bundle.reseed()
        for row in policy_true_cost(backend, demos, bundle, "test", spec, config["branches"]["anchors_per_episode"]["test"]):
            row["policy"] = name
            rows.append(row)
    backend.close()
    pd.DataFrame(rows).to_parquet(run_dir / f"eval_true_J_{args.tag}.parquet", index=False)

    first_test = demos.ids("test")[0]
    observation = demos.observation(first_test, 0)
    for bundle in bundles.values():  # selectors time a *selected* decision: give them a full history first
        if isinstance(bundle, ChunkSelector):
            bundle.reset_episode()
            for t in range(bundle.history_steps + 1):
                bundle.observe(demos.observation(first_test, t))
    write_json(
        run_dir / f"eval_latency_{args.tag}.json",
        {name: latency(bundle, observation, eval_cfg["latency_repeats"]) for name, bundle in bundles.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
