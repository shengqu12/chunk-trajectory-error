"""Replay audit: uninterrupted expert replay vs. snapshot-restore replay.

For every anchor ``(episode, t)`` of the requested split the expert actions are
executed once without interruption (reference), then the same ``K`` actions are
executed again after restoring the simulator to the anchor in several ways:

* ``snapshot_restore_same_env``    restore the in-memory snapshot in the same env
* ``snapshot_restore_after_reset`` reset with another seed, then restore the snapshot
* ``hdf5_state_restore``           reset, then load ``env_states[t]`` from the HDF5
* ``prefix_replay``                reset to the episode start and replay ``actions[:t]``
* ``snapshot_branch_repeatability`` the two snapshot-restore rollouts compared with
                                   EACH OTHER (not with the reference).  This is the
                                   property branch labels rely on: candidate and expert
                                   branches both start from the same restored snapshot.
                                   Gated by ``audit.branch_repeat_tolerance_m``.
* ``control_shift_one_frame``      negative control: reference compared against
                                   itself shifted by one step.  It must FAIL the
                                   tolerance on every anchor whose reference moves at
                                   least ``audit.control_min_motion_m`` in one step;
                                   slower anchors are reported as not applicable.

Outputs: ``splits.json``, ``anchors.json``, ``replay_audit.csv``,
``replay_audit_summary.json`` and one time-stamped side-by-side video.
"""

from __future__ import annotations

import csv
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.contracts import ChunkSpec, ReplayAudit
from chunk_trajectory_error.replay import position_errors, trajectory_cost_m2

CONTROL_MODE = "control_shift_one_frame"
REPEAT_MODE = "snapshot_branch_repeatability"
DEFAULT_BRANCH_REPEAT_TOLERANCE_M = 1e-6
DEFAULT_CONTROL_MIN_MOTION_M = 2e-3
GATED_MODES = ("snapshot_restore_same_env", "snapshot_restore_after_reset", "hdf5_state_restore", "prefix_replay")


# --------------------------------------------------------------------------- #
# splits and anchors
# --------------------------------------------------------------------------- #
def make_splits(episode_ids: list[int], seed: int, ratio: list[int]) -> dict[str, list[int]]:
    """Episode-level split; an episode never appears in two splits."""
    ids = sorted(episode_ids)
    random.Random(seed).shuffle(ids)
    total = sum(ratio)
    n_val = max(1, round(len(ids) * ratio[1] / total))
    n_test = max(1, round(len(ids) * ratio[2] / total))
    n_train = len(ids) - n_val - n_test
    if n_train <= 0:
        raise ValueError("not enough episodes for a train/val/test split")
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def anchor_times(num_actions: int, spec: ChunkSpec, count: int) -> list[int]:
    """Anchors need history [t-L, t) and r+K future actions: L <= t <= T-r-K."""
    first, last = spec.history_steps, num_actions - spec.delay - spec.evaluation_steps
    if last < first:
        return []
    return sorted({int(round(value)) for value in np.linspace(first, last, count)})


def flat_state(tree: Any, part: str = "all") -> np.ndarray:
    """Flatten actors/articulations.  ``part`` selects "pose" (positions, quaternions,
    joint positions), "vel" (linear/angular/joint velocities) or "all"."""
    parts: list[np.ndarray] = []
    for name in sorted(tree.get("actors", {})):
        vector = np.asarray(tree["actors"][name], dtype=np.float64)
        parts.append({"pose": vector[:7], "vel": vector[7:], "all": vector}[part])
    for name in sorted(tree.get("articulations", {})):
        vector = np.asarray(tree["articulations"][name], dtype=np.float64)
        dof = (vector.shape[0] - 13) // 2
        pose = np.concatenate([vector[:7], vector[13 : 13 + dof]])
        vel = np.concatenate([vector[7:13], vector[13 + dof :]])
        parts.append({"pose": pose, "vel": vel, "all": vector}[part])
    return np.concatenate(parts)


def cube_position(backend: Any) -> np.ndarray:
    return backend.state_dict_numpy()["actors"]["cube"][:3].astype(np.float64)


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #
def run_actions(backend: Any, actions: np.ndarray, frames: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Execute actions from the current state; return EE and cube positions after each step."""
    ee, cube = [], []
    for action in actions:
        backend.step(action)
        ee.append(backend.ee_position())
        cube.append(cube_position(backend))
        if frames is not None:
            frames.append(backend.render_rgb())
    return np.asarray(ee, dtype=np.float64), np.asarray(cube, dtype=np.float64)


def audit_anchor(
    backend: Any,
    episode: Any,
    t: int,
    spec: ChunkSpec,
    tolerance: float,
    video: dict | None,
    repeat_tolerance: float = DEFAULT_BRANCH_REPEAT_TOLERANCE_M,
    control_min_motion: float = DEFAULT_CONTROL_MIN_MOTION_M,
) -> list[dict]:
    r, K = spec.delay, spec.evaluation_steps
    start = t + r  # branch start index; state[start] precedes actions[start]
    stop = start + K
    branch_actions = episode.actions[start:stop]

    # ---- reference: uninterrupted expert replay from the episode start ---- #
    backend.reset(seed=episode.seed, init_state=episode.state_at(0))
    ref_frames: list | None = [] if video is not None else None
    ref_ee = [backend.ee_position()]
    recorded_state_diff = {"pose": 0.0, "vel": 0.0}
    snapshot = None
    video_first = max(0, t - spec.history_steps)
    for index in range(stop):
        live_tree, recorded_tree = backend.state_dict_numpy(), episode.state_at(index)
        live = flat_state(live_tree)
        for part in ("pose", "vel"):
            recorded_state_diff[part] = max(
                recorded_state_diff[part],
                float(np.abs(flat_state(live_tree, part) - flat_state(recorded_tree, part)).max()),
            )
        if index == start:
            snapshot = backend.snapshot()
            snapshot_flat = live.copy()
            ref_cube_start = cube_position(backend)
        if ref_frames is not None and index >= video_first:
            ref_frames.append(backend.render_rgb())
        backend.step(episode.actions[index])
        ref_ee.append(backend.ee_position())
        if index == start:
            ref_cube = []
        if index >= start:
            ref_cube.append(cube_position(backend))
    if ref_frames is not None:
        ref_frames.append(backend.render_rgb())
    ref_ee = np.asarray(ref_ee, dtype=np.float64)  # ref_ee[i] = EE position at state i
    ref_branch_ee = ref_ee[start + 1 : stop + 1]
    ref_cube = np.asarray(ref_cube, dtype=np.float64)
    assert snapshot is not None and len(ref_branch_ee) == K

    results: dict[str, dict[str, Any]] = {}

    def record(
        mode: str,
        ee: np.ndarray,
        cube: np.ndarray | None,
        restored_flat: np.ndarray | None,
        against_ee: np.ndarray | None = None,
        against_cube: np.ndarray | None = None,
    ) -> None:
        target_ee = ref_branch_ee if against_ee is None else against_ee
        target_cube = ref_cube if against_cube is None else against_cube
        distances = position_errors(ee.tolist(), target_ee.tolist())
        cost = trajectory_cost_m2(distances)
        results[mode] = {
            "ee": ee,
            "cube": cube,
            "distances": distances,
            "max": max(distances),
            "mean": float(np.mean(distances)),
            "J_m2": cost,
            "max_obj": None if cube is None else float(np.linalg.norm(cube - target_cube, axis=1).max()),
            "restore_state_diff": None if restored_flat is None else float(np.abs(restored_flat - snapshot_flat).max()),
        }

    # 1) restore the in-memory snapshot in the same (now advanced) environment
    backend.restore(snapshot)
    restored = flat_state(backend.state_dict_numpy())
    ee, cube = run_actions(backend, branch_actions)
    record("snapshot_restore_same_env", ee, cube, restored)

    # 2) dirty the environment with an unrelated reset, then restore the snapshot
    backend.reset(seed=(episode.seed or 0) + 10_007)
    backend.restore(snapshot)
    restored = flat_state(backend.state_dict_numpy())
    restore_frames: list | None = None
    if video is not None:
        restore_frames = [backend.render_rgb()]
    ee, cube = run_actions(backend, branch_actions, restore_frames)
    record("snapshot_restore_after_reset", ee, cube, restored)

    # 3) restore from the state stored in the demonstration file (float32 on disk)
    backend.reset(seed=(episode.seed or 0) + 20_011)
    backend.restore(episode.state_at(start))
    restored = flat_state(backend.state_dict_numpy())
    ee, cube = run_actions(backend, branch_actions)
    record("hdf5_state_restore", ee, cube, restored)

    # 4) deterministic prefix replay from the episode's first state
    backend.reset(seed=episode.seed, init_state=episode.state_at(0))
    run_actions(backend, episode.actions[:start])
    restored = flat_state(backend.state_dict_numpy())
    ee, cube = run_actions(backend, branch_actions)
    record("prefix_replay", ee, cube, restored)

    # 5) branch repeatability: two independent restores of the same snapshot, same
    #    actions, compared with each other.  Expert-vs-expert J from one snapshot.
    first, second = results["snapshot_restore_same_env"], results["snapshot_restore_after_reset"]
    record(REPEAT_MODE, second["ee"], second["cube"], None, against_ee=first["ee"], against_cube=first["cube"])

    # 6) negative control: a one-frame misalignment must be caught
    shifted = ref_ee[start : stop]  # positions one step too early
    record(CONTROL_MODE, shifted, None, None)
    ref_step_motion = float(np.linalg.norm(np.diff(ref_ee[start : stop + 1], axis=0), axis=1).max())
    control_applicable = ref_step_motion >= control_min_motion

    anchor_id = f"ep{episode.episode_id:04d}_t{t:04d}_r{r}"
    rows = []
    for mode, item in results.items():
        mode_tolerance = repeat_tolerance if mode == REPEAT_MODE else tolerance
        within = item["max"] <= mode_tolerance
        expected_within = mode != CONTROL_MODE
        if mode == CONTROL_MODE and not control_applicable:
            status = "not_applicable_slow_segment"
        else:
            status = "ok" if within == expected_within else "UNEXPECTED"
        audit = ReplayAudit(
            anchor_id=anchor_id,
            steps=K,
            max_ee_position_error_m=item["max"],
            mean_ee_position_error_m=item["mean"],
            within_tolerance=within,
            status=status,
        )
        rows.append(
            {
                "anchor_id": audit.anchor_id,
                "episode_id": episode.episode_id,
                "episode_seed": episode.seed,
                "t": t,
                "r": r,
                "K": K,
                "branch_first_action_index": start,
                "mode": mode,
                "steps": audit.steps,
                "max_ee_err_m": f"{audit.max_ee_position_error_m:.6e}",
                "mean_ee_err_m": f"{audit.mean_ee_position_error_m:.6e}",
                "max_ee_err_mm": f"{audit.max_ee_position_error_m * 1e3:.6f}",
                "mean_ee_err_mm": f"{audit.mean_ee_position_error_m * 1e3:.6f}",
                "J_m2": f"{item['J_m2']:.6e}",
                "sqrt_J_mm": f"{np.sqrt(item['J_m2']) * 1e3:.6f}",
                "max_obj_err_m": "" if item["max_obj"] is None else f"{item['max_obj']:.6e}",
                "restore_state_max_abs_diff": ""
                if item["restore_state_diff"] is None
                else f"{item['restore_state_diff']:.6e}",
                "continuous_vs_recorded_pose_max_abs_diff": f"{recorded_state_diff['pose']:.6e}",
                "continuous_vs_recorded_velocity_max_abs_diff": f"{recorded_state_diff['vel']:.6e}",
                "reference_max_step_motion_m": f"{ref_step_motion:.6e}",
                "compared_against": "snapshot_restore_same_env" if mode == REPEAT_MODE else "continuous_reference",
                "tolerance_m": mode_tolerance,
                "within_tolerance": audit.within_tolerance,
                "expected_within_tolerance": expected_within,
                "status": audit.status,
            }
        )

    if video is not None:
        write_video(
            video["path"],
            ref_frames,
            restore_frames,
            first_state=video_first,
            anchor=start,
            distances=results["snapshot_restore_after_reset"]["distances"],
            title=f"PickCube-v1 episode {episode.episode_id} (seed {episode.seed})",
            fps=video["fps"],
        )
    return rows


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def write_video(path: Path, ref_frames, restore_frames, first_state, anchor, distances, title, fps) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    try:
        font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=18)
    except TypeError:  # older Pillow
        font = small = ImageFont.load_default()
    height, width = ref_frames[0].shape[:2]
    bar = 96

    def compose(state_index: int, left: np.ndarray, right: np.ndarray | None, note: str) -> np.ndarray:
        canvas = Image.new("RGB", (width * 2, height + bar), (18, 18, 18))
        canvas.paste(Image.fromarray(left), (0, bar))
        if right is not None:
            canvas.paste(Image.fromarray(right), (width, bar))
        draw = ImageDraw.Draw(canvas)
        offset = state_index - anchor
        draw.text((10, 6), f"{title}   state index i = {state_index}   (anchor t = {anchor}, i - t = {offset:+d})", font=font, fill=(255, 255, 255))
        draw.text((10, 38), "LEFT: uninterrupted expert replay", font=small, fill=(140, 220, 140))
        draw.text((width + 10, 38), "RIGHT: reset -> restore snapshot at t -> same actions", font=small, fill=(140, 190, 255))
        draw.text((10, 66), note, font=small, fill=(255, 220, 120))
        if right is None:
            draw.text((width + 40, bar + height // 2), "waiting for anchor (not started)", font=font, fill=(150, 150, 150))
        return np.asarray(canvas)

    frames = []
    for offset, left in enumerate(ref_frames):
        state_index = first_state + offset
        branch_step = state_index - anchor
        if branch_step < 0:
            frames.append(compose(state_index, left, None, "history window before the anchor"))
        elif branch_step == 0:
            frame = compose(state_index, left, restore_frames[0], "snapshot restored; no branch action executed yet")
            frames.extend([frame] * max(1, fps))  # hold on the restore frame
        else:
            error_mm = distances[branch_step - 1] * 1e3
            note = f"after branch action j = {branch_step}:  |p_ee(restored) - p_ee(continuous)| = {error_mm:.6f} mm"
            frames.append(compose(state_index, left, restore_frames[branch_step], note))
    frames.extend([frames[-1]] * max(1, fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = ArgumentParser(description="Audit expert replay and simulator snapshot restoration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-anchors", type=int, default=None, help="audit only the first N anchors of the split")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-anchor-index", type=int, default=0, help="which audited anchor gets the video")
    args = parser.parse_args()
    config = load_config(args.config)
    chunk = config["chunk"]
    spec = ChunkSpec(
        horizon=chunk["horizon"],
        execute_steps=chunk["execute_steps"],
        history_steps=chunk["history_steps"],
        evaluation_steps=chunk["evaluation_steps"],
        delay=chunk["train_delay"],
    )
    spec.validate()
    if config["environment"]["backend"] != "maniskill":
        raise NotImplementedError(f"Expert replay is not implemented for {config['environment']['backend']}")

    from chunk_trajectory_error.backends.maniskill import (
        ManiSkillBackend,
        check_action_state_boundary,
        load_demo_episodes,
    )

    audit_cfg, data_cfg = config["audit"], config["data"]
    tolerance = float(audit_cfg["replay_position_tolerance_m"])  # fixed before any rollout
    repeat_tolerance = float(audit_cfg.get("branch_repeat_tolerance_m", DEFAULT_BRANCH_REPEAT_TOLERANCE_M))
    control_min_motion = float(audit_cfg.get("control_min_motion_m", DEFAULT_CONTROL_MIN_MOTION_M))
    out_dir = Path(config["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_demo_episodes(data_cfg["replayed_trajectory_path"])[: audit_cfg["num_episodes"]]
    for episode in episodes:
        check_action_state_boundary(episode.num_actions, episode.num_states)
    by_id = {episode.episode_id: episode for episode in episodes}

    split_path = Path(data_cfg["split_manifest"])
    if split_path.is_file():
        split_record = json.loads(split_path.read_text(encoding="utf-8"))
        if sorted(sum(split_record["splits"].values(), [])) != sorted(by_id):
            raise ValueError(f"{split_path} does not cover the current episodes; delete it deliberately to re-split")
        print(f"using locked split {split_path}")
    else:
        split_record = {
            "seed": config["experiment"]["seed"],
            "ratio_train_val_test": audit_cfg["split_ratio"],
            "unit": "episode",
            "source": data_cfg["replayed_trajectory_path"],
            "splits": make_splits(list(by_id), config["experiment"]["seed"], audit_cfg["split_ratio"]),
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split_record, indent=2), encoding="utf-8")
        print(f"wrote split {split_path}")
    (out_dir / "splits.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")

    anchors = []
    for split_name, ids in split_record["splits"].items():
        for episode_id in ids:
            for t in anchor_times(by_id[episode_id].num_actions, spec, audit_cfg["anchors_per_episode"]):
                anchors.append(
                    {
                        "anchor_id": f"ep{episode_id:04d}_t{t:04d}_r{spec.delay}",
                        "env": "maniskill",
                        "task": config["environment"]["task"],
                        "split": split_name,
                        "episode_id": episode_id,
                        "t": t,
                        "r": spec.delay,
                        "H": spec.horizon,
                        "e": spec.execute_steps,
                        "K": spec.evaluation_steps,
                        "L": spec.history_steps,
                    }
                )
    (out_dir / "anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    selected = [anchor for anchor in anchors if anchor["split"] == args.split]
    if args.max_anchors is not None:
        selected = selected[: args.max_anchors]
    if not selected:
        raise ValueError(f"no anchors available in split {args.split}")

    want_video = bool(audit_cfg.get("render_video")) and not args.no_video
    backend = ManiSkillBackend(config, render=want_video)
    rows: list[dict] = []
    video_index = min(max(args.video_anchor_index, 0), len(selected) - 1)
    video_path = out_dir / f"replay_audit_{selected[video_index]['anchor_id']}.mp4"
    for index, anchor in enumerate(selected):
        video = {"path": video_path, "fps": audit_cfg.get("video_fps", 10)} if want_video and index == video_index else None
        anchor_rows = audit_anchor(
            backend, by_id[anchor["episode_id"]], anchor["t"], spec, tolerance, video, repeat_tolerance, control_min_motion
        )
        for row in anchor_rows:
            row["split"] = anchor["split"]
        rows.extend(anchor_rows)
        gated = [row for row in anchor_rows if row["mode"] in (*GATED_MODES, REPEAT_MODE)]
        print(
            f"{anchor['anchor_id']}: "
            + "  ".join(f"{row['mode']}={row['max_ee_err_mm']}mm" for row in gated)
        )
    backend.close()

    csv_path = out_dir / "replay_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "split": args.split,
        "anchors_audited": len(selected),
        "tolerance_m": tolerance,
        "branch_repeat_tolerance_m": repeat_tolerance,
        "control_min_motion_m": control_min_motion,
        "chunk": chunk,
        "modes": {},
        "video": str(video_path) if want_video else None,
    }
    passed = True
    for mode in (*GATED_MODES, REPEAT_MODE, CONTROL_MODE):
        mode_rows = [row for row in rows if row["mode"] == mode]
        max_err = max(float(row["max_ee_err_m"]) for row in mode_rows)
        mean_err = float(np.mean([float(row["mean_ee_err_m"]) for row in mode_rows]))
        n_within = sum(row["within_tolerance"] for row in mode_rows)
        entry: dict[str, Any] = {
            "anchors": len(mode_rows),
            "max_ee_err_m": max_err,
            "mean_ee_err_m": mean_err,
            "max_J_m2": max(float(row["J_m2"]) for row in mode_rows),
            "anchors_within_tolerance": int(n_within),
            "tolerance_m": repeat_tolerance if mode == REPEAT_MODE else tolerance,
        }
        if mode == CONTROL_MODE:
            applicable = [row for row in mode_rows if row["status"] != "not_applicable_slow_segment"]
            rejected = sum(not row["within_tolerance"] for row in applicable)
            ok = len(applicable) > 0 and rejected == len(applicable)
            entry.update(
                {
                    "applicable_anchors": len(applicable),
                    "applicable_anchors_rejected": int(rejected),
                    "not_applicable_slow_segment": len(mode_rows) - len(applicable),
                    "min_ee_err_m_on_applicable": min((float(row["max_ee_err_m"]) for row in applicable), default=None),
                    "expectation": "every applicable anchor must fail tolerance (negative control); "
                    "applicable = reference moves >= control_min_motion_m in one step",
                }
            )
        else:
            ok = n_within == len(mode_rows)
            entry["expectation"] = (
                "two restores of one snapshot must agree within branch_repeat_tolerance_m"
                if mode == REPEAT_MODE
                else "must be within tolerance of the uninterrupted reference"
            )
        passed &= ok
        entry["gate"] = "pass" if ok else "FAIL"
        summary["modes"][mode] = entry
    for part in ("pose", "velocity"):
        key = f"continuous_vs_recorded_{part}_max_abs_diff"
        summary[key] = max(float(row[key]) for row in rows)
    summary["all_gates_passed"] = bool(passed)
    (out_dir / "replay_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"csv -> {csv_path}")
    print("REPLAY AUDIT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
