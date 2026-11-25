"""Run the full PickCube pipeline on Modal, one container per seed.

The experiments are CPU-bound (ManiSkill physx_cpu, one environment, small networks), so
this uses CPU containers and parallelism across seeds rather than GPUs.  Two volumes persist
across calls: ``cte-data`` (official demonstrations, replayed once) and ``cte-runs`` (one run
directory per seed).

    pip install modal && modal setup                     # once, on your machine
    modal run modal_run.py --prepare                     # download + replay the official demos (once)
    modal run modal_run.py --seeds 17 --smoke --wait     # 20-minute check that the image and the sim work
    modal run --detach modal_run.py --seeds 17,18,19,20,21   # full pipeline, five seeds in parallel (~4 h), survives disconnects
    modal run --detach modal_run.py --videos 17          # side-by-side closed-loop videos for a finished seed
    modal volume get cte-runs maniskill_pickcube_seed17/results.md .   # fetch outputs

Seed 17 reproduces the container run's configuration; the other seeds change the policy,
critic and branch sampling but keep the episode split (``splits.json`` is created once in the
data volume and locked).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "chunk-trajectory-error"
REPO = Path(__file__).resolve().parent
REMOTE_REPO = "/root/proj"
DATA, RUNS = "/vol/data", "/vol/runs"

# Lock file without the CUDA wheels (torch comes from the CPU index instead, a 3 GB smaller
# image), without mplib (only needed to generate demonstrations; it is also what forced
# numpy < 2) and with opencv-python pinned to a release that accepts numpy 1.26 (5.0 requires
# numpy >= 2 and pip refuses the pair; cv2 is unused by the pipeline).
# This module is imported both on the laptop (to define the image) and inside the container
# (to run the functions), where the repository is mounted at REMOTE_REPO, so look in both places.
_lock_path = next(
    p for p in (REPO / "envs/maniskill/requirements.lock.linux-py310.txt", Path(REMOTE_REPO, "envs/maniskill/requirements.lock.linux-py310.txt"))
    if p.is_file()
)
_lock = _lock_path.read_text().splitlines()
_drop = ("torch", "nvidia-", "triton", "mplib", "cuda-", "opencv-python")
_cpu_lock = [
    line for line in _lock
    if line and not line.startswith("#") and not line.split("==")[0].lower().startswith(_drop)
] + ["opencv-python==4.11.0.86"]

image = (
    # Ubuntu 24.04 rather than Modal's Debian 12 slim image: SAPIEN initialises Vulkan even with
    # state observations and no rendering, and Debian 12's Mesa 22.3 lavapipe is rejected by its
    # loader ("ErrorIncompatibleDriver"); Ubuntu 24.04 ships the Mesa the tested container uses.
    # Python 3.11: Modal's 2026.10 builder has no 3.10; the lock resolves unchanged.
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .apt_install("libvulkan1", "mesa-vulkan-drivers", "vulkan-tools", "libgl1", "libglib2.0-0", "libxrender1", "libxext6", "libsm6", "git")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*_cpu_lock)
    .env({
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "PYTHONPATH": f"{REMOTE_REPO}/src",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        REPO, REMOTE_REPO,
        ignore=[".git", ".venv*", "runs", "data", "report", "docs", "backup_*", "**/__pycache__", "*.pyc", "*.zip"],
    )
)
app = modal.App(APP)
data_volume = modal.Volume.from_name("cte-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("cte-runs", create_if_missing=True)
VOLUMES = {DATA: data_volume, RUNS: runs_volume}


def sh(cmd: list[str], log: Path | None = None) -> None:
    """Run one stage, streaming its output (warnings filtered) and keeping a tail so that a
    failure shows the real traceback instead of the filtered stream."""
    from collections import deque

    print("+", " ".join(cmd), flush=True)
    start = time.time()
    tail: deque[str] = deque(maxlen=80)
    with (log.open("a") if log else open("/dev/null", "w")) as handle:
        proc = subprocess.Popen(cmd, cwd=REMOTE_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if "warning" not in line.lower():
                print(line, end="", flush=True)
            handle.write(line)
            tail.append(line)
        code = proc.wait()
    print(f"  -> exit {code} after {(time.time() - start) / 60:.1f} min", flush=True)
    if code != 0:
        print("----- last lines of the failed stage (unfiltered) -----", flush=True)
        print("".join(tail), flush=True)
        raise RuntimeError(f"stage failed: {' '.join(cmd)}")


@app.function(image=image, cpu=2, memory=8192, timeout=3 * 3600, volumes=VOLUMES)
def prepare_data() -> str:
    """Download the official PickCube-v1 demonstrations and replay them into the state /
    pd_ee_delta_pos format the config expects.  Runs once; the result lives in ``cte-data``."""
    import shutil

    raw, processed = Path(DATA, "raw/maniskill"), Path(DATA, "processed/maniskill")
    if (processed / "trajectory.state.pd_ee_delta_pos.physx_cpu.h5").is_file():
        return "already prepared"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    sh(["sh", "-c", "ls -la /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>&1; cat /usr/share/vulkan/icd.d/lvp_icd.json 2>&1; "
        "vulkaninfo --summary 2>&1 | head -40; VK_LOADER_DEBUG=error,warn vulkaninfo --summary > /dev/null"])
    sh([sys.executable, "-c", "import gymnasium as gym, mani_skill.envs; e = gym.make('PickCube-v1', obs_mode='state', "
        "control_mode='pd_ee_delta_pos', sim_backend='physx_cpu', num_envs=1); e.reset(seed=0); e.step(e.action_space.sample()); "
        "print('simulator ok'); e.close()"])
    if not (raw / "trajectory.h5").is_file():
        sh([sys.executable, "-m", "mani_skill.utils.download_demo", "PickCube-v1", "-o", f"{DATA}/official"])
        source = next(Path(DATA, "official").rglob("motionplanning/trajectory.h5"))
        shutil.copy(source, raw / "trajectory.h5")
        shutil.copy(source.with_suffix(".json"), raw / "trajectory.json")
        data_volume.commit()
    sh([
        sys.executable, "-m", "mani_skill.trajectory.replay_trajectory", "--traj-path", str(raw / "trajectory.h5"),
        "--use-first-env-state", "-c", "pd_ee_delta_pos", "-o", "state", "--save-traj", "--num-envs", "1", "-b", "physx_cpu",
    ])
    for suffix in (".h5", ".json"):
        shutil.move(raw / f"trajectory.state.pd_ee_delta_pos.physx_cpu{suffix}", processed / f"trajectory.state.pd_ee_delta_pos.physx_cpu{suffix}")
    data_volume.commit()
    return "prepared official PickCube-v1 demonstrations"


def seed_config(seed: int, smoke: bool) -> Path:
    import yaml

    config = yaml.safe_load(Path(REMOTE_REPO, "configs/maniskill_pickcube.yaml").read_text())
    config["experiment"]["seed"] = seed
    config["experiment"]["output_dir"] = f"{RUNS}/maniskill_pickcube_seed{seed}" + ("_smoke" if smoke else "")
    config["data"]["trajectory_path"] = f"{DATA}/raw/maniskill/trajectory.h5"
    config["data"]["replayed_trajectory_path"] = f"{DATA}/processed/maniskill/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    config["data"]["split_manifest"] = f"{DATA}/processed/maniskill/splits.json"
    config["data"]["output_dir"] = f"{DATA}/processed/maniskill"
    if smoke:
        config["bc"]["steps"], config["bc"]["eval_every"] = 300, 100
        config["bc"].setdefault("flow", {})["steps"] = 300
        config["policy_finetune"]["steps"] = 50
        config["branches"]["anchors_per_episode"] = {"train": 1, "val": 1, "test": 1}
        config["evaluation"]["fresh_seeds"]["count"] = 1
        config["evaluation"]["latency_repeats"] = 10
    path = Path(f"/tmp/config_seed{seed}.yaml")
    path.write_text(yaml.safe_dump(config))
    return path


@app.function(image=image, cpu=4, memory=16384, timeout=10 * 3600, volumes=VOLUMES)
def run_seed(seed: int, smoke: bool = False) -> str:
    """The whole pipeline for one seed: Experiment 1 (MLP), 1b (flow policy) and 1c (selector).
    Resumable: a stage whose outputs already exist in the run directory is skipped, so a
    cancelled or timed-out run continues where it stopped instead of starting over."""
    import yaml

    config_path = seed_config(seed, smoke)
    run_dir = Path(yaml.safe_load(config_path.read_text())["experiment"]["output_dir"])
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    log = run_dir / "logs" / "modal_run.log"
    py, C = sys.executable, str(config_path)
    inits = ["--max-inits", "2"] if smoke else []
    ck = run_dir / "checkpoints"
    # (command, files that prove the stage finished)
    stages = [
        ([py, "scripts/00_audit_setup.py", "--config", C], [run_dir / "manifest.json"]),
        ([py, "scripts/01_replay_expert.py", "--config", C, "--split", "train", "--max-anchors", "5" if smoke else "60", "--no-video"],
         [run_dir / "replay_audit_summary.json"]),
        ([py, "scripts/02_prepare_policy.py", "--config", C] + (["--baseline-episodes", "2"] if smoke else []),
         [ck / "pi0.pt", run_dir / "pi0_baseline.json"]),
        ([py, "scripts/03_collect_branches.py", "--config", C, "--split", "train", "--sample"], [run_dir / "branches_sample.json"]),
        *[([py, "scripts/03_collect_branches.py", "--config", C, "--split", sp], [run_dir / f"branches_{sp}.h5", run_dir / f"branch_stats_{sp}.json"])
          for sp in ("train", "val", "test")],
        ([py, "scripts/04_train_error.py", "--config", C], [ck / "error_round0_history.pt", run_dir / "error_report_history.json"]),
        ([py, "scripts/04_train_error.py", "--config", C, "--no-history"], [ck / "error_round0_nohist.pt", run_dir / "error_report_nohist.json"]),
        ([py, "scripts/05_audit_gradient.py", "--config", C] + (["--val-episodes", "3"] if smoke else []), [run_dir / "gradient_audit_summary.json"]),
        *[([py, "scripts/06_train_policy.py", "--config", C, "--variant", v], [ck / f"pi_{v}" / "pi.pt", run_dir / f"policy_training_{v}.json"])
          for v in ("bc", "error", "nohist", "smooth")],
        ([py, "scripts/07_eval_closed_loop.py", "--config", C, "--tag", "main"] + inits, [run_dir / "eval_latency_main.json"]),
        # Experiment 1b: flow-matching policy class
        ([py, "scripts/02_prepare_policy.py", "--config", C, "--policy", "flow"] + (["--baseline-episodes", "2"] if smoke else []),
         [ck / "pi0_flow.pt", run_dir / "pi0_flow_baseline.json"]),
        ([py, "scripts/05b_audit_flow_gradient.py", "--config", C] + (["--checkpoints", "2,4", "--val-episodes", "3"] if smoke else []),
         [run_dir / "flow_gradient_audit.json"]),
        ([py, "scripts/05b_audit_flow_gradient.py", "--config", C, "--lr", "0.00001", "--checkpoints", "1,2,5,10,25", "--gate-steps", "25", "--tag", "lr1e-5"]
         + (["--val-episodes", "3"] if smoke else []), [run_dir / "flow_gradient_audit_lr1e-5.json"]),
        *[([py, "scripts/06_train_policy.py", "--config", C, "--policy", "flow", "--variant", v], [ck / f"pi_{v}_flow" / "pi.pt", run_dir / f"policy_training_{v}_flow.json"])
          for v in ("bc", "error", "smooth")],
        ([py, "scripts/07_eval_closed_loop.py", "--config", C, "--policies", "pi0_flow,bc_flow,error_flow,smooth_flow,error_flow_w0.3", "--tag", "flow"] + inits,
         [run_dir / "eval_latency_flow.json"]),
        # Experiment 1c: critic as a selector
        ([py, "scripts/09_audit_selection.py", "--config", C, "--samples", "8"] + (["--episodes", "3"] if smoke else []), [run_dir / "selection_audit.json"]),
        ([py, "scripts/07_eval_closed_loop.py", "--config", C, "--policies", "select_best8_flow,select_worst8_flow,select_best32_flow", "--tag", "sel"] + inits,
         [run_dir / "eval_latency_sel.json"]),
        ([py, "scripts/08_report.py", "--config", C], []),  # always regenerated
    ]
    started, ran, skipped = time.time(), 0, 0
    for command, outputs in stages:
        if outputs and all(path.is_file() for path in outputs):
            print("= skipping finished stage:", " ".join(command[1:3]), *command[3:], flush=True)
            skipped += 1
            continue
        sh(command, log)
        ran += 1
        runs_volume.commit()
        data_volume.commit()  # the first seed writes the locked split
    return (f"seed {seed}{' (smoke)' if smoke else ''}: {ran} stages run, {skipped} skipped, "
            f"{(time.time() - started) / 3600:.2f} h -> {run_dir}")


VIDEO_PAIRS = [  # (left policy, right policy, delay): the comparisons the report discusses
    ("pi0", "pi0_flow", 0),            # policy class: MLP vs flow base policy
    ("bc", "error", 0),                # Experiment 1: MLP fine-tune vs critic-regularised
    ("bc", "error", 4),
    ("bc_flow", "error_flow", 0),      # Experiment 1b: flow fine-tune vs critic-regularised
    ("bc_flow", "error_flow", 4),
    ("bc_flow", "error_flow_w0.3", 0), # the critic term when it dominates
    ("bc_flow", "select_best8_flow", 0),   # Experiment 1c: the critic as a selector
    ("bc_flow", "select_worst8_flow", 0),
]


@app.function(image=image, cpu=4, memory=16384, timeout=4 * 3600, volumes=VOLUMES)
def render_videos(seed: int, inits: str = "0,1,2") -> str:
    """Side-by-side closed-loop videos (software-rendered) for one finished seed, written to
    ``<run_dir>/videos/`` in the runs volume."""
    config_path = seed_config(seed, smoke=False)
    run_dir = Path(f"{RUNS}/maniskill_pickcube_seed{seed}")
    log = run_dir / "logs" / "modal_videos.log"
    done = []
    for init in (int(i) for i in inits.split(",") if i.strip()):
        for left, right, delay in VIDEO_PAIRS:
            target = run_dir / "videos" / f"eval_pair_{left}_vs_{right}_r{delay}_init{init}_seed*.mp4"
            if list(target.parent.glob(target.name)):
                continue
            try:
                sh([sys.executable, "scripts/07_eval_closed_loop.py", "--config", str(config_path), "--video", f"{left},{right},{delay}", "--video-init", str(init)], log)
                done.append(f"{left}_vs_{right}_r{delay}_init{init}")
            except RuntimeError as exc:  # a missing checkpoint should not stop the other videos
                print("skipped:", exc, flush=True)
            runs_volume.commit()
    return f"seed {seed}: {len(done)} videos rendered -> {run_dir}/videos"


@app.local_entrypoint()
def main(seeds: str = "", prepare: bool = False, smoke: bool = False, wait: bool = False, videos: str = "", inits: str = "0,1,2") -> None:
    """--prepare: download and replay the official demos.  --seeds 17,18: start those seeds in
    parallel (add --smoke for a short end-to-end check).  Seeds are *spawned*: they keep running
    after this command returns and after the laptop sleeps (run with ``modal run --detach``).
    Progress is in each seed's ``logs/modal_run.log`` in the ``cte-runs`` volume; re-running the
    same seed resumes at the first unfinished stage.  --wait blocks until the seeds finish."""
    if prepare:
        print(prepare_data.remote())
    if videos:
        for seed in (int(v) for v in videos.split(",") if v.strip()):
            call = render_videos.spawn(seed, inits=inits)
            print(f"videos for seed {seed}: spawned {call.object_id}")
            if wait:
                print(call.get())
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    if not seed_list:
        if not prepare and not videos:
            print("nothing to do: pass --prepare, --seeds 17,18,... and/or --videos 17")
        return
    calls = {seed: run_seed.spawn(seed, smoke=smoke) for seed in seed_list}
    for seed, call in calls.items():
        print(f"seed {seed}: spawned {call.object_id}")
    if wait:
        for seed, call in calls.items():
            print(call.get())
    else:
        print("running detached; check:  modal volume get cte-runs maniskill_pickcube_seed<N>/logs/modal_run.log - | grep -E '^\\+ |exit [0-9]' | tail")
