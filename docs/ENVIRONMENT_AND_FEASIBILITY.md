# Environment Deployment and Feasibility

## Status

This is a static setup and code-structure pass only. No packages, datasets, or model weights have been installed or downloaded; no simulator, training, or tests have been run. Version compatibility, replay determinism, memory use, and runtime remain unverified.

The target machine is now Windows with an NVIDIA GPU. The ManiSkill support table lists Windows CPU simulation and rendering, but not Windows GPU simulation. Thus the first state-based `physx_cpu` audit can be attempted natively on Windows; the GPU may still accelerate PyTorch model work. The current LeRobot LIBERO guide declares LIBERO Linux-only, so use WSL2 Ubuntu or native Linux for that stage. ManiSkill's support table does not list WSL GPU simulation or rendering. Do not start data collection until the actual target host passes the replay audit.

## Environment Separation

Use separate Python 3.10 environments: native Windows for the initial ManiSkill CPU path and WSL2/Linux for LIBERO. The repository's `pyproject.toml` contains only shared config/data utilities; `envs/maniskill/requirements.txt` and `envs/libero/requirements.txt` are bootstrap overlays, not lock files. Pin exact Python, PyTorch, simulator, LeRobot commit, checkpoint revision, and dataset hashes after a clean install and successful audit. See [SETUP_WINDOWS_AND_LIBERO.md](SETUP_WINDOWS_AND_LIBERO.md) for PowerShell and WSL2 commands.

### ManiSkill: first implementation milestone

Target the native Windows `physx_cpu` path with `PickCube-v1`, state observations, and `pd_ee_delta_pos`. The following are deployment instructions; they have not been run here:

```bash
python3.10 -m venv .venv-maniskill
source .venv-maniskill/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for the target host first.
python -m pip install torch
python -m pip install -e '.[dev]'
python -m pip install -r envs/maniskill/requirements.txt
```

Download and replay the official motion-planning demonstrations using the commands in `README_Chunk_Trajectory_Error.md`. Place the source file at `data/raw/maniskill/trajectory.h5` to match the starter config, and save replayed HDF5/JSON under `data/processed`; keep downloaded assets outside version control.

The first executable milestone is only `00_audit_setup.py` and `01_replay_expert.py`. Before branch collection, confirm action/state boundary counts (`T` actions and `T+1` states), controller and simulator metadata, same-anchor restore-vs-continuous replay within a predeclared tolerance, and an expert-vs-expert cost near zero. If restore differs, test deterministic prefix replay before accepting labels.

### LIBERO: later VLA milestone

Use a separate WSL2/Linux environment and keep `MUJOCO_GL=egl` for headless rendering where supported. Verify CUDA visibility and EGL initialization inside WSL2 before evaluation; native Linux is the fallback if WSL rendering fails. After cloning a specific LeRobot revision, install its LIBERO extra in that checkout. These are instructions only; the checkout and environment are not created by this pass:

```bash
python3.10 -m venv .venv-libero
source .venv-libero/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
git clone https://github.com/huggingface/lerobot.git external/lerobot
cd external/lerobot
# Pin a reviewed commit before installation; do not use a moving main branch for experiments.
python -m pip install -e '.[libero]'
cd ../..
python -m pip install -r envs/libero/requirements.txt
```

Fetch only the target task HDF5 and `lerobot/smolvla_libero` checkpoint after the ManiSkill contracts are stable. Audit camera mapping, 8-D policy state, 7-D normalized action, task identity, episode correspondence, and fixed-noise adapter output before any fine-tuning.

Do not treat the original LIBERO HDF5, LeRobot-converted dataset, and model checkpoint as index-aligned until episode IDs and timestamps have been verified. Pin each source revision/hash in the generated run manifest.

## Feasibility Gates

1. **Host gate:** Linux environment can import the selected simulator stack and instantiate the configured task without rendering or device errors.
2. **Data gate:** official demo metadata matches task/controller/backend; observations can be reconstructed and action/state time indices are explicit.
3. **Replay gate:** uninterrupted expert replay and snapshot-restore replay agree within the configured tolerance; failed attempts are recorded rather than silently dropped.
4. **Label gate:** same-start expert-vs-expert yields approximately zero `J`; a small bounded continuous-action perturbation yields a measurable nonzero cost with the same fixed rollout length.
5. **Policy gate (LIBERO only):** official baseline works; custom adapter agrees with reference inference under fixed inputs/noise; auxiliary-only backprop gives finite nonzero policy gradients while critic parameters remain gradient-free.

The first three gates are required before bulk branch collection. The last two are required before policy updates. Results, timings, and supported-platform claims must remain marked pending until measured on the target host.

## Entry Points and Planned Growth

```text
configs/                         # environment and experiment settings
envs/                            # isolated environment bootstrap requirements
src/chunk_trajectory_error/
  backends/                      # ManiSkill and LIBERO implementations
  contracts.py                   # backend and temporal-shape contracts
  replay.py                      # trajectory alignment and J calculation
  policies/                      # MLP BC and differentiable SmolVLA adapters (next phase)
  error_net.py                   # history-conditioned surrogate (next phase)
  scheduler.py                   # shared chunk handover logic (next phase)
  metrics.py                     # expert deviation, handover jump, success, latency
scripts/00_audit_setup.py        # setup audit
scripts/01_replay_expert.py      # split/replay/restore audit
scripts/02-08_*.py               # later stages, per project README
tests/                           # temporal contracts, replay alignment, scheduler
runs/                            # generated manifests, checkpoints and reports
```

The `00` and `01` scripts are intentionally explicit placeholders until the host-specific ManiSkill state and replay APIs are pinned. They must not report a successful audit before performing one.