# Chunk Trajectory Error: original setup guide (Windows / WSL2 / LIBERO)

> This is the original setup guide, kept for the Windows and LIBERO instructions. Its status paragraphs were written before the ManiSkill pipeline was implemented and are out of date: stages `00`-`08` are implemented and have been run (see the [root README](../README.md)). The [ManiSkill pipeline](#maniskill-pipeline-0008) section below is current.

Windows-oriented setup and run guide for the project described in [README_Chunk_Trajectory_Error.md](DESIGN_SPEC.md) and [Chunk_Trajectory_Error_Proposal_EN.pdf](proposal.pdf).

## Read This First

> **Status (2026-11-12).** The ManiSkill line is implemented end to end and has been run once on a GPU-less Linux container: `scripts/00`–`08`, `ManiSkillBackend`, the chunk scheduler and the shared pipeline module. Numbers, figures and the acceptance check are in [runs/maniskill_pickcube/results.md](../runs/maniskill_pickcube/results.md); the run order is in [ManiSkill pipeline](#maniskill-pipeline-0008) below. The demonstrations used there were generated locally with ManiSkill's motion planner. Nothing has been run on native Windows, and LIBERO / SmolVLA is still unimplemented; the Windows and LIBERO paragraphs below describe plans, not tested paths.

The repository currently contains the dual-network core, but not a runnable experiment. `00_audit_setup.py`, `01_replay_expert.py`, and both simulator adapters deliberately raise `NotImplementedError`. Dataset readers, simulator replay, optimizer/training loops, differentiable SmolVLA sampling, scheduler, and evaluation stages are not implemented yet. Installing the environments or running the project entry points below will not make those stages work. Do not start training or claim a feasibility result until the replay adapter and audit are implemented.

What exists now:

- Shared package metadata and small temporal/trajectory contracts.
- ManiSkill and LIBERO starter configurations.
- Separate bootstrap dependency lists.
- Configurable MLP chunk policy and history-conditioned trajectory-error network.
- Behavior-cloning loss, critic regression loss, and frozen-critic policy objective.
- First-stage script skeletons and unit tests, including the intended policy-gradient path check.

What remains before the first project audit:

- Implement the ManiSkill HDF5 reader, state snapshot/restore, expert replay, and audit outputs.
- Pin and record a tested Windows/Python/PyTorch/ManiSkill combination.
- Run the setup, replay, and snapshot determinism gates on the target machine.
- Add data loaders, optimizer loops, action scaling/masking, CLI integration, and checkpoint/report handling.
- Implement the differentiable SmolVLA sampler before LIBERO policy fine-tuning.

## Dual-Network Core

The neural components are in place and simulator-independent:

- `MLPChunkPolicy` maps a flattened state observation to `[batch, horizon, action_dim]` actions.
- `HistoryConditionedErrorNetwork` consumes `[batch, L, 3]` relative end-effector history, current robot state, a `[batch, K, action_dim]` candidate action tail, and delay; it returns a nonnegative estimate of `J`.
- `surrogate_regularized_policy_loss` combines the caller-provided imitation loss with the predicted cost. It disables gradients on error-network parameters while preserving gradients through the candidate actions into the policy.

Input dimensions are constructor arguments because the ManiSkill observation/action shapes and the final preprocessed features must be confirmed by the data audit. The configured hidden widths are starters, not tuned values. These components do not yet include rollout label generation, training loops, or a runnable policy pipeline.

## Windows Platform Plan

| Workload | Recommended location | Device use | Status |
|---|---|---|---|
| ManiSkill `PickCube-v1` first audit | Native Windows | `physx_cpu` simulation; NVIDIA GPU may be used by PyTorch | Supported path to try; not yet tested in this repository |
| ManiSkill GPU simulation | Linux with NVIDIA GPU | GPU simulation | Not supported by the current ManiSkill Windows support matrix |
| LIBERO + SmolVLA | WSL2 Ubuntu or native Linux | WSL2 CUDA can accelerate policy work if NVIDIA driver passthrough is healthy | LIBERO requires Linux; WSL2 rendering and package compatibility must be smoke-tested |

A high-end GPU helps PyTorch model work, but does not enable ManiSkill GPU simulation on native Windows. ManiSkill's current support table also lists WSL as lacking its GPU simulation and rendering paths. Use native Windows for the state-based ManiSkill CPU milestone; use WSL2/Linux for LIBERO. For the most reproducible full study, a native Linux NVIDIA machine is the preferred target.

## Native Windows: ManiSkill Environment

Use 64-bit Python 3.10 and a recent NVIDIA driver. Keep this environment separate from WSL2/Linux and from the future LIBERO environment. These commands create an environment; they do not imply that the project audit script is implemented.

In PowerShell, from the repository root:

```powershell
py -3.10 -m venv .venv-maniskill
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-maniskill\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build selected for the Windows/NVIDIA setup from the official [PyTorch install selector](https://pytorch.org/get-started/locally/). Then install the repository and ManiSkill dependencies:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
python -m pip install -e ".[dev]"
python -m pip install -r envs/maniskill/requirements.txt
```

Replace `cuXXX` with the CUDA wheel index shown by the PyTorch selector. If using CPU-only PyTorch, select its CPU install command instead. ManiSkill simulation remains `physx_cpu` either way; the GPU is available to PyTorch where the model and installed wheel support it.

After the environment is installed, an upstream-only smoke test is:

```powershell
python -m mani_skill.examples.demo_random_action
```

This checks the upstream package, not this project's replay code. Stop and resolve any simulator, driver, or asset errors before proceeding.

## ManiSkill Demo Data on Windows

Download the official `PickCube-v1` demonstration:

```powershell
python -m mani_skill.utils.download_demo "PickCube-v1"
$source = Join-Path $HOME ".maniskill\demos\PickCube-v1\motionplanning\trajectory.h5"
New-Item -ItemType Directory -Force data\raw\maniskill | Out-Null
Copy-Item $source data\raw\maniskill\trajectory.h5
```

The default download location can vary. If `$source` does not exist, use the location printed by the downloader and update the path. Replay the source trajectory with the controller and backend specified in the experiment plan:

```powershell
python -m mani_skill.trajectory.replay_trajectory `
  --traj-path $source `
  --use-first-env-state `
  -c pd_ee_delta_pos `
  -o state `
  --save-traj `
  --num-envs 1 `
  -b physx_cpu
```

Keep the original HDF5 and its JSON metadata. The replay output is only prepared data; it is not yet a validated label set. Confirm the generated output path and update [maniskill_pickcube.yaml](../configs/maniskill_pickcube.yaml) if it differs from the configured path.

### Project audit commands (not runnable yet)

The intended commands are:

```powershell
python scripts/00_audit_setup.py --config configs/maniskill_pickcube.yaml
python scripts/01_replay_expert.py --config configs/maniskill_pickcube.yaml --split train
```

At the current code state, both commands intentionally terminate with `NotImplementedError`. They are shown as target commands, not as a completed workflow. The implementation must produce the manifest, split record, replay comparison, and video before branch labels can be collected.

## WSL2: LIBERO Environment

Use WSL2 Ubuntu for LIBERO because the current LeRobot LIBERO guide requires Linux. Install/update WSL2 and Ubuntu from an elevated PowerShell session if needed:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart Windows if requested, launch Ubuntu, then check whether the NVIDIA GPU is visible inside WSL:

```bash
nvidia-smi
```

If that command fails, first install/update the Windows NVIDIA driver with WSL CUDA support. Do not install a separate Linux display driver inside WSL. WSL2 GPU visibility does not by itself prove MuJoCo EGL rendering works.

Inside Ubuntu, keep the repository under the Linux filesystem (for example `~/work/final_project`) rather than `/mnt/c` for better file and simulator performance. Install system prerequisites and create a separate environment:

```bash
sudo apt update
sudo apt install -y git build-essential python3.10 python3.10-venv python3-pip libgl1 libegl1
cd ~/work/final_project
python3.10 -m venv .venv-libero
source .venv-libero/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Install the PyTorch CUDA build that matches the WSL driver, using the official [PyTorch install selector](https://pytorch.org/get-started/locally/). Then clone and pin LeRobot to a reviewed commit before installing its LIBERO extra:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
git clone https://github.com/huggingface/lerobot.git external/lerobot
git -C external/lerobot checkout <REVIEWED_LEROBOT_COMMIT>
python -m pip install -e 'external/lerobot[libero]'
python -m pip install -r envs/libero/requirements.txt
export MUJOCO_GL=egl
```

Replace both placeholders with the actual CUDA wheel and reviewed commit; do not record a moving `main` branch as an experiment revision. Check `nvidia-smi` and run the LeRobot/LIBERO upstream smoke test before this project's adapter is implemented. If EGL cannot initialize in WSL2, use native Linux or a Linux GPU host rather than treating a headless-rendering failure as a model failure.

The target task config is [libero_spatial.yaml](../configs/libero_spatial.yaml). Dataset and checkpoint downloads are intentionally deferred until the LIBERO adapter and the exact revisions are fixed.

## Tests and Current Readiness

The current tests cover only small pure-Python contracts; they do not test either simulator. Run them after installing the development extra:

```powershell
python -m pytest
```

This test command is not evidence that the full project is ready. Current readiness is:

- Environment instructions and starter configs: present, but dependency versions are not locked or machine-tested.
- Native Windows ManiSkill CPU installation: plausible per upstream support matrix, not verified here.
- Windows ManiSkill GPU simulation: unsupported by the current upstream support matrix.
- WSL2 LIBERO: platform-compatible in principle; CUDA and EGL must be checked on the actual PC.
- This project's simulator adapters, audits, policy, and training pipeline: not implemented, so no end-to-end project run is available yet.

## ManiSkill pipeline (00–08)

Run from the repository root inside the ManiSkill environment (`pip install -e ".[dev]"` and `pip install -r envs/maniskill/requirements.txt`). Every stage reads `configs/maniskill_pickcube.yaml` and writes under `runs/maniskill_pickcube/`.

```bash
C=configs/maniskill_pickcube.yaml
python scripts/00_audit_setup.py --config $C                       # manifest.json
python scripts/01_replay_expert.py --config $C --split train --max-anchors 60 --video-anchor-index 1
python scripts/02_prepare_policy.py --config $C                    # checkpoints/pi0.pt, pi0_baseline.json
python scripts/03_collect_branches.py --config $C --split train --sample   # two-row label gate
for s in train val test; do python scripts/03_collect_branches.py --config $C --split $s; done
python scripts/04_train_error.py --config $C                       # history (specified variant)
python scripts/04_train_error.py --config $C --no-history          # ablation
python scripts/05_audit_gradient.py --config $C                    # gradient_audit.csv
for v in bc error nohist smooth; do python scripts/06_train_policy.py --config $C --variant $v; done
python scripts/07_eval_closed_loop.py --config $C                  # eval_*.parquet, latency
python scripts/08_report.py --config $C                            # results.md, figures/
python -m pytest -q
```

Optional runs used in the recorded results: `04 --state-source full_obs`, `04 --target sqrtJ` (exploratory, outside the specified loss), `06 --variant error_fullobs | error_sqrt`, and one critic refresh round (`03 --round 1 --checkpoint <policy> --policy-perturbed 1 --expert-perturbed 0`, `04 --round 1`, `06 --variant error --round 1`). `07 --policies a,b --tag x` shards the evaluation across processes; `08` merges every `eval_*_<tag>.parquet`.

Design choices made during implementation (all recorded in the config or the stage outputs):

- Branches start from the recorded expert state in the demonstration file (`env_states[t + r]`); the replay audit certifies that path (`hdf5_state_restore`, `snapshot_branch_repeatability`).
- Episode tails are padded with a hold command (zero translation, last gripper command) for the BC target; the error term is masked outside `L <= t <= T - r - K`.
- The error network sees the robot part of the state observation (`observation.robot_state_slice`) unless `--state-source full_obs` is given.
- Weights (`lambda`, `beta`) are selected on validation by the simulator-recomputed J of the policy's own chunk.
- `environment.max_episode_steps` is 200 for evaluation; the demonstration metadata says 50, which is shorter than every demonstration.

## Handoff to Claude Code

Open this folder on the Windows PC, start Claude Code in the project root, and give it this task:

> Read `README.md`, `README_Chunk_Trajectory_Error.md`, and `ENVIRONMENT_AND_FEASIBILITY.md` first. Prepare the native Windows Python 3.10 ManiSkill environment using the PowerShell instructions in `README.md`, with `physx_cpu`; use the NVIDIA GPU for PyTorch only. Do not select ManiSkill GPU simulation on Windows. The current `00` and `01` scripts and backend adapters are placeholders: implement the ManiSkill dataset reader, setup audit, deterministic snapshot/restore or prefix-replay audit, output manifest/split report, and one replay video before claiming these commands work. Keep the existing algorithm/data contracts and report any API/version incompatibility instead of faking a pass. Run only the upstream smoke test and focused project tests needed for this first milestone. Do not start policy training or LIBERO yet. Record exact Python, PyTorch, ManiSkill, OS, driver, and data versions in the audit output. After all replay gates pass, report the exact commands and artifacts, then stop for review.

For LIBERO later, use WSL2 Ubuntu or native Linux and first verify CUDA visibility and MuJoCo EGL rendering. Do not treat successful environment installation as proof that the project pipeline is implemented.

## Configuration Reference

- [ManiSkill PickCube](../configs/maniskill_pickcube.yaml): `H=20`, `e=5`, `L=15`, `K=10`, train delay `r=0`.
- [LIBERO Spatial + SmolVLA](../configs/libero_spatial.yaml): `H=50`, `e=10`, `L=30`, `K=20`, train delay `r=0`.
- [Environment feasibility and gates](ENVIRONMENT_AND_FEASIBILITY.md): data, replay, label, and policy acceptance criteria.
- [Research/implementation specification](DESIGN_SPEC.md): algorithm, data contract, and full planned program list.