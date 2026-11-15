# Chunk Trajectory Error

Can a learned trajectory-error critic improve action-chunk policies? This repository holds the code, data and results for a simulator-labelled study on ManiSkill `PickCube-v1` (10-703 course project).

A chunk policy predicts `H = 20` actions and executes `e = 5` of them. For a candidate action tail, the simulator restores the same state, runs `K = 10` steps of the candidate and of the expert, and records the mean squared end-effector deviation `J`. A critic `E_phi(history, robot state, action tail, delay)` is regressed on these labels, frozen, and added to the behaviour-cloning loss as `lambda * E_phi`. The weight `lambda` is chosen by simulator-recomputed `J`, never by the critic.

## Status (2026-11-15)

| Part | State | Where |
|---|---|---|
| Experiment 1: expert-deviation critic, ManiSkill PickCube-v1 | Stages `00`-`08` implemented and run twice: on a CPU-only Linux container with locally generated demonstrations (outputs in `runs/`), and on the GPU workstation with the official demonstrations (outputs not in this repository) | `src/`, `scripts/`, `runs/maniskill_pickcube/` |
| Experiment 2: sim-to-sim gap critic (latency and actuation shifts) | Run on the GPU workstation; **code and outputs not yet in this repository** | Specification: [docs/SIM2SIM_PILOT_SPEC.md](docs/SIM2SIM_PILOT_SPEC.md) |
| LIBERO / SmolVLA | Not implemented (configuration and backend stub only) | `configs/libero_spatial.yaml`, `src/chunk_trajectory_error/backends/libero.py` |

## Main findings so far

Both experiments are negative for the critic.

- The critic fits held-out labels well (history variant, test split: Pearson 0.972, within-anchor Spearman 0.880), and a small gradient step through the frozen critic lowers the true `J` (gradient audit passes: -18.8 % at step 0.05).
- Adding the critic to the BC loss does not improve closed-loop success. In the container run (150 initial states per delay), the selected-`lambda` critic policy succeeds 38 / 45 / 64 % at delay `r` = 0 / 2 / 4, against 44 / 48 / 80 % for BC fine-tuning with the same budget; the pooled paired difference is -8.4 pp [-13.6, -3.3]. An action-smoothness penalty matches or beats BC.
- With a larger weight the policy exploits the critic: as `lambda` goes from 0 to 1, the predicted `J` falls from 4.2e-5 to 3.5e-6 while the simulator-recomputed `J` rises from 2.8e-5 to 2.6e-4.
- In the sim-to-sim pilot (latency `d = 1` and stiffness `s = 0.4` shifts), the gap critic did not beat the smoothness penalty or same-budget fine-tuning on the target simulator.

Full numbers: [runs/maniskill_pickcube/results.md](runs/maniskill_pickcube/results.md) and the mid-point report (`report/midterm/`).

## Repository layout

```
configs/                 experiment configurations (ManiSkill PickCube, LIBERO Spatial)
src/chunk_trajectory_error/
  pipeline.py            shared stage logic used by scripts/00-08
  scheduler.py           chunk scheduler (H, e, delay r) shared by data collection and evaluation
  replay.py              snapshot / restore and branch replay
  error_net.py           history-conditioned trajectory-error critic
  objectives.py          BC loss, critic regression, frozen-critic policy objective
  policies/mlp_bc.py     MLP chunk policy
  backends/maniskill.py  ManiSkill adapter (libero.py is a stub)
scripts/00-08            the pipeline stages, run in order (see below)
tests/                   unit tests (contracts, scheduler, reader, pipeline, dual networks)
envs/                    dependency lists; maniskill/requirements.lock.linux-py310.txt is the full tested freeze
data/                    PickCube-v1 demonstrations: raw and replay-converted (see Data)
runs/maniskill_pickcube/ outputs of the container run: checkpoints, branch labels, evaluations, results.md
report/midterm/          figure scripts, TikZ diagram and LaTeX body of the mid-point report
docs/                    design specification, feasibility gates, proposal, setup guides, audit report, sim-to-sim spec
```

## Setup (Linux, tested)

Tested on Ubuntu 24.04 with Python 3.10.20, PyTorch 2.14.1, ManiSkill 3.0.1, SAPIEN 3.0.3 and numpy 1.26.4, using `physx_cpu` simulation with one environment.

```bash
python3.10 -m venv .venv-maniskill
source .venv-maniskill/bin/activate
python -m pip install --upgrade pip
python -m pip install -r envs/maniskill/requirements.lock.linux-py310.txt   # exact tested versions
python -m pip install -e ".[dev]"
python -m pytest -q
```

If you need a different PyTorch build (for example a specific CUDA version), install it first from the [PyTorch selector](https://pytorch.org/get-started/locally/) and then install the lock file without its `torch` line.

Two known pitfalls:

- ManiSkill initialises Vulkan even when nothing is rendered. On a machine without a GPU driver, install Mesa lavapipe and point to it, for example `export VK_ICD_FILENAMES=/opt/lavapipe/lvp_icd.json` (the path depends on how lavapipe was installed).
- `mplib 0.1.1` (needed only to generate demonstrations with the motion planner) crashes under numpy 2.x; keep `numpy<2`.

Windows, WSL2 and the planned LIBERO environment are covered in [docs/SETUP_WINDOWS_AND_LIBERO.md](docs/SETUP_WINDOWS_AND_LIBERO.md).

## Data

- `data/raw/maniskill/trajectory.h5` / `.json`: 600 successful `PickCube-v1` demonstrations generated locally with ManiSkill's official motion-planning solution, because the Hugging Face download was blocked on the container. They are generated the same way as the official set but are not the published file.
- `data/processed/maniskill/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` / `.json`: the same demonstrations replayed into the `pd_ee_delta_pos` controller with state observations (42-D observation, 4-D action), plus `splits.json` (400 / 100 / 100 episodes).

To use the official demonstrations instead, download them with `python -m mani_skill.utils.download_demo PickCube-v1`, copy `trajectory.h5` / `.json` to `data/raw/maniskill/`, replay them:

```bash
python -m mani_skill.trajectory.replay_trajectory --traj-path data/raw/maniskill/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b physx_cpu
```

move the converted files to `data/processed/maniskill/` (the paths are set in `configs/maniskill_pickcube.yaml`), and rerun the pipeline from stage `00`.

## Reproducing Experiment 1

Run from the repository root inside the environment. Every stage reads `configs/maniskill_pickcube.yaml` and writes under `runs/maniskill_pickcube/`; rerunning overwrites the stored outputs.

```bash
C=configs/maniskill_pickcube.yaml
python scripts/00_audit_setup.py --config $C                       # manifest.json
python scripts/01_replay_expert.py --config $C --split train --max-anchors 60 --video-anchor-index 1
python scripts/02_prepare_policy.py --config $C                    # checkpoints/pi0.pt, pi0_baseline.json
python scripts/03_collect_branches.py --config $C --split train --sample   # two-row label gate
for s in train val test; do python scripts/03_collect_branches.py --config $C --split $s; done
python scripts/04_train_error.py --config $C                       # history critic (specified variant)
python scripts/04_train_error.py --config $C --no-history          # ablation
python scripts/05_audit_gradient.py --config $C                    # gradient_audit.csv
for v in bc error nohist smooth; do python scripts/06_train_policy.py --config $C --variant $v; done
python scripts/07_eval_closed_loop.py --config $C                  # eval_*.parquet, latency
python scripts/08_report.py --config $C                            # results.md, figures/
```

Optional runs that appear in the recorded results: `04 --state-source full_obs`, `04 --target sqrtJ` (exploratory, outside the specified loss), `06 --variant error_fullobs | error_sqrt`, and one critic refresh round (`03 --round 1 --checkpoint <policy> --policy-perturbed 1 --expert-perturbed 0`, `04 --round 1`, `06 --variant error --round 1`). `07 --policies a,b --tag x` shards the evaluation across processes; `08` merges every `eval_*_<tag>.parquet`.

The container run, including the optional variants, took about two hours of wall time on a 2-core CPU; stages `03` (branch labels) and `07` (closed-loop evaluation, sharded over two processes) take most of it.

## Stored outputs (`runs/maniskill_pickcube/`)

| File | Content |
|---|---|
| `results.md`, `figures/` | the report generated by stage `08` |
| `manifest.json`, `replay_audit*.csv/json/mp4` | environment versions and the replay / determinism audit |
| `branches_{train,val,test}*.h5`, `branch_stats_*.json`, `anchors_*.parquet` | simulator labels `J` for every candidate tail |
| `checkpoints/` | `pi0`, the critics and every fine-tuned policy |
| `error_report_*.json` | critic accuracy on held-out branches |
| `gradient_audit*.csv/json` | does a gradient step through the critic lower the true `J`? |
| `policy_training_*.json` | weight selection on validation for each variant |
| `eval_*.parquet`, `eval_latency_*.json` | closed-loop episodes, chunk boundaries, recomputed `J`, latency |
| `logs/` | stage logs |

## Mid-point report

`report/midterm/` holds what is needed to rebuild the report figures; the report itself is on Overleaf, using the course template. See [report/midterm/README.md](report/midterm/README.md).

## Not in this repository yet

- The sim-to-sim pilot code and outputs (Phases A-C, latency and actuation shifts).
- The Experiment 1 outputs from the GPU workstation run with the official demonstrations. The success counts in report Figure 2 come from that run, so they differ from `runs/maniskill_pickcube/results.md`.

Both are on the GPU workstation and should be added from there.
