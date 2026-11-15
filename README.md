# Chunk Trajectory Error

Can a learned trajectory-error critic improve action-chunk policies? This repository holds the code, data and results for a simulator-labelled study on ManiSkill `PickCube-v1` (10-703 course project).

A chunk policy predicts `H = 20` actions and executes `e = 5` of them. For a candidate action tail, the simulator restores the same state, runs `K = 10` steps of the candidate and of the expert, and records the mean squared end-effector deviation `J`. A critic `E_phi(history, robot state, action tail, delay)` is regressed on these labels, frozen, and added to the behaviour-cloning loss as `lambda * E_phi`. The weight `lambda` is chosen by simulator-recomputed `J`, never by the critic.

## Status (2026-11-15)

| Part | State | Where |
|---|---|---|
| Experiment 1: expert-deviation critic, ManiSkill PickCube-v1 | Stages `00`-`08` implemented and run twice: on a CPU-only Linux container with locally generated demonstrations, and on the GPU workstation with the official demonstrations | `src/`, `scripts/`, `runs/maniskill_pickcube/` (container), `runs/maniskill_pickcube_official/` (GPU, official demonstrations) |
| Experiment 2: sim-to-sim gap critic (latency `d = 1` and actuation `s = 0.4` shifts), Phases A-C | Run on the GPU workstation; code, configurations and summary outputs in this repository | `scripts/sim2sim/`, `configs/sim2sim_pilot*.yaml`, `runs/sim2sim_pilot/`; specification [docs/SIM2SIM_PILOT_SPEC.md](docs/SIM2SIM_PILOT_SPEC.md) |
| LIBERO / SmolVLA | Not implemented (configuration and backend stub only) | `configs/libero_spatial.yaml`, `src/chunk_trajectory_error/backends/libero.py` |

## Main findings so far

Both experiments are negative for the critic.

- The critic fits held-out labels well (history variant, test split: Pearson 0.972, within-anchor Spearman 0.880), and a small gradient step through the frozen critic lowers the true `J` (gradient audit passes: -18.8 % at step 0.05).
- Adding the critic to the BC loss does not improve closed-loop success. In the container run (150 initial states per delay), the selected-`lambda` critic policy succeeds 38 / 45 / 64 % at delay `r` = 0 / 2 / 4, against 44 / 48 / 80 % for BC fine-tuning with the same budget; the pooled paired difference is -8.4 pp [-13.6, -3.3]. An action-smoothness penalty matches or beats BC.
- With a larger weight the policy exploits the critic: as `lambda` goes from 0 to 1, the predicted `J` falls from 4.2e-5 to 3.5e-6 while the simulator-recomputed `J` rises from 2.8e-5 to 2.6e-4.
- In the sim-to-sim pilot (latency `d = 1` and stiffness `s = 0.4` shifts), the gap critic did not beat the smoothness penalty or same-budget fine-tuning on the target simulator.

Full numbers: [runs/maniskill_pickcube/results.md](runs/maniskill_pickcube/results.md) (container run), [runs/maniskill_pickcube_official/results.md](runs/maniskill_pickcube_official/results.md) (official demonstrations; side-by-side table in [comparison_vs_container_reference.md](runs/maniskill_pickcube_official/comparison_vs_container_reference.md)), the sim-to-sim reports [runs/sim2sim_pilot/latency_d1/results.md](runs/sim2sim_pilot/latency_d1/results.md) and [runs/sim2sim_pilot/actuation_s0.4/results.md](runs/sim2sim_pilot/actuation_s0.4/results.md), and the mid-point report (`report/midterm/`).

## Repository layout

```
configs/                 experiment configurations (ManiSkill PickCube container / official, sim-to-sim pilot, LIBERO Spatial)
src/chunk_trajectory_error/
  pipeline.py            shared stage logic used by scripts/00-08
  scheduler.py           chunk scheduler (H, e, delay r) shared by data collection and evaluation
  replay.py              snapshot / restore and branch replay
  error_net.py           history-conditioned trajectory-error critic
  objectives.py          BC loss, critic regression, frozen-critic policy objective
  policies/mlp_bc.py     MLP chunk policy
  backends/maniskill.py  ManiSkill adapter (libero.py is a stub)
  shifts.py              sim-to-sim target shifts (actuation, latency, contact) and the target backend
  sim2sim.py             target-episode logging and gap labels (source vs restored target)
  sim2sim_jobs.py        process-pool workers for Phase C rollouts and diagnostics
scripts/00-08            the Experiment 1 pipeline stages, run in order (see below)
scripts/sim2sim/         the sim-to-sim pilot stages 10-17 and one diagnostic (they reuse scripts/00, 02, 06, 08)
tests/                   unit tests (contracts, scheduler, reader, pipeline, dual networks, shifts, sim-to-sim labels)
envs/                    dependency lists; maniskill/requirements.lock.linux-py310.txt is the full tested freeze
data/                    PickCube-v1 demonstrations: raw and replay-converted (see Data)
runs/maniskill_pickcube/ outputs of the container run: checkpoints, branch labels, evaluations, results.md
runs/maniskill_pickcube_official/  the same pipeline on the GPU workstation with the official demonstrations
runs/sim2sim_pilot/      sim-to-sim pilot summaries (JSON / CSV / Markdown) and checkpoints
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

The GPU-workstation run and the sim-to-sim pilot used the official demonstrations. They are not committed; `configs/maniskill_pickcube_official.yaml` expects them next to, not in place of, the container data:

```bash
python -m mani_skill.utils.download_demo PickCube-v1        # prints where it saved the files
mkdir -p data/raw/maniskill_official data/processed/maniskill_official
cp <download dir>/PickCube-v1/motionplanning/trajectory.{h5,json} data/raw/maniskill_official/
python -m mani_skill.trajectory.replay_trajectory --traj-path data/raw/maniskill_official/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b physx_cpu
mv data/raw/maniskill_official/trajectory.state.pd_ee_delta_pos.physx_cpu.{h5,json} data/processed/maniskill_official/
cp runs/maniskill_pickcube_official/splits.json data/processed/maniskill_official/   # the locked 400 / 100 / 100 split
```

The file has 1000 episodes; the first 600 are used (`audit.num_episodes`). `00_audit_setup.py` records the file hashes in `manifest.json`; compare them with `runs/maniskill_pickcube_official/manifest.json`.

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

To rerun with the official demonstrations, set `C=configs/maniskill_pickcube_official.yaml`; outputs go to `runs/maniskill_pickcube_official/`. The GPU run used the same stages and optional variants plus one extra weight, `06 --variant error --weights 0.03` (evaluated as `*_w0.03`); its exact command sequence is in `runs/maniskill_pickcube_official/logs/_driver_progress.log`. That run read `configs/maniskill_pickcube.yaml` with the official files in `data/{raw,processed}/maniskill/`; `configs/maniskill_pickcube_official.yaml` differs from it only in the data paths, the output directory and the provenance note.

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

## Reproducing the sim-to-sim pilot (Experiment 2)

Specification and pre-registered amendments: [docs/SIM2SIM_PILOT_SPEC.md](docs/SIM2SIM_PILOT_SPEC.md). The source simulator is `configs/maniskill_pickcube_official.yaml` unchanged; the target is the same task with one shift applied. Prerequisites: the official demonstrations in `data/*/maniskill_official/` (see [Data](#data)) and the BC policy `runs/maniskill_pickcube_official/checkpoints/pi_bc/pi.pt` (committed). Run from the repository root; every stage writes under `runs/sim2sim_pilot/`. `--workers` sets the process-pool size (the recorded run used the defaults on a 24-core workstation).

```bash
P=configs/sim2sim_pilot.yaml              # latency d = 1
A=configs/sim2sim_pilot_actuation.yaml    # actuation (stiffness) s = 0.4

# Phase A: shift scan with pi_bc (gate 1) -> shift_scan.csv, shift_scan_episodes.parquet, manifest.json
python scripts/sim2sim/10_shift_scan.py --config $P

# Phase B (latency d = 1): target logs and gap labels, gap networks, gates 2-3
python scripts/sim2sim/11_collect_gap_labels.py --config $P --split train
python scripts/sim2sim/11_collect_gap_labels.py --config $P --split val
python scripts/sim2sim/11_collect_gap_labels.py --config $P --split train --shift identity   # noise floor, diagnostic
python scripts/sim2sim/12_train_gap.py --config $P             # checkpoints/gap_J.pt, gap_sqrtJ.pt, gap_report.json
python scripts/sim2sim/13_audit_gap.py --config $P             # gap_audit_sim2sim_only.{csv,json}
python scripts/sim2sim/13_audit_gap_demo.py --config $P        # gap_demo_diagnostic_sim2sim_only.{csv,json}, non-gating

# Phase C (latency d = 1): per-seed policies, selection on val seeds, final evaluation, report (gates 4-5)
for s in 17 18 19 20 21; do python scripts/sim2sim/14_train_policy_gap.py --config $P --stage train --seed $s; done
python scripts/sim2sim/14_train_policy_gap.py --config $P --stage select
python scripts/sim2sim/15_eval_transfer.py --config $P         # eval_episodes.parquet, eval_summary.json, paired_differences.json
python scripts/sim2sim/16_report_sim2sim.py --config $P        # latency_d1/results.md
python scripts/sim2sim/diag_consistency_latency_d1.py          # optional: phase_c/consistency_diagnosis.json

# Actuation s = 0.4: Phase B and Phase C back to back (Phase A is not rerun)
python scripts/sim2sim/11_collect_gap_labels.py --config $A --split train
python scripts/sim2sim/11_collect_gap_labels.py --config $A --split val
python scripts/sim2sim/12_train_gap.py --config $A
python scripts/sim2sim/13_audit_gap.py --config $A
python scripts/sim2sim/13_audit_gap_demo.py --config $A
for s in 17 18 19 20 21; do python scripts/sim2sim/14_train_policy_gap.py --config $A --stage train --seed $s; done
python scripts/sim2sim/14_train_policy_gap.py --config $A --stage select
python scripts/sim2sim/15_eval_transfer.py --config $A
python scripts/sim2sim/17_report_actuation.py --config $A --latency-config $P   # actuation_s0.4/results.md
```

`14 --stage train` reruns `scripts/02_prepare_policy.py` per seed and uses `train_one()` from `scripts/06_train_policy.py`, so the pilot's base policies and fine-tunes follow the Experiment 1 recipe exactly.

### Stored pilot outputs (`runs/sim2sim_pilot/`)

| Path | Content |
|---|---|
| `manifest.json`, `shift_scan.csv` | Phase A scan and the run manifest (file hashes, gate records) |
| `<shift>/collection_{train,val}.json` | Phase B target-log collection summaries (`identity/` = noise floor) |
| `<shift>/checkpoints/gap_{J,sqrtJ}.pt`, `gap_report.json` | gap networks and their held-out accuracy |
| `<shift>/gap_audit_sim2sim_only.*`, `gap_demo_diagnostic_sim2sim_only.*` | gate-3 gradient audits (sim-to-sim only) |
| `<shift>/phase_c/seed_*/` | per-seed `training.json`, `pi0_baseline.json` and policy checkpoints |
| `<shift>/phase_c/{selection,eval_summary,paired_differences}.json` | selection, final evaluation and paired bootstrap differences |
| `<shift>/results.md` | the Phase C report (stage `16` / `17`) |

`<shift>` is `latency_d1` or `actuation_s0.4`. The `results.md` files were regenerated from the full run data after the scripts moved to `scripts/sim2sim/`; apart from the script path and the renamed `runs/maniskill_pickcube_official` reference they are byte-identical to the originals.

## Not in this repository yet

- The official `PickCube-v1` demonstrations (raw 29 MB, replayed 41 MB). Download and replay them as described in [Data](#data).
- The raw sim-to-sim pilot data, kept on the GPU workstation: target logs (`<shift>/target_log_*.h5`, 32 MB), gap labels (`<shift>/labels_*.npz`, 18 MB), per-episode tables (`shift_scan_episodes.parquet`, `<shift>/phase_c/*.parquet`, 1.4 MB), stage logs and per-seed `base_config.yaml` files. Stages `12`-`17` read these files, so rerunning any stage after `11` (including the reports) needs them: regenerate them with stages `10`-`11` or copy them from the workstation.
- Figures for the pilot: the pilot scripts write none; the sim-to-sim figure of the mid-point report is built by `report/midterm/make_figures.py`.
- `runs/container_reference/`, the local copy of the container run that `runs/maniskill_pickcube_official/comparison_vs_container_reference.md` was compared against; the same run is committed as `runs/maniskill_pickcube/`.
