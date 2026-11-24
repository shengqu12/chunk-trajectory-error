# Running on Modal

`modal_run.py` runs the whole pipeline (Experiments 1, 1b, 1c) in Modal CPU containers, one
container per seed. The experiments are CPU-bound (physx_cpu, one environment, small networks),
so GPUs would not help; parallel seeds do. Rough cost: a 4-CPU container for about 4 hours per
seed, i.e. a few dollars per seed at Modal's CPU rates.

```bash
pip install modal
modal setup                                   # browser login, once
modal run modal_run.py --prepare              # official PickCube-v1 demos, downloaded + replayed into the cte-data volume (once, ~20 min)
modal run modal_run.py --seeds 17 --smoke     # end-to-end check with tiny budgets (~20-30 min, cents)
modal run modal_run.py --seeds 17,18,19,20,21 # five seeds in parallel; ~4 h wall
modal volume ls cte-runs                      # one directory per seed
modal volume get cte-runs maniskill_pickcube_seed17/results.md ./seed17_results.md
modal volume get cte-runs maniskill_pickcube_seed17 ./runs_seed17   # everything (checkpoints, parquet, logs)
```

Notes

- Seed 17 is the configuration of the container run in `runs/maniskill_pickcube/`; other seeds
  change network initialisation, batch order, flow-policy noise and branch perturbations. The
  episode split is created once by the first stage 01 and locked in the data volume, so all
  seeds share the same train/val/test episodes and the same 150 evaluation states.
- The image installs the tested lock file with a CPU-only torch wheel and Mesa's lavapipe Vulkan
  driver (SAPIEN needs a Vulkan ICD even without rendering). If SAPIEN fails to initialise, the
  log will say so in stage 00; the fallback is a GPU container with the NVIDIA ICD, see the
  ManiSkill installation notes.
- Logs stream to the terminal and to `<run_dir>/logs/modal_run.log` in the volume; each stage
  commits the volume, so a timed-out or interrupted run keeps the finished stages. Re-running a
  seed overwrites its directory.
- `--smoke` shrinks every budget (300 BC steps, 50 fine-tune steps, 1 anchor per episode,
  2 evaluation states); its numbers mean nothing, it only checks that every stage runs.
