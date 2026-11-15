# Sim-to-sim pilot: implementation spec

Status: proposal to be implemented. Nothing below has been run except the small shift check in section 2.
Place this file at `docs/SIM2SIM_PILOT_SPEC.md` in the Chunk Trajectory Error repository.

## 1. Question and setting

Does a learned estimate of the source-vs-target trajectory gap, used as a training penalty, give a
policy that transfers better to an environment with shifted physics?

- **Source environment**: nominal ManiSkill PickCube-v1, exactly as in `configs/maniskill_pickcube.yaml`
  (`physx_cpu`, one env, `pd_ee_delta_pos`, state observations). Demonstrations and all training use it.
- **Target environment**: the same task with one physics shift. It stands in for a real robot. Training
  never uses its dynamics directly; only a fixed budget of logged rollouts is allowed.
- **Hardware-faithful rule**: during data collection the target environment is only *run forward and
  logged*. It is never restored to an earlier state. Only the source simulator is restored. Target
  restores are allowed solely in diagnostics that are explicitly marked "sim-to-sim only".

Chunk settings stay H = 20, e = 5, L = 15, K = 10, delay r = 0. Reuse `ChunkScheduler`,
`PolicyBundle`, `ErrorBundle`, `ManiSkillBackend`, `load_demoset`, `closed_loop_episode`-style loops,
`bootstrap_ci` and the report helpers. Write outputs under `runs/sim2sim_pilot/`. Do not modify or
overwrite anything under `runs/maniskill_pickcube/`.

## 2. Physics shifts

Three families. Apply a shift **once**, after the first `reset` of a freshly created environment,
relative to the nominal values read at that moment. PickCube does not rebuild the scene on later
resets, so the change persists; applying it again after every reset compounds it (this mistake was
made once in the preliminary check and inflated the effect).

| Family | How to apply (verified to run on mani_skill 3.0.1 / sapien 3.0.3) |
|---|---|
| Actuation | For the 7 arm joints in `env.unwrapped.agent.robot.get_active_joints()[:7]`: `joint.set_drive_properties(stiffness * s, damping * s, force_limit=joint.force_limit)`. Nominal stiffness is 1000, damping 100. |
| Latency | Wrapper around `step`: the command sent at step k is applied at step k + d. For the first d steps apply zero translation with the current gripper command. The pending commands are part of the target state and must be logged. |
| Contact | For each body in `env.unwrapped.cube._bodies`: `body.set_mass(body.get_mass() * m)`; for each collision shape, scale static and dynamic friction of `shape.get_physical_material()` by f. Nominal cube mass is 0.064 kg. |

Assert after the next reset that the changed values are still in place, and record them in the manifest.

Preliminary check (GPU-less container, its own `pi_bc` checkpoint, 60 test initial states, r = 0, one
run, so differences of a few episodes mean nothing):

| Target | Success |
|---|---:|
| Nominal | 27 / 60 |
| Actuation 0.8 | 27 / 60 |
| Actuation 0.6 | 23 / 60 |
| Latency 1 step | 11 / 60 |
| Contact f = 0.5, m = 2 | 28 / 60 |
| Contact f = 0.7, m = 1.5 | 29 / 60 |

Reading: one step of latency clearly hurts. The actuation and contact levels first proposed create
little or no gap, so stronger levels must be scanned before they are used.

## 3. Phases, gates and stop points

Stop for review after each phase. A failed gate is a result: report it and stop.

### Phase A — find shifts that create a gap (gate 1)

Script: `scripts/10_shift_scan.py`.

- Policy: existing `runs/maniskill_pickcube/checkpoints/pi_bc/pi.pt`. No training.
- Initial states: the same 150 used by `07_eval_closed_loop.py` (100 test episodes + 50 fresh seeds).
- Conditions: nominal; actuation s in {0.8, 0.6, 0.4, 0.3}; latency d in {1, 2};
  contact (f, m) in {(0.5, 2), (0.3, 3), (0.2, 4)}.
- Output: `shift_scan.csv` with successes, Wilson 95% interval, paired difference to nominal with a
  bootstrap 95% interval, mean episode steps.
- **Gate 1**: per family, pick the mildest level whose paired success drop is at least 15 points with
  an interval that excludes zero. A family with no such level is dropped. If no family passes, stop.

**STOP. Report the table and the chosen level per family.**

### Phase B — gap labels, gap network, diagnostics (gates 2 and 3)

Run for one chosen shift first (the one with the clearest gap); repeat for the others only after review.

Scripts: `11_collect_gap_labels.py`, `12_train_gap.py`, `13_audit_gap.py`.

**Rollouts in the target (forward only).**
- Policy: `pi_bc`. Initial states: seeds of the *train* split episodes for training labels, *val* split
  for validation labels, *test* split is not used here.
- Exploration: in half of the episodes add a bounded low-frequency perturbation to each executed chunk
  (`low_frequency_perturbation`, amplitude range from the config, gripper untouched), so that the data
  contains a range of actions per state. No target restores.
- Log every step: full simulator state dict before the action, state observation, commanded action,
  end-effector and cube position after the action, and for the latency shift the pending commands.
- Budget: 20,000 target steps for training labels. Validation uses a separate 5,000 steps. Record the
  exact counts; these are the "real-data" budgets all methods must respect.

**Labels.** For every step t with a full history (t >= L) and K further logged steps: restore the
*source* simulator to the logged state at t, execute the K logged commanded actions, and compute

    J_gap = mean over j = 1..K of || p_target[t+j] - p_source[t+j] ||^2      (end effector, m^2)

Store the cube deviation the same way as a second value. Store per sample: episode, t, history
(relative end-effector positions over the previous L logged steps), robot state, the K actions, both
labels. Keep whole episodes in one split.

**Gap network.** Reuse `ErrorBundle` and the `04` training loop. Train both targets (`J` and `sqrtJ`).
Report on held-out validation episodes: MAE against the constant-mean predictor, Pearson, rank
correlation, and mean predicted value on the tenth of samples with the smallest true gap.
- **Gate 2**: MAE below the constant predictor and global rank correlation above 0.5 on validation.

**Diagnostics (sim-to-sim only; these restore the target).** On 300 validation samples of the policy's
own unperturbed chunks: step size chosen from {0.02, 0.05, 0.1, 0.2} on half of them, tested on the
other half; compare a step along the negative gradient of G with an equal-norm random step by
executing both in source and target from the logged state and recomputing the true gap.
- **Gate 3**: the gradient step lowers the true gap, and lowers it more than the random step, with
  episode-clustered bootstrap 95% intervals below zero.

**STOP. Report gates 2 and 3 with the tables, plus the label distribution (median, 90th percentile).**

### Phase C — policies and transfer (gates 4 and 5)

Scripts: `14_train_policy_gap.py`, `15_eval_transfer.py`, `16_report_sim2sim.py`.

Five seeds (17, 18, 19, 20, 21). For each seed train a base policy with `02`'s recipe on the source
demonstrations, then fine-tune from it with the `06` recipe (same steps, learning rate, batches):

| Method | Fine-tuning data and loss | Target data used |
|---|---|---|
| Zero-shot | Source demonstrations, imitation only | None |
| Gap penalty | Source demonstrations, imitation + lambda * G on the policy chunk (G frozen) | Training labels only |
| Same-budget fine-tune | Source demonstrations plus the successful episodes among the 20,000 logged target steps, imitation only | The same logged steps |

G is trained once per shift from the seed-17 base policy's rollouts and reused for all seeds; say so in
the report. Lambda grid: {0.001, 0.003, 0.01, 0.03, 0.1}. Select lambda per seed by target success on
30 validation-split initial states; the fine-tune baseline selects its checkpoint the same way. Count
and report these validation steps separately.

Evaluation: 150 initial states, r = 0, in both target and source. Record success, episode steps, mean
end-effector step per control step. Also measure the true gap of each policy's own chunks on held-out
logged states (sim-to-sim only).

- **Gate 4**: gap penalty beats zero-shot in target success; paired over initial states and seeds,
  95% interval above zero; no drop in source success and no slowdown that explains the change.
- **Gate 5**: gap penalty is at least as good as the same-budget fine-tune (interval of the difference
  not below zero).

Report (`runs/sim2sim_pilot/results.md`): one table per gate, per-seed numbers, pooled paired
differences, and a verdict line per gate. State plainly which gates failed.

### Phase D — optional, only after review

Domain-randomization and target-oracle baselines. Both need demonstrations generated by the motion
planner under shifted physics (`mplib==0.1.1` with `numpy<2`). Do not start without confirmation.

## 4. Rules

- Do not change the replay tolerance, gate thresholds, lambda grid or selection rules after seeing results.
- Never tune on the 150 evaluation states.
- Log skipped or aborted rollouts with counts; do not drop them silently.
- Record versions, seeds, shift values, budgets and file hashes in `runs/sim2sim_pilot/manifest.json`.
- Add unit tests for: shift applied once and persisting across resets; the latency wrapper's command
  ordering; label alignment (a zero shift must give J_gap = 0 exactly on every sample).
- Run with `PYTHONPATH` and `LD_LIBRARY_PATH` unset.
