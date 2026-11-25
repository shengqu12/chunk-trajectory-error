# Five-seed findings

Five seeds (17-21) run on Modal CPU containers (Ubuntu 24.04, Python 3.11.17,
PyTorch 2.14.1+cpu, ManiSkill 3.0.1, SAPIEN 3.0.3, physx_cpu) with locally
generated PickCube-v1 demonstrations. All seeds share the same 400/100/100
episode split and 150 evaluation initial states; each seed changes network
initialisation, batch order, flow-policy noise and branch perturbations.

Intervals below are 95 % t-intervals across seeds; paired differences are
computed within each seed on the shared initial states and then pooled across
seeds.

---

## 1. Experiment 1: critic as regulariser (MLP chunk policy)

**Across-seed result.** `error` (MLP + history error net, selected lambda) vs
`bc` (lambda = 0 fine-tune): pooled paired success difference
**-6.6 pp [-18.5, +5.3]** (4 of 5 seeds negative, 1 positive).

The 95 % CI includes zero, so the five-seed data do not establish that the
critic regulariser hurts the MLP policy at the conventional level. However 4
of 5 seeds are negative and the point estimate is -6.6 pp.

Per-seed paired success difference (pp, pooled over delays):
| Seed | 17 | 18 | 19 | 20 | 21 |
|---|---:|---:|---:|---:|---:|
| error vs bc | +2.2 | -8.4 | -3.3 | -22.2 | -1.1 |

The smoothness penalty (`smooth`) is indistinguishable from `bc`:
-0.4 pp [-6.7, +5.9] (4-/1+).

---

## 2. Experiment 1b: critic as regulariser (flow-matching chunk policy)

**Across-seed result.** `error_flow` (flow + history error net, selected
lambda) vs `bc_flow`: pooled paired success difference
**-8.6 pp [-14.8, -2.4]** (5 of 5 seeds negative). The CI excludes zero.

Per-seed paired success difference (pp, pooled over delays):
| Seed | 17 | 18 | 19 | 20 | 21 |
|---|---:|---:|---:|---:|---:|
| error_flow vs bc_flow | -8.4 | -14.7 | -12.2 | -2.4 | -5.1 |

The smoothness penalty (`smooth_flow`) is indistinguishable from `bc_flow`:
+0.7 pp [-0.0, +1.4] (1-/4+).

---

## 3. Experiment 1c: critic as a best-of-N selector

**Across-seed results** (all vs `bc_flow`):

| Policy | pooled diff (pp) [95 % CI] | seeds neg/pos |
|---|---:|---:|
| select_best8_flow | **-7.4 [-13.5, -1.3]** | 5-/0+ |
| select_best32_flow | **-15.9 [-22.9, -8.9]** | 5-/0+ |
| select_worst8_flow | +2.3 [-3.6, +8.1] | 3-/2+ |

Per-seed paired success difference (pp, pooled over delays):
| Seed | 17 | 18 | 19 | 20 | 21 |
|---|---:|---:|---:|---:|---:|
| select_best8_flow | -1.8 | -12.9 | -5.6 | -12.2 | -4.7 |
| select_best32_flow | -8.9 | -21.6 | -15.3 | -21.6 | -12.2 |
| select_worst8_flow | -0.7 | +4.0 | -0.4 | +9.8 | -1.3 |

Best-of-8 loses 7.4 pp across five seeds (CI excludes zero); best-of-32 loses
15.9 pp. The control (worst-of-8) is indistinguishable from the reference.

---

## 4. Gates per seed

| Seed | critic test Pearson | stage-05 gradient gate (policy ΔJ %) | G2 lr 1e-5 (ΔJ % at 1 / 5 / 25 updates), gate | G2 lr 3e-4 (ΔJ % at 10 updates) | pi0 val r0 | pi0_flow val r0 |
|---|---:|---:|---|---:|---:|---:|
| 17 | 0.974 | -22 %, pass | -0 / +8 / +30, fail | +1151 % | 6/30 | 29/30 |
| 18 | 0.974 | -20 %, pass | +1 / +9 / +30, fail | +575 % | 8/30 | 26/30 |
| 19 | 0.973 | -31 %, pass | -1 / +3 / +9, fail | +294 % | 4/30 | 29/30 |
| 20 | 0.958 | -20 %, pass | -0 / +5 / +15, fail | +908 % | 3/30 | 26/30 |
| 21 | 0.970 | -29 %, pass | -1 / +4 / +38, fail | +1468 % | 8/30 | 27/30 |

- The critic fits held-out labels consistently well across seeds (Pearson 0.958-0.974).
- The MLP gradient gate (stage 05) passes on all 5 seeds (ΔJ -20 % to -31 %).
- The flow gradient gate G2 fails on all 5 seeds: at lr 1e-5 the simulator J
  rises after 5 updates in every seed; at lr 3e-4 it explodes (+294 % to
  +1468 % after 10 updates).
- The MLP pi0 is weak in closed loop on all seeds (3-8 of 30 val states at
  r = 0); the flow pi0 is strong on all seeds (26-29 of 30).

---

## 5. Selection audit (validation anchors, 8 samples per anchor)

| Seed | within-anchor Spearman | J picked / random / rejected / oracle (x1e-5) | picked - random (%) | picked is oracle | mean rank |
|---|---:|---:|---:|---:|---:|
| 17 | 0.47 | 2.56 / 4.30 / 7.05 / 1.80 | -40 | 37 % | 3.1 / 8 |
| 18 | 0.43 | 2.47 / 4.54 / 8.12 / 1.77 | -46 | 28 % | 3.4 / 8 |
| 19 | 0.47 | 2.11 / 3.98 / 6.67 / 1.55 | -47 | 37 % | 3.1 / 8 |
| 20 | 0.38 | 2.98 / 4.63 / 7.51 / 2.17 | -36 | 30 % | 3.4 / 8 |
| 21 | 0.40 | 2.34 / 4.13 / 7.29 / 1.64 | -43 | 30 % | 3.5 / 8 |
| **mean [CI]** | **0.43 [0.38, 0.48]** | | **-42 [-48, -37]** | **33 [27, 38] %** | **3.3 [3.0, 3.5]** |

The ranking is consistent across seeds: the critic's pick lowers J by 42 %
relative to a random pick, and is the oracle 33 % of the time (chance 12.5 %).

---

## 6. Discrepancies between the five-seed aggregate and the single-run numbers

The single-run numbers are from docs/FLOW_POLICY_RESULTS.md and
docs/SELECTION_RESULTS.md (the seed-17 container run in `runs/maniskill_pickcube/`,
Python 3.10). The Modal runs use Python 3.11 and re-run the pipeline from
scratch per seed; even seed 17 is a fresh run and need not match.

### 6a. Sign differences

| Policy | single-run pooled diff (pp) | five-seed pooled diff (pp) [CI] | sign change? |
|---|---:|---:|---|
| select_best8_flow | +0.4 [-2.0, +2.9] | -7.4 [-13.5, -1.3] | **yes** |

The single-run found best-of-8 indistinguishable from the reference (+0.4 pp,
CI includes zero); all five seeds find it negative (-1.8 to -12.9 pp). The
single-run result was within the noise floor; the five-seed data show a
consistent loss of about 7 pp.

### 6b. Magnitude differences > 5 pp (same sign)

| Policy | single-run pooled diff (pp) | five-seed pooled diff (pp) [CI] | gap |
|---|---:|---:|---:|
| error_flow | -32.4 [-38.4, -26.4] | -8.6 [-14.8, -2.4] | 23.8 pp |
| select_best32_flow | -5.1 [-8.7, -1.6] | -15.9 [-22.9, -8.9] | 10.8 pp |
| error (MLP) | -8.4 [-13.6, -3.3] | -6.6 [-18.5, +5.3] | 1.8 pp (but CI now includes 0) |

**error_flow** is the largest discrepancy. The single-run result was dominated
by the r = 0 / r = 2 delays (success at 77 / 59 / 45 %, a loss of 13 / 33 /
51 pp); across five seeds the losses are shallower (-14.7 / -4.1 / -6.9 pp at
r = 0 / 2 / 4). A likely contributor is that different seeds select different
lambda values: the original run selected lambda = 0.03, while the five Modal
seeds selected lambda = 0.01 (seeds 17, 18), 0.03 (seeds 19, 20) and 0.003
(seed 21). A smaller lambda gives a weaker critic term and a smaller success
loss.

**select_best32_flow** is worse across five seeds (-15.9 pp) than in the
single run (-5.1 pp); every seed agrees in sign.

**error (MLP)** changes little in point estimate (-8.4 to -6.6 pp) but the
across-seed CI widens to include zero because seed 17 is positive (+2.2 pp)
and seed 20 is very negative (-22.2 pp). The sign is not established at five
seeds.

### 6c. Quantities that agree well

| Quantity | single-run | five-seed |
|---|---|---|
| pi0_flow vs bc_flow pooled | +2.0 pp [-1.3, +5.6] | +6.0 pp [-4.3, +16.4] |
| smooth_flow vs bc_flow pooled | +0.4 pp [-2.0, +2.9] | +0.7 pp [-0.0, +1.4] |
| select_worst8_flow pooled | +0.0 pp [-2.2, +2.2] | +2.3 pp [-3.6, +8.1] |
| selection audit Spearman | 0.47 | 0.43 [0.38, 0.48] |
| selection audit picked-random | -43 % | -42 [-48, -37] % |
| selection audit picked is oracle | 36 % | 33 [27, 38] % |
| critic test Pearson (seed 17) | 0.972 | 0.974 |

---

## 7. Additional observations

**Seed 20 is an outlier for bc_flow.** Its bc_flow success (56 / 71 / 87 %)
is 15-30 pp below the other four seeds (81-91 / 92-99 / 95-99 %). This pulls
down the mean and widens the CI for every flow policy. All flow variants on
seed 20 are low, so the within-seed paired comparisons remain valid, but the
seed-level variance is dominated by the base policy quality.

**error_flow_w0.3 and error_w0.03 (fixed high lambda) destroy the policy on
all seeds.** error_flow_w0.3 pooled: 2.4 % [CI includes zero]; error_w0.03
pooled: 8.5 % (the fixed lambda = 0.03 on the MLP corresponds to the
selected lambda = 0.03 on the flow policy, but the MLP is far more sensitive
to it). Every seed agrees.

**The flow policy is a far stronger base than the MLP on all seeds.** pi0_flow
pooled success: 94.7 % [91.8, 97.5]; MLP pi0 pooled success: 36.2 % [22.3,
50.1]. This holds on all five seeds.

---

## 8. Videos

Side-by-side closed-loop videos for seed 17 (8 policy pairs x 3 initial states
= 24 mp4 files) were rendered on Modal with software Vulkan (lavapipe). The
videos are in `report/final/videos/` if the total is under 100 MB; otherwise
they are on the Modal volume `cte-runs` at
`maniskill_pickcube_seed17/videos/`.

---

## Summary of the five-seed pooled paired differences

| Policy | pooled diff vs same-class reference (pp) [95 % CI] | seeds neg/pos |
|---|---:|---:|
| error (MLP, selected lambda) | -6.6 [-18.5, +5.3] | 4-/1+ |
| error_flow (flow, selected lambda) | -8.6 [-14.8, -2.4] | 5-/0+ |
| select_best8_flow | -7.4 [-13.5, -1.3] | 5-/0+ |
| select_best32_flow | -15.9 [-22.9, -8.9] | 5-/0+ |
| select_worst8_flow | +2.3 [-3.6, +8.1] | 3-/2+ |

The critic-as-regulariser result is negative for both policy classes: the flow
result is established (CI excludes zero, 5/5 seeds agree); the MLP result is
suggestive but not established (CI includes zero, 4/5 seeds agree). The
critic-as-selector result is negative at five seeds: best-of-8 and best-of-32
both lose success, with every seed negative. The single-run finding that
best-of-8 was neutral (+0.4 pp) does not replicate.
