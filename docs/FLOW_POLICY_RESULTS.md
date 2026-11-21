# Experiment 1b: does the conclusion depend on the policy class?

Run on 2026-11-20 on the CPU-only container (same demonstrations, critic, branch labels and
evaluation states as the Experiment 1 run in `runs/maniskill_pickcube/`). Everything below is
also in `runs/maniskill_pickcube/results.md` (rows marked "Flow").

## What changed

Only the policy class. `FlowChunkPolicy` learns a velocity field over the 20 x 4 action chunk
(conditional flow matching; MLP 3 x 512, inputs chunk + 42-D state + time features) and samples a
chunk with 8 Euler steps. The sampler is ordinary PyTorch, so the frozen-critic term
`lambda * E(tail of the sampled chunk)` back-propagates to the parameters through every step.
The critic, the labels, the fine-tuning budget (5000 steps, lr 3e-4, batch 256), the
`lambda`/`beta` grids, the selection rule (simulator-recomputed J on validation) and the 150
evaluation states are unchanged.

`pi0_flow` needed a longer schedule than the MLP: 20k constant-lr steps gave val J 9.1 mm and a
sampled-chunk MSE 2.2x the MLP's; 60k cosine-decay steps with the 3 x 512 field gave val J 4.2 mm
(MLP: 4.6 mm) and MSE 1.9x. Both settings are recorded in `configs/maniskill_pickcube.yaml`
(`models.policy_flow`, `bc.flow`).

## Gates

**G1 (base policy comparable to the MLP): pass.** Val J 3.25e-5 vs 2.94e-5 m^2; val closed-loop at
r = 0: 29/30 vs 5/30 for the MLP pi0.

**G2 (critic gradient through the sampler lowers the true J): fail.** `scripts/05b_audit_flow_gradient.py`
takes Adam steps on the critic term alone from `pi0_flow` and recomputes J in the simulator on
120 validation anchors; a control run takes steps of the same size in random directions.

| lr | updates | predicted E (normalised) | mean J (m^2) | paired dJ vs pi0_flow [95 % CI] |
|---|---:|---:|---:|---:|
| – | 0 | 0.220 (train batch) / 0.247 | 3.19e-5 | – |
| 1e-5 | 1 | 0.236 | 3.18e-5 | -1.8e-7 [-8.3e-7, +3.7e-7] |
| 1e-5 | 2 | 0.226 | 3.18e-5 | -1.3e-7 [-1.4e-6, +9.0e-7] |
| 1e-5 | 5 | 0.202 | 3.33e-5 | +1.3e-6 [-1.0e-6, +3.4e-6] |
| 1e-5 | 10 | 0.176 | 3.68e-5 | +4.8e-6 [+0.9e-6, +8.2e-6] |
| 1e-5 | 25 | 0.124 | 4.01e-5 | +8.2e-6 [+2.8e-6, +12.9e-6] |
| 3e-4 | 10 | 0.023 | 3.43e-4 | +3.1e-4 [+2.9e-4, +3.3e-4] |
| 3e-4 | 200 | 0.0006 | 2.67e-3 | +2.6e-3 [+2.6e-3, +2.7e-3] |

Random directions of the same step size leave J unchanged (|dJ| < 1e-7 at lr 1e-5; +1.3e-5 after
200 updates at 3e-4). The parameter gradient is finite and non-zero on all 8 parameter tensors and
the critic receives none. So the gradient path works; the direction it gives does not lower the
simulator J at any step size, and the predicted E falls the whole time.

**G3 (closed loop, 150 initial states, paired against `bc_flow`):**

| Policy | test-anchor J (m^2) | r = 0 | r = 2 | r = 4 | pooled success difference |
|---|---:|---:|---:|---:|---:|
| pi0_flow | 2.99e-5 | 134 (89 %) | 144 (96 %) | 148 (99 %) | +2.0 pp [-1.3, +5.6] |
| bc_flow (lambda = 0) | 3.46e-5 | 135 (90 %) | 138 (92 %) | 144 (96 %) | reference |
| error_flow (selected lambda = 0.03) | 2.93e-5 (-15 %, CI excludes 0) | 115 (77 %) | 88 (59 %) | 68 (45 %) | -32.4 pp [-38.4, -26.4] |
| smooth_flow (beta = 0.1) | 3.46e-5 | 131 (87 %) | 143 (95 %) | 145 (97 %) | +0.4 pp [-2.0, +2.9] |
| error_flow, fixed lambda = 0.3 | 4.58e-4 | 1 (1 %) | 0 | 0 | -57.1 pp |

Validation sweep of `error_flow` (mean J, m^2): 0.001: 4.10e-5, 0.003: 3.91e-5, 0.01: 3.72e-5,
**0.03: 3.61e-5**, 0.1: 4.28e-5, 0.3: 4.57e-4, 1: 3.70e-3 (bc_flow: 4.33e-5).

## Reading

1. The negative result holds for the second policy class. Unlike the MLP, the critic term here
   does lower the quantity it approximates (test-anchor J, -15 %), and closed-loop success
   still falls by 13 / 33 / 51 pp. Expert deviation over K = 10 steps at demonstration states is
   not what decides task success: `error_flow` chunks are rougher (mean step-to-step translation
   jump 0.035 vs 0.029 normalised units on validation; P95 handover jump 0.16-0.65 vs 0.07-0.11
   in closed loop), episodes run longer (114-153 steps vs 89), and the damage grows with the
   injected delay.
2. The policy exploits the critic as soon as the term is active: on validation samples the
   predicted E drops from 0.33 (bc_flow) to 0.11 (error_flow) and 0.014 (lambda = 0.3) while the
   open-loop xyz error rises (0.00072 -> 0.00084 -> 0.0147). G2 shows the same thing from the
   first update.
3. The smoothness penalty has no effect on the flow policy (its chunks are already smooth; the
   three betas give the same J to three digits), unlike the MLP case where it was the best variant.
4. Side finding: `pi0_flow` is far stronger in closed loop than the MLP `pi0` (89 % vs 14 % at
   r = 0) at a slightly *higher* open-loop MSE (xyz 0.00062 vs 0.00045). The gripper channel is
   not the reason (both ~98 % saturated). The most likely reason is that sampling from a
   distribution instead of regressing its mean avoids averaged actions at multimodal states and
   lets the policy escape stalls; this was not investigated further.

## Caveats

- One seed, one run per variant, CPU container, locally generated demonstrations (not the
  official file). The GPU workstation should repeat this with the official demonstrations.
- The critic and branch labels were produced with the MLP `pi0` as the policy candidate source;
  they were not regenerated for the flow policy.
- The flow fine-tune at lr 3e-4 slightly degrades the cosine-annealed `pi0_flow` (val J 3.25e-5
  -> 4.33e-5 for `bc_flow`). All flow variants share this recipe, so the within-class
  comparison is fair, and `pi0_flow` is evaluated alongside them.
- Sampled-chunk latency with 8 Euler steps is in `eval_latency_flow_*.json` (CPU, one thread).
