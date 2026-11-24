# Experiment 1c: the critic as a selector instead of a training signal

Run on 2026-11-24 on the CPU-only container, same data, critic, labels and evaluation states as
Experiments 1 and 1b. Numbers are also in `runs/maniskill_pickcube/results.md` (rows "critic
selects ...") and in `runs/maniskill_pickcube/selection_audit.json`.

## Question

The critic ranks held-out candidate chunks well (Pearson 0.97, within-anchor Spearman 0.88 on
the labelled set) but fails as a training signal in both policy classes. Does it help if it is
only allowed to *choose*? At every decision the `bc_flow` policy samples N chunks, the frozen
critic scores each tail `A[r : r + K]` with the end-effector history (rebuilt online from the
state observation), the robot state and the delay, and the lowest-scoring chunk is executed.
No parameter changes. Controls: execute the *highest*-scoring chunk (`worst`), and N = 32.

## 1. Selection audit on validation anchors (`scripts/09_audit_selection.py`)

300 anchors x 8 samples from `bc_flow`, every sample rolled out in the simulator from the same
restored state, expert reference from the same anchor:

| | mean true J (m^2) | vs random pick |
|---|---:|---:|
| oracle pick (lowest true J) | 1.80e-5 | |
| **critic pick** (lowest predicted E) | **2.47e-5** | **-1.83e-5 [-2.48e-5, -1.28e-5] (-43 %)** |
| random pick (mean of the 8) | 4.30e-5 | reference |
| critic reject (highest predicted E) | 7.14e-5 | +2.84e-5 [+2.05e-5, +3.82e-5] (+66 %) |
| worst sample | 7.96e-5 | |

Within-anchor Spearman between predicted and true cost among the policy's own 8 samples:
0.47 [0.42, 0.51] (0.88 on the labelled candidates, which include large perturbations). The
critic's pick is the oracle 36 % of the time (chance 12.5 %) and in the better half 70 % of the
time; mean rank 3.1 of 8. The spread it chooses among is real but small: 3.8 mm between the
best and worst sample in sqrt(J).

So at demonstration states the critic does tell which of the policy's own chunks stays closer to
the expert, and by a wide margin.

## 2. Closed loop (150 initial states, paired against `bc_flow`)

| Policy | test-anchor J (m^2) | handover jump | r = 0 | r = 2 | r = 4 | pooled success difference |
|---|---:|---:|---:|---:|---:|---:|
| bc_flow (reference) | 3.46e-5 | – | 135 (90 %) | 138 (92 %) | 144 (96 %) | – |
| critic selects best of 8 | 1.99e-5 (-42 %) | -6 to -7 % | 133 (89 %) | 142 (95 %) | 144 (96 %) | +0.4 pp [-2.0, +2.9] |
| critic selects best of 32 | 1.62e-5 (-53 %) | -1 to -7 % | 125 (83 %) | 132 (88 %) | 137 (91 %) | **-5.1 pp [-8.7, -1.6]**, p = 0.009 |
| critic selects worst of 8 | 5.77e-5 (+67 %) | +5 to +10 % | 131 (87 %) | 138 (92 %) | 148 (99 %) | +0.0 pp [-2.2, +2.2] |

Latency per decision on one CPU thread: 2.0 ms (bc_flow), 3.6 ms (best of 8), 6.3 ms (best of 32).

## Reading

1. The ranking is real and it transfers to the test anchors: choosing by the critic lowers the
   simulator-recomputed J by 42-53 % and makes handovers 6-7 % smoother, and choosing against it
   raises J by 67 %. This is the strongest positive statement the data support about the critic.
2. It makes no difference to task success. Best-of-8 and worst-of-8 are both indistinguishable
   from the reference, on the same 150 initial states, at all three delays. What the critic
   measures (10-step deviation from the expert at demonstration states) is not what decides
   whether the cube ends up at the goal: the base policy is already at 90-96 % and its failures
   are not of the kind the label describes.
3. With more candidates the selector starts to exploit the critic, as the regulariser did:
   best-of-32 lowers J further and loses 5 pp of success. The same mechanism appears here
   without any gradient step, so it is a property of the critic's errors, not of back-propagation.
4. Online, the critic is further from its training distribution than at anchors: the mean
   predicted cost of the chosen chunk is 0.65 (normalised) in closed loop against 0.33 for the
   same policy at demonstration anchors, because the history and robot state come from the
   policy's own trajectory. Whether its ranking is still correct there cannot be checked with
   the label as defined, since J needs an expert continuation from the same state and none
   exists off the demonstrations. That is the conceptual limit of the label, not a bug.

Together with Experiments 1 and 1b this answers the question the instructor raised: the critic
can tell which chunk stays closer to the reference, at the states where the reference exists,
and that is precisely why it does not improve success anywhere else.

## Caveats

One seed, one critic (trained on labels generated with the MLP pi0 as the policy candidate
source), locally generated demonstrations, CPU container. `select_worst8_flow` is excluded from
the acceptance table on purpose (it is a control). `select_first8_flow` is available as a
sanity check that sampling 8 and taking the first equals `bc_flow` up to the RNG stream; it was
not run.
