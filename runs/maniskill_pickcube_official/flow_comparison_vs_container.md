# Experiments 1b and 1c on the GPU workstation vs the container

Run on 2026-11-24 with `configs/maniskill_pickcube.yaml` (snapshot: `config_flow_gpu.yaml`;
only `experiment.output_dir` and the data paths point at the workstation run). The Experiment 1
critic `error_round0_history.pt` and the branch labels were reused, not regenerated. Container
numbers are from `runs/maniskill_pickcube/` on `flow-policy` (8605b99).

Data note: the official `trajectory.h5` has 1000 episodes, but the splits use episodes 0-599, and
those are the same trajectories as the container's locally generated file (same seeds; actions
equal to within 1.8e-6). This run therefore repeats the container experiment on different
hardware; it does not test different demonstrations.

## Gates

- G1 pass: pi0_flow val J 3.250e-5 vs pi0 2.583e-5 m^2 (ratio 1.26); val r = 0 success 29/30 vs 5/30.
  pi0_flow matches the container's to six significant digits.
- G2 fail at lr 3e-4 and at lr 1e-5 (`gate.pass = false`, no passing checkpoint in either file).
- G3: see the tables below. Selected lambda is 0.01 here (container 0.03); sweep
  0.001: 4.11e-5, 0.003: 3.93e-5, 0.01: 3.72e-5, 0.03: 3.91e-5, 0.1: 5.65e-5, 0.3: 1.61e-3, 1: 7.93e-3.

## Selection audit (val anchors, 8 samples)

| | GPU | container |
|---|---:|---:|
| picked - random J (m^2) | -1.70e-5 [-2.37e-5, -1.09e-5] | -1.83e-5 [-2.48e-5, -1.28e-5] |
| picked is oracle | 0.34 | 0.36 |
| within-anchor Spearman | 0.46 [0.40, 0.51] | 0.47 [0.42, 0.51] |

## Results.md rows, GPU | container

Columns: policy | GPU | container. Sign changes of a paired difference are marked.

## Section 5 (paired dJ vs bc_flow)
| Flow: pi0 (initial flow matching) | 2.985e-05 [2.20e-05, 3.95e-05] | 2.985e-05 [2.20e-05, 3.95e-05] | -4.740e-06 [-6.15e-06, -3.43e-06] (-13.7%) | -4.731e-06 [-6.13e-06, -3.43e-06] (-13.7%) |
| Flow: fine-tune (lambda = 0) | 3.459e-05 [2.65e-05, 4.46e-05] | 3.459e-05 [2.65e-05, 4.45e-05] | – (–) | – (–) |
| Flow + error net (history), selected lambda | 2.957e-05 [2.37e-05, 3.63e-05] | 2.926e-05 [2.40e-05, 3.53e-05] | -5.020e-06 [-8.87e-06, -2.09e-06] (-14.5%) | -5.326e-06 [-1.05e-05, -1.25e-06] (-15.4%) |
| Flow + action smoothness, selected beta | 3.460e-05 [2.65e-05, 4.46e-05] | 3.461e-05 [2.65e-05, 4.46e-05] | 9.428e-09 [-1.36e-08, 3.17e-08] (+0.0%) | 2.247e-08 [-5.51e-10, 4.83e-08] (+0.1%) |
| Flow + error net (history), fixed lambda = 0.3 | 1.588e-03 [1.42e-03, 1.76e-03] | 4.584e-04 [4.13e-04, 5.05e-04] | 1.569e-03 [1.40e-03, 1.74e-03] (+8470.5%) | 4.400e-04 [3.96e-04, 4.84e-04] (+2386.4%) |
| Flow (lambda = 0) + critic selects best of 8 chunks | 1.775e-05 [1.43e-05, 2.26e-05] | 1.991e-05 [1.52e-05, 2.61e-05] | -1.684e-05 [-2.33e-05, -1.13e-05] (-48.7%) | -1.468e-05 [-2.22e-05, -7.42e-06] (-42.4%) |
| Flow (lambda = 0) + critic selects best of 32 chunks | 1.464e-05 [1.23e-05, 1.74e-05] | 1.616e-05 [1.31e-05, 2.02e-05] | -1.995e-05 [-2.84e-05, -1.28e-05] (-57.7%) | -1.843e-05 [-2.75e-05, -1.06e-05] (-53.3%) |
| Flow (lambda = 0) + critic selects worst of 8 chunks | 5.726e-05 [4.49e-05, 7.21e-05] | 5.765e-05 [4.52e-05, 7.25e-05] | 2.266e-05 [1.63e-05, 2.94e-05] (+65.5%) | 2.307e-05 [1.66e-05, 3.01e-05] (+66.7%) |

## Section 6 success
| Flow: pi0 (initial flow matching) | 134/150 (89%) / 144/150 (96%) / 148/150 (99%) | 134/150 (89%) / 144/150 (96%) / 148/150 (99%) |
| Flow: fine-tune (lambda = 0) | 128/150 (85%) / 138/150 (92%) / 144/150 (96%) | 135/150 (90%) / 138/150 (92%) / 144/150 (96%) |
| Flow + error net (history), selected lambda | 115/150 (77%) / 133/150 (89%) / 149/150 (99%) | 115/150 (77%) / 88/150 (59%) / 68/150 (45%) |
| Flow + action smoothness, selected beta | 128/150 (85%) / 143/150 (95%) / 145/150 (97%) | 131/150 (87%) / 143/150 (95%) / 145/150 (97%) |
| Flow + error net (history), fixed lambda = 0.3 | 0/150 (0%) / 0/150 (0%) / 0/150 (0%) | 1/150 (1%) / 0/150 (0%) / 0/150 (0%) |
| Flow (lambda = 0) + critic selects best of 8 chunks | 121/150 (81%) / 129/150 (86%) / 134/150 (89%) | 133/150 (89%) / 142/150 (95%) / 144/150 (96%) |
| Flow (lambda = 0) + critic selects best of 32 chunks | 118/150 (79%) / 119/150 (79%) / 124/150 (83%) | 125/150 (83%) / 132/150 (88%) / 137/150 (91%) |
| Flow (lambda = 0) + critic selects worst of 8 chunks | 134/150 (89%) / 143/150 (95%) / 144/150 (96%) | 131/150 (87%) / 138/150 (92%) / 148/150 (99%) |

## Section 6 paired success vs bc_flow
| Flow: pi0 (initial flow matching) | r=0 | +4.0 [-2.7, +10.7] p=0.307 | -0.7 [-6.7, +5.3] p=1.000 | **SIGN CHANGE**
| Flow: pi0 (initial flow matching) | r=2 | +4.0 [-1.3, +9.3] p=0.238 | +4.0 [-1.3, +9.3] p=0.238 |
| Flow: pi0 (initial flow matching) | r=4 | +2.7 [-0.7, +6.7] p=0.289 | +2.7 [-0.7, +6.7] p=0.289 |
| Flow: pi0 (initial flow matching) | r=all | +3.6 [+0.0, +7.1] p=0.033 | +2.0 [-1.3, +5.6] p=0.253 |
| Flow + error net (history), selected lambda | r=0 | -8.7 [-16.0, -2.0] p=0.029 | -13.3 [-20.0, -6.7] p=0.000 |
| Flow + error net (history), selected lambda | r=2 | -3.3 [-10.0, +2.7] p=0.405 | -33.3 [-42.0, -25.3] p=0.000 |
| Flow + error net (history), selected lambda | r=4 | +3.3 [+0.0, +6.7] p=0.125 | -50.7 [-59.3, -42.0] p=0.000 | **SIGN CHANGE**
| Flow + error net (history), selected lambda | r=all | -2.9 [-6.4, +0.4] p=0.124 | -32.4 [-38.4, -26.4] p=0.000 |
| Flow + action smoothness, selected beta | r=0 | +0.0 [-6.0, +6.0] p=1.000 | -2.7 [-8.0, +2.7] p=0.481 |
| Flow + action smoothness, selected beta | r=2 | +3.3 [-0.7, +8.0] p=0.227 | +3.3 [-0.7, +8.0] p=0.227 |
| Flow + action smoothness, selected beta | r=4 | +0.7 [-1.3, +2.7] p=1.000 | +0.7 [-1.3, +2.7] p=1.000 |
| Flow + action smoothness, selected beta | r=all | +1.3 [-1.1, +4.0] p=0.405 | +0.4 [-2.0, +2.9] p=0.860 |
| Flow + error net (history), fixed lambda = 0.3 | r=0 | -56.7 [-64.7, -48.7] p=0.000 | -43.3 [-51.3, -35.3] p=0.000 |
| Flow + error net (history), fixed lambda = 0.3 | r=2 | -59.3 [-67.3, -51.3] p=0.000 | -48.0 [-56.0, -40.0] p=0.000 |
| Flow + error net (history), fixed lambda = 0.3 | r=4 | -84.7 [-90.0, -78.7] p=0.000 | -80.0 [-86.0, -73.3] p=0.000 |
| Flow + error net (history), fixed lambda = 0.3 | r=all | -66.9 [-72.4, -61.1] p=0.000 | -57.1 [-62.7, -51.6] p=0.000 |
| Flow (lambda = 0) + critic selects best of 8 chunks | r=0 | -4.7 [-10.7, +2.0] p=0.210 | -1.3 [-7.3, +4.7] p=0.832 |
| Flow (lambda = 0) + critic selects best of 8 chunks | r=2 | -6.0 [-12.0, +0.0] p=0.078 | +2.7 [-1.3, +7.3] p=0.344 | **SIGN CHANGE**
| Flow (lambda = 0) + critic selects best of 8 chunks | r=4 | -6.7 [-12.0, -2.0] p=0.021 | +0.0 [-4.7, +4.7] p=1.000 |
| Flow (lambda = 0) + critic selects best of 8 chunks | r=all | -5.8 [-8.9, -2.9] p=0.001 | +0.4 [-2.0, +2.9] p=0.880 | **SIGN CHANGE**
| Flow (lambda = 0) + critic selects best of 32 chunks | r=0 | -6.7 [-13.3, +0.0] p=0.087 | -6.7 [-14.0, +0.7] p=0.121 |
| Flow (lambda = 0) + critic selects best of 32 chunks | r=2 | -12.7 [-20.0, -5.3] p=0.001 | -4.0 [-10.0, +2.0] p=0.263 |
| Flow (lambda = 0) + critic selects best of 32 chunks | r=4 | -13.3 [-20.0, -7.3] p=0.000 | -4.7 [-10.0, +0.7] p=0.143 |
| Flow (lambda = 0) + critic selects best of 32 chunks | r=all | -10.9 [-14.7, -7.3] p=0.000 | -5.1 [-8.7, -1.6] p=0.009 |
| Flow (lambda = 0) + critic selects worst of 8 chunks | r=0 | +4.0 [-2.0, +10.0] p=0.286 | -2.7 [-8.7, +3.3] p=0.523 | **SIGN CHANGE**
| Flow (lambda = 0) + critic selects worst of 8 chunks | r=2 | +3.3 [-0.7, +7.3] p=0.180 | +0.0 [-4.7, +4.7] p=1.000 |
| Flow (lambda = 0) + critic selects worst of 8 chunks | r=4 | +0.0 [-4.7, +4.7] p=1.000 | +2.7 [+0.0, +6.0] p=0.219 |
| Flow (lambda = 0) + critic selects worst of 8 chunks | r=all | +2.4 [-0.2, +5.1] p=0.126 | +0.0 [-2.2, +2.2] p=1.000 |

## Wall time per stage (24-core CPU; stages are CPU-bound)

```
stage	wall_s	exit
02_prepare_policy_flow	548	0
05b_audit_flow_gradient	89	0
05b_audit_flow_gradient_lr1e-5	63	0
06_train_policy_bc_flow	59	0
06_train_policy_error_flow	2712	0
06_train_policy_smooth_flow	1128	0
07_eval_flow	822	0
09_audit_selection	60	0
07_eval_sel	498	0
08_report	3	0
pytest	0	1
pytest_rerun	2	0
```

The first pytest call failed at collection: the shell PYTHONPATH included ROS Jazzy, whose launch_testing pytest plugin needs `lark`. Rerun with PYTHONPATH=src: 30 passed.
