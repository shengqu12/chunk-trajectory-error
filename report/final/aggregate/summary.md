# Cross-seed summary (5 seeds: 17, 18, 19, 20, 21)

Intervals are 95 % t-intervals across seeds of a per-seed quantity; paired differences are computed within each seed on the shared initial states against the same-class reference (`bc` for MLP policies, `bc_flow` for flow policies).

## 1. Closed-loop success (%, mean across seeds [95 % CI])

| Policy | r = 0 | r = 2 | r = 4 | pooled |
|---|---:|---:|---:|---:|
| MLP pi0 | 18.7 [10.6, 26.7] | 30.5 [15.5, 45.6] | 59.3 [38.5, 80.1] | 36.2 [22.3, 50.1] |
| MLP BC fine-tune (lambda = 0) | 46.3 [26.2, 66.3] | 61.7 [44.0, 79.5] | 74.1 [63.2, 85.0] | 60.7 [44.8, 76.6] |
| MLP + error net, selected lambda | 35.7 [12.3, 59.2] | 52.3 [31.6, 72.9] | 74.4 [61.3, 87.5] | 54.1 [35.7, 72.6] |
| MLP + error net (no history) | 36.0 [17.4, 54.6] | 52.4 [32.4, 72.4] | 70.1 [52.0, 88.2] | 52.8 [34.5, 71.1] |
| MLP + smoothness, selected beta | 47.6 [34.0, 61.2] | 60.4 [47.4, 73.4] | 73.1 [65.2, 81.0] | 60.4 [49.1, 71.6] |
| MLP + error net, lambda = 0.03 | 2.9 [-1.9, 7.8] | 7.7 [1.1, 14.4] | 14.9 [-7.5, 37.4] | 8.5 [-2.0, 19.1] |
| Flow pi0 | 90.7 [85.7, 95.7] | 95.9 [93.2, 98.6] | 97.5 [96.1, 98.8] | 94.7 [91.8, 97.5] |
| Flow fine-tune (lambda = 0) | 80.3 [62.8, 97.8] | 90.5 [76.6, 104.5] | 95.1 [89.3, 100.9] | 88.6 [76.3, 100.9] |
| Flow + error net, selected lambda | 65.6 [57.8, 73.4] | 86.4 [78.3, 94.5] | 88.1 [72.0, 104.3] | 80.0 [70.5, 89.5] |
| Flow + smoothness, selected beta | 81.3 [62.0, 100.7] | 91.2 [78.6, 103.8] | 95.5 [88.1, 102.8] | 89.3 [76.4, 102.3] |
| Flow + error net, lambda = 0.3 | 0.0 [0.0, 0.0] | 0.1 [-0.2, 0.5] | 6.9 [-11.9, 25.7] | 2.4 [-3.9, 8.6] |
| Flow + critic selects best of 8 | 72.9 [48.9, 97.0] | 82.3 [63.3, 101.2] | 88.4 [80.6, 96.2] | 81.2 [64.5, 97.9] |
| Flow + critic selects best of 32 | 67.7 [40.1, 95.4] | 73.2 [54.1, 92.3] | 77.2 [69.8, 84.6] | 72.7 [55.4, 90.1] |
| Flow + critic selects worst of 8 | 84.5 [75.0, 94.1] | 92.7 [86.1, 99.3] | 95.5 [89.8, 101.2] | 90.9 [83.7, 98.1] |

## 2. Paired success difference vs same-class reference (pp, mean across seeds [95 % CI]; seeds with negative / positive difference)

| Policy | r = 0 | r = 2 | r = 4 | pooled |
|---|---:|---:|---:|---:|
| MLP pi0 | -27.6 [-40.8, -14.4] (5−/0+) | -31.2 [-38.2, -24.2] (5−/0+) | -14.8 [-29.8, +0.2] (4−/1+) | -24.5 [-33.2, -15.9] (5−/0+) |
| MLP + error net, selected lambda | -10.5 [-20.0, -1.1] (4−/1+) | -9.5 [-24.1, +5.1] (5−/0+) | +0.3 [-13.7, +14.3] (2−/3+) | -6.6 [-18.5, +5.3] (4−/1+) |
| MLP + error net (no history) | -10.3 [-32.4, +11.9] (3−/1+) | -9.3 [-32.8, +14.1] (2−/3+) | -4.0 [-26.5, +18.5] (2−/3+) | -7.9 [-30.5, +14.7] (2−/3+) |
| MLP + smoothness, selected beta | +1.3 [-6.2, +8.9] (3−/2+) | -1.3 [-9.6, +6.9] (4−/1+) | -1.1 [-5.6, +3.5] (4−/1+) | -0.4 [-6.7, +5.9] (4−/1+) |
| MLP + error net, lambda = 0.03 | -43.3 [-62.0, -24.7] (5−/0+) | -54.0 [-66.5, -41.5] (5−/0+) | -59.2 [-80.2, -38.2] (5−/0+) | -52.2 [-65.9, -38.4] (5−/0+) |
| Flow pi0 | +10.4 [-4.6, +25.4] (0−/5+) | +5.3 [-6.3, +16.9] (0−/4+) | +2.4 [-2.7, +7.5] (1−/3+) | +6.0 [-4.3, +16.4] (0−/5+) |
| Flow + error net, selected lambda | -14.7 [-30.2, +0.8] (4−/1+) | -4.1 [-13.9, +5.6] (4−/1+) | -6.9 [-18.3, +4.4] (3−/2+) | -8.6 [-14.8, -2.4] (5−/0+) |
| Flow + smoothness, selected beta | +1.1 [-1.3, +3.5] (1−/3+) | +0.7 [-1.9, +3.2] (2−/3+) | +0.4 [-1.3, +2.1] (1−/4+) | +0.7 [-0.0, +1.4] (1−/4+) |
| Flow + error net, lambda = 0.3 | -46.3 [-66.3, -26.2] (5−/0+) | -61.6 [-79.3, -43.9] (5−/0+) | -67.2 [-85.7, -48.7] (5−/0+) | -58.4 [-73.1, -43.6] (5−/0+) |
| Flow + critic selects best of 8 | -7.3 [-15.1, +0.4] (5−/0+) | -8.3 [-14.7, -1.8] (5−/0+) | -6.7 [-11.5, -1.8] (5−/0+) | -7.4 [-13.5, -1.3] (5−/0+) |
| Flow + critic selects best of 32 | -12.5 [-23.6, -1.5] (5−/0+) | -17.3 [-25.0, -9.6] (5−/0+) | -17.9 [-25.0, -10.7] (5−/0+) | -15.9 [-22.9, -8.9] (5−/0+) |
| Flow + critic selects worst of 8 | +4.3 [-5.0, +13.5] (2−/2+) | +2.1 [-5.9, +10.1] (3−/2+) | +0.4 [-0.5, +1.3] (1−/3+) | +2.3 [-3.6, +8.1] (3−/2+) |

## 3. Test-anchor J (simulator-recomputed, m²) and handover jump, relative to the reference

| Policy | mean J (×1e-5) | ΔJ vs reference (%) | Δ mean handover jump vs reference (%) |
|---|---:|---:|---:|
| MLP pi0 | – | – | – |
| MLP BC fine-tune (lambda = 0) | – | – | – |
| MLP + error net, selected lambda | – | – | – |
| MLP + error net (no history) | – | – | – |
| MLP + smoothness, selected beta | – | – | – |
| MLP + error net, lambda = 0.03 | – | – | – |
| Flow pi0 | – | – | – |
| Flow fine-tune (lambda = 0) | – | – | – |
| Flow + error net, selected lambda | – | – | – |
| Flow + smoothness, selected beta | – | – | – |
| Flow + error net, lambda = 0.3 | – | – | – |
| Flow + critic selects best of 8 | – | – | – |
| Flow + critic selects best of 32 | – | – | – |
| Flow + critic selects worst of 8 | – | – | – |

## 4. Gates and audits per seed

| Seed | critic test Pearson | 05 gradient gate (policy samples, ΔJ %) | G2 fine (lr 1e-5): ΔJ % at 1 / 5 / 25 updates, gate | G2 coarse (lr 3e-4): ΔJ % at 10 updates | pi0 r0 val | pi0_flow r0 val |
|---|---:|---:|---|---:|---:|---:|
| 17 | 0.974 | -22 %, pass | -0 / +8 / +30, fail | +1151 % | 6/30 | 29/30 |
| 18 | 0.974 | -20 %, pass | +1 / +9 / +30, fail | +575 % | 8/30 | 26/30 |
| 19 | 0.973 | -31 %, pass | -1 / +3 / +9, fail | +294 % | 4/30 | 29/30 |
| 20 | 0.958 | -20 %, pass | -0 / +5 / +15, fail | +908 % | 3/30 | 26/30 |
| 21 | 0.970 | -29 %, pass | -1 / +4 / +38, fail | +1468 % | 8/30 | 27/30 |

## 5. Selection audit on validation anchors (critic ranking among the policy's own 8 samples)

| Seed | within-anchor Spearman | J picked / random / rejected / oracle (×1e-5) | picked − random (%) | picked is oracle | mean rank of pick |
|---|---:|---:|---:|---:|---:|
| 17 | 0.47 | 2.56 / 4.30 / 7.05 / 1.80 | -40 | 37 % | 3.1 / 8 |
| 18 | 0.43 | 2.47 / 4.54 / 8.12 / 1.77 | -46 | 28 % | 3.4 / 8 |
| 19 | 0.47 | 2.11 / 3.98 / 6.67 / 1.55 | -47 | 37 % | 3.1 / 8 |
| 20 | 0.38 | 2.98 / 4.63 / 7.51 / 2.17 | -36 | 30 % | 3.4 / 8 |
| 21 | 0.40 | 2.34 / 4.13 / 7.29 / 1.64 | -43 | 30 % | 3.5 / 8 |
| **mean [CI]** | 0.43 [0.38, 0.48] | | -42 [-48, -37] | 33 [27, 38] % | 3.3 [3.0, 3.5] |
