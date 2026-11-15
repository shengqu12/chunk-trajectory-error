# ManiSkill 第一阶段审计报告（00–01）

运行时间 2026-11-12，主机为 Linux 云端容器（2 核 CPU、7 GB 内存、无 GPU）。**不是 Windows，也没有在 Windows 上验证过。**

## 1. 环境

| 项目 | 值 |
|---|---|
| OS | Ubuntu 24.04，Linux 6.18 x86_64 |
| Python | 3.10.20 |
| PyTorch | 2.14.1+cu130，`torch.cuda.is_available()` = **False**（本机无 GPU） |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| gymnasium / numpy / h5py | 1.4.0 / 1.26.4 / 3.16.0 |
| NVIDIA 驱动 | 无（`nvidia-smi` 不存在） |
| 仿真 | `physx_cpu`，1 个环境，控制 20 Hz，仿真 100 Hz |
| 渲染 | Mesa lavapipe 软件 Vulkan（`VK_ICD_FILENAMES=/opt/lavapipe/lvp_icd.json`） |

完整版本锁定见 `envs/maniskill/requirements.lock.linux-py310.txt`，机器可读版本见 `manifest.json`。

两个安装坑：
- 这个版本的 ManiSkill 即使不渲染，创建环境时也要初始化 Vulkan；没有可用 Vulkan 驱动会直接报 `ErrorIncompatibleDriver`。
- `mplib 0.1.1` 在 numpy 2.x 下 segfault，降到 `numpy==1.26.4` 后正常。只有本地生成示范才需要 mplib。

## 2. 官方 smoke test

```bash
python -m mani_skill.examples.demo_random_action -e PickCube-v1 -b physx_cpu --seed 0
```
通过：环境启动，跑满 50 步随机动作后 `truncated=True` 正常结束。日志 `logs/smoke_random_action.log`。

## 3. 示范与回放产物

**示范不是下载的。** `python -m mani_skill.utils.download_demo PickCube-v1` 访问 huggingface.co 被本机网络策略拦截（403）。改用 ManiSkill 自带的官方 motion-planning 求解器在本地生成 10 条成功示范，生成方式与官方数据集相同，但文件内容不是官方发布的那一份：

```bash
python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1 -o none -n 10 \
  --only-count-success -b cpu --traj-name trajectory --record-dir data/raw/maniskill/_gen
python -m mani_skill.trajectory.replay_trajectory --traj-path data/raw/maniskill/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b physx_cpu
```

| 文件 | 控制器 | 说明 |
|---|---|---|
| `data/raw/maniskill/trajectory.h5` / `.json` | `pd_joint_pos`，动作 8 维 | 原始 motion-planning 示范，10 条 |
| `data/processed/maniskill/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` / `.json` | `pd_ee_delta_pos`，动作 4 维，观测 42 维 | 回放转换结果，10/10 成功 |

SHA-256 记录在 `manifest.json` 的 `files` 字段。

## 4. 审计命令与结果

```bash
python scripts/00_audit_setup.py --config configs/maniskill_pickcube.yaml
python scripts/01_replay_expert.py --config configs/maniskill_pickcube.yaml --split train --video-anchor-index 1
python -m pytest -q
```

- `00`：18 项检查全部通过。10 条 episode 全部满足 **T 个动作对应 T+1 个状态**（T = 49–88）。
- `pytest`：10 项通过（原有 6 项 + 新增读取器/对齐 4 项）。
- 切分（按 episode，seed 17）：train `[0,1,3,5,7,9]`，val `[2,4]`，test `[6,8]`。

### 单锚点：episode 0，t = 40（抓取阶段），K = 10，r = 0

容差在运行前固定为 1 mm（`audit.replay_position_tolerance_m`）。参考是从 episode 初态不中断回放。

| 恢复方式 | 末端最大误差 | 末端平均误差 | 方块最大误差 | 结论 |
|---|---:|---:|---:|---|
| 同环境内恢复内存快照 | 0.1133 mm | 0.0640 mm | 0.265 mm | 通过 |
| 换种子 reset 后恢复内存快照 | 0.1133 mm | 0.0640 mm | 0.265 mm | 通过 |
| 从 HDF5 的 `env_states[t]` 恢复 | 0.1133 mm | 0.0638 mm | 0.265 mm | 通过 |
| 从初态重放前缀 `actions[:t]` | 0 | 0 | 0 | 通过 |
| 负对照：故意错一帧 | 7.20 mm | 2.57 mm | – | 按预期被判不合格 |

视频：`replay_audit_ep0000_t0040_r0.mp4`（左连续回放，右恢复快照后回放，每帧标注状态下标 i、i−t 和当帧误差）。

### train 全部 18 个锚点

| 恢复方式 | 最大误差 | 各锚点平均误差的均值 | 容差内 |
|---|---:|---:|---:|
| 恢复内存快照（两种） | 0.2876 mm | 0.0417 mm | 18/18 |
| 从 HDF5 状态恢复 | 0.2882 mm | 0.0417 mm | 18/18 |
| 前缀重放 | 0.0010 mm | 0.00004 mm | 18/18 |
| 负对照错一帧 | 13.96 mm | 3.93 mm | 0/18 |

逐行数据在 `replay_audit.csv`，汇总在 `replay_audit_summary.json`。

## 5. 需要你判断的地方

1. **快照恢复不是逐位精确。** 恢复后的状态向量与快照一致（差 ≤ 1.5e-8），但继续执行 10 步后末端会差到 0.29 mm；自由空间锚点（t=15）只有约 0.0005 mm，接触/抓取阶段明显变大。推测是 `set_state_dict` 不恢复 PhysX 的接触缓存，这一点没有进一步验证。前缀重放基本精确。
2. 这意味着专家对专家的 `J` 用快照恢复时不是严格 0：最大 4.5e-8 m²（√J ≈ 0.21 mm）。如果后面扰动产生的 `J` 量级接近这个底噪，建议 `03` 改用前缀重放，代价是每条分支多跑 t 步。
3. 连续回放与 HDF5 里记录的状态相比，位姿最大差 1.4e-5，速度最大差 0.033（来自被夹住的方块的速度分量）。末端位置与记录观测的差小于 0.002 mm。
4. `set_state_dict` 不恢复 episode 步数计数器；`pd_ee_delta_pos` 控制器状态为空（`use_target=False`），无需额外恢复。
5. 示范只有 10 条且为本地生成。配置里 `num_episodes: 30` 未达到；换成官方下载文件后需重跑 `00`/`01`。

## 6. 未做

`02` 及之后的脚本、策略训练、LIBERO 全部未动。Windows 原生路径、有 GPU 时的 PyTorch 未验证。
