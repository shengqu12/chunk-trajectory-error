# Action Chunk Trajectory Error：实现与验收 README

**交付目标。** 用一个仅在训练时运行的误差网络，把「动作 chunk 实际执行后偏离专家多远」变成可回传给策略的损失。先在 ManiSkill 跑通算法，再在 LIBERO 用预训练 SmolVLA 验证。部署时只跑策略和原来的 chunk 调度器。本文是实现规范；下列项目脚本和实验结果尚未生成。

## 1. 要回答的问题

相近的逐步动作误差可能有不同的轨迹后果：连续多步向同一方向偏，末端位置可能逐步漂离专家。异步执行中，连续两个 chunk 还基于不同时间的观测，接管时可能出现动作跳变。我们训练时直接针对前一个问题，检验它是否也能改善后一个问题。

| 待验证的主张 | 对应证据 |
|---|---|
| 误差网能提供有效的动作更新方向 | 在未见示范分支上，沿误差网的动作梯度做小步更新，比同幅度随机更新更能降低仿真重算的真实轨迹偏差 |
| 加入误差损失能改善策略 | 相对同起点、同预算的纯模仿微调，真实轨迹偏差下降；闭环接管跳变和任务成功率单独报告 |
| 不给部署增加误差修正开销 | 推理路径没有误差网、专家历史或额外采样；在同一机器上测策略延迟 |

**定义固定下来：** 本版误差网的标签比较「从同一仿真状态起步，在同一物理时间点的专家末端位置与候选动作末端位置」。它不直接预测两个新旧 policy chunk 的相互差异，也不把接管跳变当训练标签。后两者不能用这个标签代替。

## 2. 两套数据与仿真

| 次序 | 数据集与单任务 | 仿真和策略 | 用途 |
|---|---|---|---|
| 先跑 | **ManiSkill `PickCube-v1`** 官方 motion-planning demonstrations：抓起方块并移到目标位置 | ManiSkill 3 / SAPIEN，`physx_cpu`，状态观测、`pd_ee_delta_pos`，小型 chunk MLP | 低成本验证时间对齐、专家回放、误差网和梯度；不是 VLA 结果 |
| 主实验 | **LIBERO-Spatial**：`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`，从桌面中央拾起黑碗放到盘子 | LIBERO / robosuite / MuJoCo，预训练 `lerobot/smolvla_libero`；图像、指令、机器人状态按检查点输入 | 验证同一个训练目标能否改善真实 VLA 的闭环接管 |

两个环境各自训练，不迁移权重，不混合动作维度。使用独立 Python 环境；首次跑通后锁定 Python、仿真库、LeRobot Git commit、Hugging Face revision、数据哈希、控制器、控制频率、相机映射和随机种子。仿真直接使用数据集所属环境，无需额外接 Gazebo。

### ManiSkill：下载并转换示范

官方链接：[安装](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) · [示范下载](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) · [数据结构与回放](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) · [官方回放脚本](https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh)。

在已安装 ManiSkill 的独立环境里执行：

```bash
python -m mani_skill.utils.download_demo PickCube-v1
TRAJ="$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$TRAJ" --use-first-env-state -c pd_ee_delta_pos \
  -o state --save-traj --num-envs 1 -b physx_cpu
```

如下载目录不同，用 CLI 打印的实际路径。保留生成的 HDF5 和同名 JSON；后续只用转换后的示范训练，记录转换成功/失败数。检查 `traj_*` 的 `actions`、`env_states` 和 JSON 的环境及 reset 参数；动作使用与转换时相同的控制器。读取 env state 树时核对每条轨迹的 `T` 个动作与 `T+1` 个状态边界。状态回放优先走 `get_state_dict()` / `set_state_dict()`，并通过第 4 节的重复回放测试。

### LIBERO：只下载单任务 HDF5 和模型

官方/作者链接：[LIBERO 数据主页](https://libero-project.github.io/datasets.html) · [LIBERO 源码](https://github.com/Lifelong-Robot-Learning/LIBERO) · [单任务 HDF5 直链](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5) · [SmolVLA 检查点](https://huggingface.co/lerobot/smolvla_libero) · [LeRobot LIBERO 评测说明](https://huggingface.co/docs/lerobot/libero)。

```bash
python -m pip install huggingface_hub
hf download yifengzhu-hf/LIBERO-datasets \
  libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
  --repo-type dataset --local-dir data/raw/libero
hf download lerobot/smolvla_libero --local-dir assets/smolvla_libero
```

- 用任务名从 `libero_spatial` 解析任务索引和 BDDL 文件；不靠手填 ID。HDF5 的 `data/demo_*/actions`、`states` 以及模型/初始状态元数据用于重建专家回放。不要假设预存图片、状态与动作具有相同下标；先检查读取器，再从审计通过的回放导出配对样本。
- 使用 [检查点配置](https://huggingface.co/lerobot/smolvla_libero/blob/main/config.json) 与原预处理/后处理。检查两路相机的次序、方向及输入键，8 维策略状态、7 维动作与夹爪符号；仿真全状态与策略的 8 维状态是两种数据。先跑官方风格的零微调 baseline，确认同一观测送入自写适配器和参考调用时，固定噪声下的输出一致。
- 如需查阅现成的模仿学习格式，可参照 [LeRobot 版 LIBERO](https://huggingface.co/datasets/lerobot/libero)。这是同一基准的另一种格式，不另算数据集，也不要求下载全套。使用其中图像或动作前必须核对任务、示范 ID、控制时钟及预处理映射。

## 3. 算法与重点公式

### 3.1 时间、输入和超参数

`s_t, o_t` 都是执行 `a_t` **之前**的状态/观测。`p_{t+1}` 是执行 `a_t` 后的末端位置，单位米。`t` 是采集观测的仿真步，`r` 是采集到接管的总步数。用半开切片，预测从 `t` 对应的动作起算。

| 参数 | ManiSkill 首轮 | LIBERO 首轮 |
|---|---:|---:|
| `H`：策略输出步数 | 20 | 50 |
| `e`：实际执行步数 | 5 | 10 |
| `L`：误差网的历史步数 | 15 | 30 |
| `K`：轨迹偏差评估步数 | 10 | 20 |
| `r`：训练 / 闭环评测 | `0` / `0,2,4` | `0` / `0,2,4` |

`L=3e` 表示之前三个执行段长度的**连续状态历史**，不是重复或重叠的三个完整预测序列。`K>e` 是刻意的：研究多步偏差，评估的是候选的假想尾部；执行器仍只走 `e` 步。要求 `H ≥ r + max(e,K)`。首轮只训练 `r=0`，所以非零 `r` 的评测是对调度延迟的压力测试；若要进一步训练 `r>0`，单列实验。

### 3.2 给误差网生成真实标签

1. 按完整专家 episode 分割 train/val/test，锁定 split。取观测时刻 `t`，历史覆盖 `[t-L,t)`，并且后面至少有 `r+K` 步。记录 `o_t`、当前机器人状态 `q_t`、末端历史 `h_t`、参考轨迹与分支快照。
2. 以冻结策略生成 `A = π(o_t, ξ)`，`ξ` 是采样噪声（MLP 无需噪声）。训练可加少量有界的低频连续动作扰动，覆盖同方向漂移；夹爪不参加扰动。用专家动作分支做数据校验。测试单独统计真实策略样本，避免大量人为扰动掩盖误差网在策略分布上的表现。
3. 让候选分支与专家分支从**同一份**快照 `s*_{t+r}` 开始。候选执行 `A[r:r+K]`，参考执行专家的同一时段动作。两条轨迹分别走完，期间不重置。标签是同一时刻的状态差，首轮只取末端 XYZ：

$$
J_t(A;r)=\frac{1}{K}\sum_{j=1}^{K}
\left\|p^{\mathrm{cand}}_{t+r+j}-p^*_{t+r+j}\right\|_2^2.
$$

`J` 的单位是 m²；报表另给 `sqrt(J)` 的毫米数，方便解释。即使候选失败，也不能因此丢弃其轨迹偏差；使用允许固定长度记录的分支环境，异常中断单记并报告比例。`r>0` 时，`s*_{t+r}` 是离线分支用的专家快照，误差网和策略的输入仍只取到 `t`；这种 teacher-forced 标签不能充当真实异步闭环结果。

### 3.3 训练误差网、冻结后回传给策略

误差网 `E_φ` 输入：`(h_t, q_t, A[r:r+K], r)`，首轮 `K` 固定。`h_t` 用过去的末端位置相对当前末端的位置序列，`q_t` 用当前可观测机器人状态，动作使用**转换到执行器坐标后的实际命令**。用历史/动作各一支小型 MLP 或 1D Conv，拼接状态后回归一个非负标量。首轮不接图像，不预测置信度或数学上界；有必要再做冻结视觉特征消融。专家未来状态和候选 rollout 状态只用于标签，不进网络。

$$
\mathcal L_E(\phi)=\mathbb E[(E_\phi(h_t,q_t,A[r:r+K],r)-J_t(A;r))^2].
$$

固定 `φ`，以当前策略生成新候选 `A_θ`，更新策略：

$$
\mathcal L_\pi(\theta)=\mathcal L_{\mathrm{imit}}(\theta)
+\lambda\,\mathbb E[E_{\bar\phi}(h_t,q_t,A_\theta[r:r+K],r)].
$$

`L_imit` 在 ManiSkill 用动作回归，在 SmolVLA 用原生 flow-matching 模仿损失。两组使用相同的 BC 窗口和尾部 padding/mask；episode 开头历史不足的样本仍可参与 BC，只屏蔽误差项。若对 `J` 做尺度归一化，只用训练集拟合尺度，并在报表还原。`λ` 按验证集选择，至少含 `0`，同时记录两项梯度量级，避免正则掩盖模仿目标。

$$
\nabla_\theta E_{\bar\phi}
=\underbrace{\frac{\partial E_{\bar\phi}}{\partial A_\theta}}_{\text{误差网给动作的梯度}}
\underbrace{\frac{\partial A_\theta}{\partial\theta}}_{\text{动作生成对策略参数的梯度}}.
$$

- 冻结误差网权重 (`requires_grad_(False)`)，**保留**其对动作输入的梯度。仿真只产生标签，不在反向传播图里。基线和方法从相同 `π₀` 起步，使用相同数据、可训练层和更新步数。
- [SmolVLA 源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py) 的公开推理入口带 `no_grad`。实现 `sample_chunk_grad()`，在锁定版本中复用或改写底层采样，使去噪、动作后处理和必要的切片保留梯度。固定观测和初始噪声，先测普通推理与可微版本数值一致，再单独反传 `E`，要求可训练策略参数有有限非零梯度、误差网参数无梯度。不得把专家动作或训练时的噪声残差代替真正生成的 chunk。先冻结视觉/语言编码器，只微调 action expert 的选定层；两组保持一致。
- 循环：`冻结 π₀ → 采样/仿真 → 训练 E₀ → 冻结 E₀ → 微调 π₁`。首轮是主实验；若 `π₁` 上代理误差明显失准，再用 `π₁` 重采 train 分支，刷新 `E` 后更新策略，最多先做 1–3 轮。用真实仿真误差检查每轮，不能凭 `E` 的分数变小就判定收敛。

## 4. 仿真实现与数据合同

后端统一提供 `reset(seed_or_init)`、`observe()`、`snapshot()`、`restore(snapshot)`、`step(env_action)`、`ee_position()` 和官方成功标志。策略观测、用于恢复的完整仿真状态和动作命令是独立字段。训练阶段从专家历史取上下文；闭环评测从官方初始状态开始，由策略驱动后续图像/状态。

**先审计回放，再生成标签。** LIBERO 用 [`OffScreenRenderEnv` 包装器](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py) 或锁定版本的等价 API。仿真状态恢复未必包含控制器内部状态。对同一个锚点，用相同专家动作比较「不中断的前缀回放」与「快照恢复再执行」的末端和物体轨迹，并记录恢复前后首帧/状态；在生成候选之前固定容差并写入报告。如果仅恢复快照不一致，改为每条分支都从同一 episode 初态重放已记录前缀，让控制器走到一致状态。这个办法也不通过，就停止采标签，先查版本、控制器、初始姿态和时间轴。导出至少一段带时刻编号的专家/候选同起点回放视频。

**最小记录格式。** `anchors.parquet` 保存 `env, task, split, episode_id, t, r, H, e, K, L, obs_ref, snapshot_ref, history_ref`；`branches.h5` 按 `anchor_id/candidate_id` 保存 `action_cmd[K,D], p_candidate[K,3], p_expert[K,3], J_m2, rollout_status`，并记录 `policy_ckpt, round, seed, candidate_type`。只给每个锚点存一次图像与参考轨迹，分支用引用。单元测试中用专家动作与同分支真值验证 `J≈0`，并故意错一帧确认对齐检查能抓到。

**闭环调度。** 同步首个 chunk。以后设实际接管点为 `b`，在 `b-r` 的最新可用观测发起预测，旧 chunk 在这 `r` 步内照常执行；接管时用 `A[r:r+e]`。例如 LIBERO `e=10,r=4`：旧段执行第 6 步边界采样，执行完剩余 4 步后接管，执行新预测的下标 `4…13`（从 0 数）。可在离散仿真中先算结果但延后 `r` 步才允许取用，结果不得影响这 `r` 步的旧动作。用 mock policy 测命令索引、无重复执行、无空档。`r` 是**可控注入延迟**，不会自动证明满足真实控制时限；真实前向延迟另测。首轮不接 RTC 推理修补，两组共用完全相同的调度器。

## 5. 程序清单与交接顺序

计划目录：`configs/{maniskill_pickcube,libero_spatial}.yaml`；`src/backends/{maniskill,libero}.py`；`src/policies/{mlp_bc,smolvla}.py`；`src/{error_net,scheduler,metrics}.py`；`scripts/`、`tests/`、`runs/`。每次运行保存解析后的配置、软件版本和种子。以下是**待实现**的统一入口：`python scripts/<script>.py --config configs/<env>.yaml`；多轮或切分所需参数加 `--round`、`--split` 和 `--checkpoint`。

| 程序 | 做什么 | 必须产出 |
|---|---|---|
| `00_audit_setup.py` | 读取一条示范；校验任务、控制器、相机、动作、时间戳和模型 | `manifest.json`、一份样本的输入/输出形状 |
| `01_replay_expert.py` | 锁定 episode split、验证恢复与重复回放、导出锚点 | `splits.json`、`replay_audit.csv`、回放视频 |
| `02_prepare_policy.py` | ManiSkill 训练 BC；LIBERO 加载检查点，做零微调评测与适配器对照 | `pi0` 引用、初始基线记录 |
| `03_collect_branches.py` | 冻结策略，在专家锚点生成候选并仿真打标签 | `anchors.parquet`、`branches.h5`、失败/裁剪统计 |
| `04_train_error.py` | 训练误差网；支持 `--no-history`，归一化统计只用 train | `error_round*.pt`、留出集回归与排序报告 |
| `05_audit_gradient.py` | 测参数梯度、采样一致性、代理梯度方向在仿真中是否有效 | `gradient_audit.csv` |
| `06_train_policy.py` | 同脚本跑 `λ=0` 与 `λ>0`；支持换 checkpoint 刷新下一轮 | `checkpoints/pi_bc/`、`checkpoints/pi_error/`、模仿/代理损失与梯度日志 |
| `07_eval_closed_loop.py` | 相同初始种子与调度器，评测成功率、动作跳变及延迟 | `eval_episodes.parquet`、`eval_boundaries.parquet`、配对视频 |
| `08_report.py` | 汇总留出分支及闭环评测，做成对统计和失败分析 | `results.md`、曲线和实验配置索引 |

**建议把实现交给两个 agent，按接口接力：** 第一位负责 `00–03`、后端和索引测试，先在 ManiSkill 提交一次可复现的回放与真实标签；第二位负责 `04–08`、策略适配器和调度器，消费固定格式的数据。两人可先用同一份小 fixture 并行开发，开始正式训练前必须拿到真实回放审计。LIBERO 先跑基线和三条示范；可微采样审计通过后才启动 SmolVLA 微调。

**第一张工单只做 ManiSkill `00–01`。** 给出环境依赖、可运行命令、至少一个锚点的恢复误差和一段视频。通过后，`02–03` 从同一锚点交付「专家重放 `J≈0` / 有界扰动后 `J>0`」的两行样本，再批量生产标签。这样可以先看到仿真，数据对齐有问题时也不必重跑训练。

## 6. 结果验证清单

- [ ] **数据与回放可复现。** 交付两个数据源与锁定版本；专家与专家从同起点分支的误差在审计容差内；保存一次专家 vs. 候选视频。示范按 episode 切分，同一条 episode 的全部候选在同一 split。小样本可先取 30 条有效示范（若足够，20/5/5），每条均匀取 3–4 个锚点；报告实际数量、失败原因和丢弃比例。LIBERO 预训练检查点可能见过同源示范，留出只保证本项目的微调/误差训练分割。
- [ ] **误差网能拟合，梯度方向也有效。** 对留出锚点报告 MAE 和同锚点候选排序，与常数/无历史基线比较。只在训练或验证时选扰动步长。测试时，在相同候选上做 `A' = A - η∇_A E / ||∇_A E||` 与等范数随机更新，屏蔽夹爪；保证有效扰动幅度一致，优先使用不触及边界的小步。恢复同一快照重算两组的 `J`，报配对下降量和置信区间。若排序不错但梯度并未改善真实 `J`，不要扩大策略训练。
- [ ] **公平的对照与消融。** 主表：初始 `π₀`、同预算纯 BC (`λ=0`)、BC + 带历史的误差网。关键消融：去历史但保留当前状态与候选动作；加入直接动作平滑对照 `L_imit + β·mean(||a_{h+1}-a_h||²)`（仅连续通道），检验是否值得学误差网。首轮有效后再比固定误差网 vs. 交替刷新，或 `K=e` vs. `K>e`。超参数在验证集定好后再锁测试。
- [ ] **分别看专家偏差、真实跳变、任务成功。** 专家偏差用留出同起点分支的 `J`；真实接管跳变只在闭环**已执行**边界 `b` 上测 `||u_b-u_{b-1}||₂`，其中 `u` 为执行器归一化后的连续命令，平移与旋转分开报，夹爪另报；给中位数、P95 和边界数量。另报末端位移/速度变化及 episode 步数，排查靠减速或停住得到低跳变。成功率用环境官方判定，给实际成功数；`r=0,2,4` 各条件用同一组测试初态。先跑约 30 个配对初态/条件作 pilot；正式比较按预算扩样，优先按 episode/初态做配对置信区间。小样本没有足够证据时，写明「成功率影响不确定」。
- [ ] **部署路径和延迟。** 导出能单独加载的策略权重，评测不加载误差网、不读专家历史；同设备、相同预处理/采样步数/批量，预热后测单 chunk 端到端 p50/p95，GPU 计时正确同步，记录设备与输入规格。把实测毫秒和人为注入的 `r` 分开报告。

**验收判断。** 如果真实 `J` 下降但接管跳变不降，能支持「训练改变了轨迹偏差」，不能支持「解决了 chunk 接管断层」。如果跳变下降但成功率下降，报告平滑与任务完成的取舍。只有两项改善且成功率证据支持保持或提高时，才得出闭环改善结论。两个环境中，LIBERO 是 VLA 主结果；ManiSkill 用于排除数据、梯度和调度实现错误。
