# HU_D03 Training Machine Handoff / 新机器训练交接

> Agent instruction: read this file completely before installing, changing, or
> running the HU_D03 task. Work through the phases in order. Do not skip asset
> validation, and stop before any real-robot deployment.

本文档用于在一台新的 Linux/NVIDIA GPU 机器上复现
`Velocity-HU-D03-History-v0`、`StandUp-HU-D03-v0` 和
`Tracking-Flat-HU-D03-v0` 的仿真与训练环境。当前分支包含 14-DoF 下肢速度
跟踪、从仰卧/俯卧/侧卧/随机倒地姿态恢复的 31-DoF 全身起立，以及 31-DoF
舞蹈参考动作跟踪；这些都还不是可直接上真机的控制器。

## 0. Agent 执行边界

Agent 必须遵守以下约束：

1. 使用 Isaac Lab `v2.3.2` 和 Isaac Sim `5.1`，不要自行升级版本。
2. WBC-AGILE 与机器人描述仓库必须同时存在；不要用 URDF 临时转换结果替换
   已提供的 USD 物理层。
3. 先完成资产校验、环境 smoke test，再扩大并行环境数开始正式训练。
4. 不要启用源 USD 中缺失的 `Robot=Robot` payload。当前配置有意选择
   `Physics=PhysX`、`Sensor=None`、`Robot=None`。
5. 不要把当前策略直接部署到真机。电机参数、并行连杆传动、急停与安全状态机
   尚未完成硬件标定。
6. 执行期间记录代码 SHA、资产 SHA、GPU、驱动、Isaac Sim/Isaac Lab 版本和完整
   启动命令。出现 NaN、关节/刚体解析失败、持续穿地或爆炸时立即停止，不要靠扩大
   reward/惩罚掩盖模型问题。

## 1. 机器与软件前提

- x86_64 Linux 训练机和可运行 Isaac Sim 5.1 的 NVIDIA GPU/驱动。
- Git、Git LFS、Conda，以及足够存放 Isaac Sim、Isaac Lab、机器人网格和训练日志
  的磁盘空间。
- GitHub 读取权限；W&B 仅在使用 `--logger wandb` 时需要。
- 不要在 Isaac Lab 环境外单独安装另一套 PyTorch/CUDA 来覆盖 Isaac 自带版本。

先记录机器状态：

```bash
nvidia-smi
git --version
git lfs version
uname -a
```

## 2. 获取两个仓库

两个仓库默认必须是同级目录，并使用下面的目录名：

```text
hu_d03_training/
├── WBC-AGILE/
└── limx_oli_description/
    └── HU_D03_description/
```

执行：

```bash
mkdir -p ~/hu_d03_training
cd ~/hu_d03_training

git lfs install
git clone \
  --branch agent/add-hu-d03-wbc-training \
  https://github.com/sabernagato/WBC-AGILE.git
git -C WBC-AGILE lfs pull

git clone \
  https://github.com/limxdynamics/humanoid-description.git \
  limx_oli_description
git -C limx_oli_description checkout a90f734c153aa3ecffc8b674af1e0a323cb55d1a
```

记录实际版本：

```bash
git -C WBC-AGILE rev-parse HEAD
git -C limx_oli_description rev-parse HEAD
git -C WBC-AGILE status --short
git -C limx_oli_description status --short
```

如果不能使用同级目录，通过环境变量显式给出资产位置：

```bash
export HU_D03_DESCRIPTION_ROOT=/absolute/path/to/HU_D03_description
# 只覆盖 Isaac USD 时也可使用：
# export HU_D03_USD_PATH=/absolute/path/to/HU_D03_03.usd
```

不要把这些变量写死为某一台机器的路径提交到代码仓库。

## 3. 安装 Isaac Sim、Isaac Lab 和 AGILE

按 Isaac Lab `v2.3.2` 的 binaries installation 流程安装 Isaac Sim 5.1，然后：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
git -C ~/IsaacLab checkout v2.3.2

# 根据实际 Isaac Sim 目录创建链接。
ln -s /absolute/path/to/isaac-sim ~/IsaacLab/_isaac_sim

cd ~/IsaacLab
./isaaclab.sh --conda agile_env
conda activate agile_env
./isaaclab.sh --install
```

安装本 fork：

```bash
conda activate agile_env
export ISAACLAB_PATH=~/IsaacLab

cd ~/hu_d03_training/WBC-AGILE
./scripts/setup/install_deps_local.sh
python scripts/verify_rsl_rl.py
```

`verify_rsl_rl.py` 必须确认加载的是仓库内
`agile/algorithms/rsl_rl` 的定制版本，而不是环境中残留的其他 `rsl_rl`。

## 4. 资产预检

优先使用 Isaac Lab 的 Python，这样 `pxr` 可直接检查 USD：

```bash
cd ~/hu_d03_training/WBC-AGILE
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/validate_hu_d03_assets.py
```

成功标准是进程退出码为 `0`，并显示 `PASSED`。下面这些是当前已知、可解释的
warning，不等同于失败：

- URDF 只有少量 collision；训练使用 USD 的 PhysX collision。
- 源 USD 的可选 `HU_D03_03_robot.usd` payload 缺失；任务明确使用
  `Robot=None`。
- MJCF 的脚踝和腰部是并行机构执行器；通用 sim-to-MuJoCo 评估前仍需传动映射。

任何 missing asset、31 个主动关节不匹配、USD articulation/collision 缺失都属于
阻塞错误，必须先修复。

## 5. 分阶段启动

### Phase A：加载环境

先用两个环境、有限步数、无窗口模式检查注册、关节匹配和仿真稳定性：

```bash
python scripts/play.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 2 \
  --num_steps 200 \
  --headless
```

日志中的 action dimension 应为 `14`。检查机器人初始姿态、双脚接触、base 高度，
并确认没有 unresolved joint/body、NaN 或 PhysX articulation 错误。

### Phase B：短训练

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 32 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

确认能够完成迭代并生成 checkpoint：

```text
logs/rsl_rl/velocity_hu_d03_lower/<timestamp>_velocity_hu_d03_lower/model_*.pt
```

### Phase C：逐步扩大

先尝试 `256` 或 `512` 个环境，再根据显存和仿真吞吐扩大；不要第一次启动就假定
`2048` 一定适合当前 GPU。

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --headless \
  --logger tensorboard
```

确认 reward、episode length、关节速度、接触力和 value loss 没有异常后，正式训练：

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 2048 \
  --headless \
  --logger wandb \
  --log_project_name Velocity-HU-D03-Lower
```

默认配置训练 `50,000` iterations，每 `250` iterations 保存一次。使用 W&B 前只在
机器本地完成 `wandb login`，不要把 API key 写入仓库、命令日志或交接文档。

## 6. 恢复、评估和保留产物

从绝对路径恢复：

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 2048 \
  --headless \
  --resume True \
  --checkpoint /absolute/path/to/model_5000.pt
```

评估 checkpoint：

```bash
python scripts/eval.py \
  --task Velocity-HU-D03-History-v0 \
  --headless \
  --seed 42 \
  --checkpoint /absolute/path/to/model_5000.pt \
  --run_evaluation \
  --eval_config agile/algorithms/evaluation/configs/hu_d03_velocity_eval.yaml \
  --metrics_file /absolute/path/to/eval_metrics.json
```

该评估配置固定使用 4 个环境、100 秒 episode，分别统计受控下半身和默认位姿保持的
上半身关节，并关闭训练阶段的随机推力事件。评估成功表示 episode 完整存活；速度
跟踪误差和关节动态指标仍需结合 `eval_metrics.json` 判断。

### 2026-07-31 本机仿真基线

此前阶段推荐 checkpoint 是：

```text
logs/rsl_rl/velocity_hu_d03_lower/2026-07-31_13-27-57_mixed_recovery_from_1000/model_1250.pt
```

在 seed 42 的固定 4 环境 × 100 秒评估中，x、y 和静止工况完整存活，yaw 工况在
1.16 秒终止，成功率为 75%。成功 episode 的受控下肢平均关节加速度为 0.532，
完整指标和唯一环境轨迹归档在同一 run 的
`evaluation/seed42_fixed_4x100s/`。该结果尚未通过纯 yaw 工况，不能描述为完整速度
能力通过，也不能进入真机阶段。

评估器在 `num_episodes=1` 时只接受每个物理环境的首次 episode；失败环境 reset 后
的重复终止不会增加完成数或覆盖该环境的首次轨迹。

两个后续实验没有升格为推荐模型：将训练 yaw 范围缩到 ±0.5 后仍为 75%，yaw
仅延长到 1.36 秒且下肢平均关节加速度恶化到 1.036；启用足部 yaw 正则的默认
转向降权后降为 50%，x/yaw 分别在 1.76/1.14 秒终止。两者的指标分别归档在
对应 `yaw_half_range_from_1250` 和 `command_aware_feet_yaw_from_1250` run 中。

### 2026-07-31 零吊挂前进行走进展

#### 本轮训练过程总结

本轮目标是把已有的稳定站立策略推进到可验证的持续前进行走。训练和筛选过程如下：

| 阶段 | 主要方法 | 结果 |
|---|---|---|
| 评估纠偏 | 将固定前进验收设为单环境、30 秒、`vx=0.45 m/s`、`vy=0`、`yaw_rate=0`，同时保存轨迹并分别检查存活、速度和偏航 | 排除了“只站着但被记为成功”的假阳性 |
| 步态启动 | 加入左右腿反相 gait phase、抬脚/接触调度/负载转移奖励和关节参考 | 模型开始明显交替迈步，但早期候选只能维持约 2 秒 |
| 吊挂课程 | 使用训练专用虚拟吊挂，从强辅助逐步降到 27% | 学会抬脚和换重心，但直接移除辅助仍会快速摔倒 |
| 25% 无辅助混训 | 25% 环境完全无吊挂，75% 环境保留 27% 辅助 | `model_1100` 的严格零吊挂评估仅存活 1.86 秒，未升级 |
| 50% 无辅助混训 | 无辅助环境提高到 50%，yaw L2 权重设为 -5 | 产生 `model_1200`，首次在零吊挂固定前进指令下完整行走 30 秒 |
| 强化 yaw 惩罚 | 从 `model_1200` 继续训练，将 yaw L2 权重提高到 -20 | 后继 `model_1299` 在 7.40 秒摔倒，说明单纯加大惩罚破坏稳定性 |
| 动作层偏航探测 | 对双髋 yaw 测试闭环阻尼和 0.05/0.10/0.20 rad 步频同步前馈 | 0.10 rad 前馈最优，将 30 秒 yaw-rate 误差从 0.4084 降到 0.2909 |
| 前馈适配短训 | 在 0.10 rad 前馈下从 `model_1200` 再训练 100 iterations | 新 `model_1299` 在 13.48 秒摔倒，继续保留 `model_1200` |

这一路径证明，稳定站立、短暂迈步和持续行走必须通过固定命令轨迹分开判断；训练
episode length、timeout 比例或肉眼看到腿动，都不能单独作为“已经会走”的证据。

当前仿真推荐 checkpoint 已更新为：

```text
logs/rsl_rl/velocity_hu_d03_lower/2026-07-31_17-33-33_phase_residual_mixed50_yaw_from1100/model_1200.pt
```

它来自实验任务
`Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Mixed-v0`。训练批次中
50% 环境完全无吊挂，另外 50% 保留 27% 虚拟辅助；评估时会删除吊挂动作和训练
扰动。控制器是确定性反相步态参考加 14 维学习残差的混合方案，不应描述为纯
端到端 RL。

seed 42、单环境、固定 `vx=0.45 m/s`、`vy=0`、`yaw_rate=0` 的 30 秒无吊挂评估中，
原始 checkpoint 完整存活，平均平面速度误差为 0.1429，平均绝对 yaw-rate 误差为
0.4084。左右髋、膝关节相关系数分别为 -0.708 和 -0.615，证明不是原地站立，而是
持续交替行走。

评估时增加 `--phase_yaw_amplitude 0.10` 的步频同步双髋 yaw 前馈后仍完整存活
1500 帧，平均平面速度误差为 0.1516，平均绝对 yaw-rate 误差降至 0.2909，机身高度
范围为 0.876–0.998 m，左右髋/膝相关系数为 -0.711/-0.680。完整指标和轨迹位于：

```text
/tmp/hu_d03_eval_model1200_yawphase010_30s/
```

复现命令：

```bash
python scripts/eval.py \
  --task Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Mixed-v0 \
  --checkpoint logs/rsl_rl/velocity_hu_d03_lower/2026-07-31_17-33-33_phase_residual_mixed50_yaw_from1100/model_1200.pt \
  --num_envs 1 \
  --headless \
  --run_evaluation \
  --eval_config agile/algorithms/evaluation/configs/hu_d03_forward_eval.yaml \
  --metrics_file /tmp/hu_d03_eval_model1200_yawphase010_30s/metrics.json \
  --save_trajectories \
  --phase_yaw_amplitude 0.10
```

该结果首次证明零吊挂、固定前进指令下可持续行走 30 秒，但仍未通过
`mean_yaw_rate_error <= 0.15`，加前馈后的线速度误差也以 0.0016 超过 0.15。
因此不得称为完整前进跟踪验收通过，更不能进入 sim-to-real。

不要升级两个失败后继模型：将 yaw L2 权重增至 -20 的
`2026-07-31_17-38-38_phase_residual_mixed50_yaw20_from1200/model_1299.pt`
在 7.40 秒摔倒；在 0.10 rad yaw 前馈下继续适配的
`2026-07-31_17-55-54_phase_yawff010_from1200/model_1299.pt` 在 13.48 秒摔倒。

#### 验证、版本与下一步

- 评估器测试 8 项、运动指标测试 9 项、HU_D03 reward 测试 11 项，共 28 项通过。
- 已用 GUI 按固定 `vx=0.45 m/s` 指令多次展示同一零吊挂配置，完整播放未提前摔倒。
- 本轮训练与评估实现提交在 WBC-AGILE commit `7bdee36`。
- 当前结论是“持续行走已实现，严格直线跟踪尚未通过”；下一阶段应从
  `model_1200` 出发，针对航向保持和步频同步偏航振荡设计训练或控制修正，不再
  仅靠提高 yaw reward 权重。
- 仍然禁止进入真机阶段；开始 sim-to-sim 或 sim-to-real 前必须完成第 0 节规定的
  执行器辨识、并行连杆映射和硬件安全工作。

每次重要实验至少保留：

- `model_*.pt`；
- 同一 run 目录下的 `params/env.yaml` 和 `params/agent.yaml`；
- W&B/TensorBoard 指标；
- WBC-AGILE 和机器人描述仓库 SHA；
- 启动命令、seed、GPU/驱动与 Isaac 版本；
- 训练中发现的模型、碰撞或关节映射异常。

## 7. 常见故障

| 现象 | 首要检查 |
|---|---|
| `HU_D03_03.usd` not found | 两仓库是否同级，或 `HU_D03_DESCRIPTION_ROOT` 是否为绝对路径 |
| Task ID not found | 是否 checkout 正确分支、执行 `install_deps_local.sh`，以及当前是否在 `agile_env` |
| `rsl_rl` API/import 错误 | 重新执行安装脚本，再运行 `python scripts/verify_rsl_rl.py` |
| USD 引用 `HU_D03_03_robot.usd` 失败 | 不要启用 `Robot=Robot`；确认使用本分支的 `hu_d03.py` |
| CUDA OOM | 降低 `--num_envs`，关闭视频与其他 GPU 进程 |
| 仿真抖动、穿地或爆炸 | 停止训练，检查 USD 版本、初始姿态、collision、质量/惯量和电机参数 |
| MuJoCo 脚踝/腰部动作不一致 | 当前缺少并行连杆 transmission mapping，不要据此判断策略可部署 |

## 8. 容器和 OSMO 特别说明

当前 `workflows/Dockerfile` 只复制 WBC-AGILE build context，不会自动复制其同级的
`limx_oli_description`。因此 HU_D03 目前不能仅靠原有 `./run.py ... --rebuild`
在远端直接成功。

采用 Docker/OSMO 时，Agent 必须先完成其中一种方案：

1. 在运行时只读挂载 `HU_D03_description`，并设置容器内
   `HU_D03_DESCRIPTION_ROOT`；或
2. 调整受控 build context，把固定 SHA 的机器人描述复制进镜像，并设置同一变量。

容器内再次执行第 4 节资产校验后，才能提交训练作业。不要把宿主机绝对路径写入
镜像，也不要在没有资产版本 SHA 的情况下使用浮动 `master`。

## 9. 当前实现入口

- Robot/actuator：`agile/rl_env/assets/robots/hu_d03.py`
- Environment：`agile/rl_env/tasks/locomotion/hu_d03/velocity_history_env_cfg.py`
- PPO：`agile/rl_env/tasks/locomotion/hu_d03/agents/rsl_rl_ppo_cfg.py`
- Task registration：`agile/rl_env/tasks/locomotion/hu_d03/__init__.py`
- Stand-up environment：`agile/rl_env/tasks/stand_up/hu_d03/stand_up_env_cfg.py`
- Stand-up PPO：`agile/rl_env/tasks/stand_up/hu_d03/agents/rsl_rl_ppo_cfg.py`
- Fallen-state hook：`agile/rl_env/tasks/stand_up/hu_d03/pre_learn.py`
- Dance environment：`agile/rl_env/tasks/tracking/hu_d03/flat_env_cfg.py`
- Dance PPO：`agile/rl_env/tasks/tracking/hu_d03/agents/rsl_rl_ppo_cfg.py`
- Motion schema：`agile/common/hu_d03_motion.py`
- Starter motion generator：`scripts/utils/generate_hu_d03_dance_motion.py`
- Motion validator：`scripts/validate_hu_d03_motion.py`
- Asset validator：`scripts/validate_hu_d03_assets.py`
- Human-readable integration notes：`docs/source/hu-d03.md`

完成上述本地仿真训练只说明软件链路可用。进入 sim-to-MuJoCo 或 sim-to-real 前，
仍需单独完成执行器辨识、并行连杆映射、状态/动作接口和硬件安全验证。

## 10. 倒地起身任务

`StandUp-HU-D03-v0` 的策略动作维度为 `31`，控制所有主动关节。它的目标是从倒地
状态恢复并保持稳定站立，不包含“主动卧倒”命令。

首次启动时，`pre_learn` 会自动生成并缓存两套倒地状态：

- primary：仰卧、默认关节位置、零初速度；
- secondary：随机方向、随机关节位置和小范围初速度，用于覆盖俯卧、左右侧卧与
  非规则倒地姿态。

先可视化检查倒地状态。远程无显示时可追加 `--headless`，但应保存视频或在有显示
的机器上至少检查一次：

```bash
python scripts/play.py \
  --task StandUp-HU-D03-v0 \
  --num_envs 16 \
  --validate-fallen-states \
  --num_steps 500
```

检查动作维度为 `31`，并确认倒地状态没有穿地、关节爆炸或自碰撞锁死。然后做短
训练：

```bash
python scripts/train.py \
  --task StandUp-HU-D03-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

正式训练前按 `64 → 256 → 512/1024` 逐步扩大环境数：

```bash
python scripts/train.py \
  --task StandUp-HU-D03-v0 \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name StandUp-HU-D03
```

产物位于：

```text
logs/rsl_rl/stand_up_hu_d03/<timestamp>_stand_up_hu_d03/model_*.pt
```

成功标准不能只看机器人“被拉起来”。训练初期有最高 90% 自重的虚拟 lift assist；
必须确认 `adaptive_lift` curriculum 已将 assist 降为 `0`，再分别统计仰卧、俯卧、
左右侧卧的无辅助起立成功率。当前策略也不能直接上真机，因为倒地接触会显著放大
尚未标定的碰撞、执行器和并行连杆误差。

## 11. 舞蹈参考动作跟踪

`Tracking-Flat-HU-D03-v0` 是独立的全身策略，动作维度为 `31`。它逐帧跟踪
50 Hz 参考动作中的关节位置/速度和 15 个关键身体的位姿/速度；它不会替代行走或
倒地起身策略。

不要直接复用 G1 舞蹈 `.npz`。HU_D03 与 G1 的关节数、腕部轴定义、头部关节、
连杆尺寸和动作数组顺序都不同。训练文件必须包含 `joint_names` 和 `body_names`
元数据，并通过本仓库校验。

先生成可自由使用的 pipeline integration 动作。该动作只有摆动、屈膝和手臂波浪，
用于验证完整训练链路，不代表最终舞蹈质量：

```bash
cd ~/hu_d03_training/WBC-AGILE
mkdir -p motions

python scripts/utils/generate_hu_d03_dance_motion.py \
  --output-file motions/hu_d03_starter_dance.npz

python scripts/validate_hu_d03_motion.py \
  motions/hu_d03_starter_dance.npz
```

校验必须显示：`50 fps`、`31 joints`、`15 bodies` 且退出码为 `0`。自定义舞蹈
也必须先重定向到同一契约；如果没有 HU_D03 FK 后的 `body_*` 数据，不能只拿一组
关节角开始训练。

先可视化 reference reset：

```bash
export MOTION_FILE="$PWD/motions/hu_d03_starter_dance.npz"
python scripts/play.py \
  --task Tracking-Flat-HU-D03-v0 \
  --num_envs 4 \
  --num_steps 500
```

确认身体和 ghost/reference 姿态一致，双脚不穿地，动作维度为 `31`，没有
joint/body remap 或 NaN 错误。然后做短训练：

```bash
export MOTION_FILE="$PWD/motions/hu_d03_starter_dance.npz"
python scripts/train.py \
  --task Tracking-Flat-HU-D03-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

通过后按 `256 → 512/1024` 逐步扩大：

```bash
export MOTION_FILE="$PWD/motions/hu_d03_starter_dance.npz"
python scripts/train.py \
  --task Tracking-Flat-HU-D03-v0 \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name Tracking-HU-D03
```

默认训练预算为 `30,000` iterations，产物位于：

```text
logs/rsl_rl/hu_d03_flat_tracking/<timestamp>_hu_d03_dance/model_*.pt
```

结果验收至少包括：全片段 tracking error、脚底非预期滑动、非脚/手接触、关节/扭矩
限位、随机扰动恢复能力，以及不同 phase 起始的成功率。starter motion 训通只说明
参考跟踪链路有效；要获得真正的舞蹈效果，还需要高质量 HU_D03 重定向动作和进一步
调参。任何策略上真机前仍必须完成第 0 节安全边界中的硬件标定与保护。
