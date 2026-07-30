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
  --num_envs 32 \
  --headless \
  --checkpoint /absolute/path/to/model_5000.pt
```

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
