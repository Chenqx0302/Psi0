# Psi0 项目结构与复现导览

本文档基于当前工作区实际文件梳理，目标是在正式复现前，先把 Psi0 的代码框架、文档体系和数据资产边界搞清楚。重点结论：仓库本体提供了训练/部署/评估代码、机器人模型资产、统计文件、FAST tokenizer 和若干 baseline 适配；大规模训练数据、Psi0 主权重、部分 Git LFS 张量和本地训练输出不在当前 checkout 内，需要按 README 或脚本约定另行下载。

## 1. 从项目目标看整体架构

Psi0 是面向 humanoid loco-manipulation 的 VLA 项目。主链路可以按 5 层理解：

1. 数据层：真实遥操作数据、SIMPLE 仿真数据、EgoDex 人类第一视角数据、Humanoid Everyday 数据。
2. 统一格式层：真实/仿真数据主要转换为 LeRobot 格式；EgoDex 和 HE raw 也有独立 dataset adapter。
3. 训练层：`scripts/train.py` 动态加载 `src/psi/config/train/*.py` 配置，再动态实例化 `src/psi/trainers/*.py`。
4. 模型层：`src/psi/models/psi0.py` 组合 Qwen3-VL backbone 与 diffusion/flow action expert。
5. 部署评估层：SIMPLE 用 HTTP/FastAPI 服务与 eval client；真实机器人用 RTC/HTTP/WebSocket 客户端执行预测动作。

复现时最短路径不是先读所有 baseline，而是先跑通 Psi0 本体：环境 -> 数据格式 -> 单任务 fine-tune -> open-loop eval -> SIMPLE 或真机部署。baseline 主要用于对比和消融。

## 2. 顶层目录

| 路径 | 作用 | 复现关注点 |
|---|---|---|
| `README.md` | 仓库总入口，串起环境、真实机器人、baseline、SIMPLE、预训练/后训练、checkpoint。 | 第一入口。 |
| `pyproject.toml` | Python/uv 项目定义；包名 `psi`；依赖组含 `serve`、`viz`、`psi`、`simple`。 | 环境复现入口。 |
| `flake.nix`、`nix/` | Nix dev shell/runtime。 | 需要 SIMPLE/Isaac/Nix 时再深入。 |
| `src/psi/` | Psi0 主包。 | 核心阅读对象。 |
| `src/{act,dp,openpi,gr00t,h_rdt,egovla,InternVLA-M1}/` | 各 baseline 的源码或移植代码。 | 对比实验时阅读。 |
| `scripts/` | 数据转换、训练、部署、可视化、回归脚本。 | 复现命令入口。 |
| `baselines/` | baseline 的封装 README、训练/服务/eval 脚本。 | 比直接读源码更适合先上手。 |
| `examples/` | quick start、open-loop eval notebook、SIMPLE eval 示例。 | 低成本 sanity check。 |
| `real/` | 真机部署、遥操作采集、机器人资产、真机 client。 | 真机复现或理解数据采集时阅读。 |
| `assets/` | README 图片、统计文件、任务文本映射。 | 本地提供的小型数据资产。 |
| `tests/` | 当前只有 SIMPLE/CuRobo 缺失场景测试。 | 覆盖面很窄。 |
| `third_party/SIMPLE/` | SIMPLE 仿真平台子模块。 | 仿真数据生成和仿真评估依赖。 |
| `doc/`、`tasks/` | 本地任务过程文档和历史教训。 | 按项目约束维护，不属于论文复现资产。 |

## 3. Psi0 主包 `src/psi`

| 子目录/文件 | 作用 |
|---|---|
| `config/config.py` | 定义 `LaunchConfig`、`TrainConfig`、`DataConfig`、`ServerConfig` 等通用配置。 |
| `config/model_psi0.py` | Psi0 模型超参：action dim/chunk、diffusion/flow steps、Qwen3-VL 路径、VLM 是否训练等。 |
| `config/model_qwen3vl.py`、`model_act.py`、`model_dp.py` | Qwen3-VL 预训练与 ACT/DP baseline 的模型配置。 |
| `config/data_lerobot.py` | LeRobot 数据入口；约定 `root_dir/repo_id` 布局和 stats 加载。 |
| `config/data_egodex.py` | EgoDex 数据入口。 |
| `config/data_he.py` | Humanoid Everyday raw 数据入口。 |
| `config/data_mix.py` | EgoDex 与 HE 混合训练入口，支持 batch/token mixture sampler。 |
| `config/transform.py` | 数据转换主干：`repack -> field -> model`。这里把原始字段变成模型输入、归一化动作/状态、构造 Qwen3-VL prompt。 |
| `config/train/*.py` | 动态训练配置 schema；文件名由 `scripts/train.py` 第一个参数选择。 |
| `data/dataset.py` | map-style/iterable dataset wrapper 和 mixture dataset。 |
| `data/sampler.py` | batch mixture 与 token mixture 采样器。 |
| `data/lerobot/` | LeRobot 兼容层和 wrapper。 |
| `data/egodex/`、`data/humanoid/` | EgoDex 与 HE raw 的 dataset adapter。 |
| `models/psi0.py` | Psi0 模型实现，组合 Qwen3-VL 与 action transformer/action head。 |
| `tokenizer/fast_action_tokenizer.py` | FAST action tokenizer 封装。 |
| `trainers/trainer.py` | 抽象训练器，统一 optimizer、scheduler、accelerate、checkpoint、log。 |
| `trainers/pretrain.py` | Qwen3-VL + FAST action token 的预训练路径。 |
| `trainers/posttrain.py` | action expert 后训练路径。 |
| `trainers/finetune.py` | Psi0 real/simple fine-tune 路径。 |
| `trainers/act_g1.py`、`diffusion_policy_g1.py` | ACT/DP 训练器。 |
| `deploy/psi0_serve_simple.py` | SIMPLE HTTP policy server，暴露 `/act` 和 `/health`。 |
| `deploy/psi_serve_rtc-*.py` | 面向真机 RTC 的 Psi0 server。 |

训练入口的动态关系：

```text
scripts/train.py
  -> import psi.config.train.<config_name>.DynamicLaunchConfig
  -> tyro 解析 CLI 参数
  -> Trainer.instantiate(cfg.train.name)
  -> psi.trainers.<train_name>.<PascalName>Trainer
  -> create_datasets/create_dataloaders/init_models/step/evaluate/save
```

## 4. 数据与训练流

### 4.1 数据转换

| 脚本 | 作用 |
|---|---|
| `scripts/data/raw_to_lerobot.py` | 把真实遥操作 raw episode 转成 LeRobot 格式。 |
| `scripts/data/raw_to_lerobot_v2.py` | LeRobot 转换的另一版实现。 |
| `scripts/data/raw_he_to_psi0.py` | HE raw 数据转 Psi0/训练可用格式。 |
| `scripts/data/raw_dexmate_psi0.py` | DexMate 数据转 Psi0 格式。 |
| `scripts/data/calc_modality_stats.py` | 计算动作/状态统计量。 |
| `scripts/data/patch_lerobot_meta.py` | 修复 README 提到的 LeRobot meta 已知问题。 |
| `scripts/data/fix_lerobot_parquet.py` | 修复 LeRobot parquet。 |
| `scripts/data/download.py` | 下载辅助脚本。 |

### 4.2 训练脚本

| 路径 | 用途 |
|---|---|
| `scripts/train/psi0/pretrain-egodex-psi0-fast.sh` | EgoDex 预训练。 |
| `scripts/train/psi0/pretrain-he-psi0-fast.sh` | HE 预训练。 |
| `scripts/train/psi0/pretrain-mix-psi0-fast.sh` | EgoDex + HE 混合预训练。 |
| `scripts/train/psi0/posttrain-he-psi0.sh` | HE action expert 后训练。 |
| `scripts/train/psi0/posttrain-mix-psi0.sh` | 混合后训练。 |
| `scripts/train/psi0/finetune-real-psi0.sh` | 真实 G1 数据 fine-tune。 |
| `scripts/train/psi0/finetune-simple-psi0.sh` | SIMPLE 数据 fine-tune。 |
| `scripts/train/psi0/finetune-simple-psi0-no-he.sh` | 不加载 HE 后训练头的 SIMPLE fine-tune 变体。 |
| `scripts/train/ddp.launch.yaml`、`deepspeed.launch.yaml` | accelerate launch 配置。 |
| `scripts/deepspeed/*.json` | ZeRO 配置。 |

### 4.3 部署与评估

| 路径 | 用途 |
|---|---|
| `scripts/deploy/serve_psi0_simple.sh` | 启动 SIMPLE HTTP policy server。 |
| `scripts/deploy/serve_psi0-rtc*.sh` | 启动真实机器人 RTC server。 |
| `examples/simple/simple_eval.py` | SIMPLE eval client。 |
| `examples/simple/openloop_eval.ipynb` | 训练集 open-loop 推理检查。 |
| `scripts/viz/viz_episode_real.py` | 可视化真实 LeRobot episode。 |
| `scripts/viz/viz_episode_simple.py` | 可视化 SIMPLE episode。 |
| `scripts/viz/fk.py`、`g1.py` | G1 forward kinematics/可视化辅助。 |

## 5. Baseline 框架

| baseline | 封装入口 | 源码位置 | 作用 |
|---|---|---|---|
| ACT | `baselines/act/` | `src/act/`、`src/psi/trainers/act_g1.py` | sequence/action-chunk baseline。 |
| Diffusion Policy | `baselines/dp/` | `src/dp/`、`src/psi/trainers/diffusion_policy_g1.py` | diffusion action baseline。 |
| GR00T N1.6 | `baselines/gr00t-n1.6/` | `src/gr00t/` | NVIDIA GR00T 风格 VLA baseline。 |
| OpenPI π0.5 | `baselines/pi05/` | `src/openpi/` | OpenPI policy baseline。 |
| H-RDT | `baselines/h-rdt/` | `src/h_rdt/` | human-data pretrain + robot fine-tune baseline。 |
| EgoVLA | `baselines/egovla/` | `src/egovla/` | human egocentric video VLA baseline。 |
| InternVLA-M1 | `baselines/internvla-m1/` | `src/InternVLA-M1/` | InternVLA-M1 baseline。 |

当前 `baselines/egovla/README.md`、`baselines/h-rdt/README.md`、`baselines/internvla-m1/README.md` 是空文件；对应可读文档主要在 `src/egovla/quick_start.md`、`src/h_rdt/README.md`、`src/InternVLA-M1/README.md`。

## 6. 真实机器人部分 `real/`

| 路径 | 作用 |
|---|---|
| `real/README.md` | 真机环境、Unitree SDK2、AVP/PICO、图像服务、数据采集与部署流程。 |
| `real/teleop/main.py` | 遥操作主入口。 |
| `real/teleop/manager.py`、`worker.py`、`writers.py` | 遥操作任务管理、采集 worker、数据写入。 |
| `real/teleop/image_server/` | RealSense 图像 server/client。 |
| `real/teleop/robot_control/` | G1 body、arm、hand、IK、retargeting、remote controller。 |
| `real/teleop/webrtc/` | WebRTC 图像/视频服务。 |
| `real/deploy/psi-inference.py` | 真机 Psi0 推理客户端。 |
| `real/deploy/psi-inference_rtc.py` | RTC/WebSocket 推理执行客户端。 |
| `real/deploy/*_inference.py` | 各 baseline 的真机 inference client。 |
| `real/scripts/deploy_psi0-rtc.sh` | 真机 client 启动脚本。 |
| `real/assets/` | G1/H1/hand URDF、MJCF/XML、YAML、mesh。 |

真机侧提供了大量机器人几何与控制资产，但真实采集数据本身被 `.gitignore` 排除，需要本地采集或从 Hugging Face 下载。

## 7. Markdown 文件清单

### 7.1 项目主文档与示例

| 文件 | 内容 |
|---|---|
| `README.md` | 项目总览、环境、真实机器人 fine-tune、baseline、SIMPLE、预训练/后训练、checkpoint、排障、引用。 |
| `baselines/README.md` | baseline 环境管理说明，强调 uv 与共享 `src/`。 |
| `baselines/act/README.md` | ACT 的 real/simple 训练、服务、SIMPLE eval。 |
| `baselines/dp/README.md` | Diffusion Policy 的 real/simple 训练、服务、SIMPLE eval。 |
| `baselines/gr00t-n1.6/README.md` | GR00T 训练、部署、open-loop eval 的最小入口。 |
| `baselines/pi05/README.md` | OpenPI π0.5 环境、权重、数据、训练、服务、eval 与适配说明。 |
| `baselines/egovla/README.md` | 空文件，占位。 |
| `baselines/h-rdt/README.md` | 空文件，占位。 |
| `baselines/internvla-m1/README.md` | 空文件，占位。 |
| `examples/visualize.md` | real/simple episode 可视化说明。 |
| `examples/simple/README.md` | SIMPLE policy eval 示例，含 Docker/Nix 路径。 |
| `examples/quick_start/psi.md` | 仓库级 quick start：clone、submodule、Nix、uv、baseline 资产。 |
| `examples/quick_start/simple.md` | SIMPLE 作为库使用的 quick start、datagen、eval。 |
| `examples/quick_start/gr00t.md` | GR00T xmove-pick 训练与 SIMPLE eval。 |
| `examples/quick_start/hrdt.md` | H-RDT xmove-pick 训练、open-loop eval、SIMPLE/DR eval。 |
| `examples/quick_start/egovla.md` | EgoVLA xmove-pick 训练、open-loop eval、SIMPLE/DR eval。 |

### 7.2 `src/` 内文档

| 文件 | 内容 |
|---|---|
| `src/fast/README.md` | FAST action tokenizer：EgoDex 预处理、下载官方 tokenizer、拟合 tokenizer。 |
| `src/gr00t/README.md` | GR00T 训练、部署、open-loop eval 简要说明。 |
| `src/h_rdt/README.md` | H-RDT 完整说明：安装、EgoDx 预训练、RobotWin2 微调、训练模式、配置。 |
| `src/h_rdt/quick_start.md` | H-RDT 快速复现路径，含 SIMPLE bend-pick eval。 |
| `src/h_rdt/datasets/pretrain/README.md` | H-RDT 预训练数据预处理，面向 EgoDx。 |
| `src/h_rdt/datasets/robotwin2/README.md` | H-RDT RobotWin2 数据准备、统计和语言 embedding。 |
| `src/h_rdt/inference/robotwin2_example/README.md` | H-RDT RobotWin2 inference 示例。 |
| `src/egovla/quick_start.md` | EgoVLA Nix/uv 环境、checkpoint、短训、服务、open-loop eval。 |
| `src/InternVLA-M1/README.md` | InternVLA-M1 环境、real/sim 训练、部署、eval client。 |
| `src/InternVLA-M1/InternVLA/dataloader/gr00t_lerobot/README.md` | 空文件，占位。 |

### 7.3 真实机器人资产文档

| 文件 | 内容 |
|---|---|
| `real/README.md` | 真实机器人部署与采集指南，含 G1/Host、Unitree SDK2、AVP/PICO、图像 server、任务采集。 |
| `real/assets/g1/README.md` | G1 机器人 URDF/MJCF 说明和 MuJoCo 查看方式。 |
| `real/assets/h1_2/README.md` | H1 51-DOF 机器人描述、URDF/MJCF、MuJoCo 可视化。 |
| `real/assets/h1_inspire/README.md` | H1 + Inspire hand 的 ROS/URDF/Gazebo/RViz 说明。 |

### 7.4 过程文档

| 文件 | 内容 |
|---|---|
| `tasks/lessons.md` | 历史教训，任务开始前必须回看。 |
| `doc/task_issue.md` | 本地任务状态登记。 |
| `doc/task_plan.md` | 本地任务阶段计划。 |
| `doc/progress.md` | 本地执行进度记录。 |
| `doc/findings.md` | 本地结论、风险、阻塞记录。 |
| `doc/project_walkthrough.md` | 本文档，项目结构、框架、Markdown 与数据资产导览。 |

### 7.5 `third_party/SIMPLE` 文档

| 文件 | 内容 |
|---|---|
| `third_party/SIMPLE/README.md` | SIMPLE 仿真平台主入口、系统要求、Nix/uv/Docker、eval/datagen。 |
| `third_party/SIMPLE/license.md` | SIMPLE 许可声明。 |
| `third_party/SIMPLE/data/README.md` | 数据输出目录占位说明。 |
| `third_party/SIMPLE/docs/source/index.md` | SIMPLE 文档站首页与章节索引。 |
| `third_party/SIMPLE/docs/source/license.md` | 文档站许可页。 |
| `third_party/SIMPLE/docs/source/developer.md` | 开发者说明：四元数、Typer、懒下载、HF 上传。 |
| `third_party/SIMPLE/docs/source/troubleshooting.md` | Nix、CUDA、Isaac Sim、CuRobo、GLFW、libc 等排障。 |
| `third_party/SIMPLE/docs/source/nix-runtime.md` | Nix runtime 设计与入口脚本说明。 |
| `third_party/SIMPLE/docs/source/nix-setup/index.md` | Nix 设置章节索引。 |
| `third_party/SIMPLE/docs/source/nix-setup/installation.md` | Nix 安装、prereq check、dev shell。 |
| `third_party/SIMPLE/docs/source/nix-setup/runtime.md` | Nix runtime 的 host/GPU 边界、变量、失败模型。 |
| `third_party/SIMPLE/docs/source/core/index.md` | core 概念索引。 |
| `third_party/SIMPLE/docs/source/core/task.md` | task 概念占位页。 |
| `third_party/SIMPLE/docs/source/core/robot.md` | robot 概念占位页。 |
| `third_party/SIMPLE/docs/source/dr/index.md` | Domain Randomization 章节占位。 |
| `third_party/SIMPLE/docs/source/user-guides/index.md` | 用户指南索引。 |
| `third_party/SIMPLE/docs/source/workflows/index.md` | 工作流索引。 |
| `third_party/SIMPLE/docs/source/workflows/build_custom_robot.md` | 自定义机器人构建简要指南。 |
| `third_party/SIMPLE/docs/source/tasks/index.md` | 内置任务索引。 |
| `third_party/SIMPLE/docs/source/tasks/aloha_tabletop_grasp.md` | Aloha 桌面抓取任务页，占位为主。 |
| `third_party/SIMPLE/docs/source/tasks/franka_tabletop_grasp.md` | Franka 桌面抓取任务页，占位为主。 |
| `third_party/SIMPLE/docs/source/tutorials/index.md` | 教程索引。 |
| `third_party/SIMPLE/docs/source/tutorials/installation.md` | uv、CuRobo、资源下载、数据目录结构。 |
| `third_party/SIMPLE/docs/source/tutorials/docker.md` | Docker 镜像、构建、运行、代理和容器内命令。 |
| `third_party/SIMPLE/docs/source/tutorials/run_env.md` | 创建 SIMPLE 环境并手动 step。 |
| `third_party/SIMPLE/docs/source/tutorials/replay.md` | 数据集 episode replay。 |
| `third_party/SIMPLE/docs/source/tutorials/eval.md` | policy evaluation：下载 eval 数据、server/client、结果。 |
| `third_party/SIMPLE/docs/source/tutorials/data_gen.md` | motion planner agent、LerobotRecorder、episode 采集。 |
| `third_party/SIMPLE/docs/source/tutorials/teleop.md` | SIMPLE VR teleop，PICO/XRoboToolkit 配对、校准、控制映射。 |
| `third_party/SIMPLE/docs/source/tutorials/teleop.old.md` | 旧版 teleop 教程。 |
| `third_party/SIMPLE/third_party/evdev/README.md` | evdev 第三方依赖说明。 |

## 8. 数据资产与外部依赖资产

### 8.1 当前仓库实际存在的本地资产

| 路径 | 规模/状态 | 用途 |
|---|---:|---|
| `assets/media/*` | 约 1.3M | README/项目图：teaser、architecture、envs、pico。 |
| `assets/stats/egodex_stat_all.json` | 存在 | EgoDex 动作/状态统计。 |
| `assets/stats/he_raw_rel_stats_combined*.json` | 存在 | HE raw 相对动作/状态统计。 |
| `assets/stats/task_description_dict.json` | 存在 | 大量任务名到自然语言描述的映射。 |
| `scripts/data/task_description_dict.json` | 存在 | 用户转换 raw data 时编辑/使用的任务描述示例。 |
| `src/fast/egodex-rel-50w-1x48-v2048-s100/` | 约 140K | 已发布 FAST action tokenizer/processor/tokenizer JSON。 |
| `real/assets/{g1,h1_2,h1_inspire,inspire_hand,unitree_hand}/` | 约 181M | URDF/XML/YAML/STL 机器人模型资产。 |
| `real/teleop/adapter_jit.pt`、`adapter_norm_stats.pt`、`amo_jit.pt` | 存在 | 真机遥操作适配/AMO TorchScript 资产。 |
| `real/teleop/task_defs/example.json` | 存在 | 采集任务元数据示例。 |
| `src/h_rdt/datasets/pretrain/*.json` | 存在 | H-RDT EgoDex 预训练统计。 |
| `src/h_rdt/datasets/robotwin2/stats.json`、`task_instructions.csv` | 存在 | RobotWin2 统计和任务文本。 |
| `src/gr00t/**/Eagle-*/*.{json,txt}` | 约 10M | GR00T Eagle tokenizer/config/vocab；不是完整大模型权重。 |
| `examples/simple/openloop_eval.ipynb` | 存在 | open-loop eval notebook 示例。 |
| `third_party/SIMPLE/data/` | 只有 README | SIMPLE 数据输出约定目录，当前没有任务数据。 |

### 8.2 当前仓库引用但未实际落地的资产

| 路径/引用 | 状态 | 说明 |
|---|---|---|
| Hugging Face `USC-PSI-Lab/psi-data` | 未本地下载 | README 中 real/sim 数据下载来源。 |
| Hugging Face `USC-PSI-Lab/psi-model` | 未本地下载 | Psi0 checkpoint 来源。 |
| `/hfm/data/{egodex,HE_RAW,HE_RAW_no_static,simple}` | 本机不存在 | 训练脚本常用默认数据根。 |
| `/hfm/cache/checkpoints/psi0/*` | 本机不存在 | README/脚本默认 checkpoint 根。 |
| `data/`、`cache/`、`checkpoints/`、`.runs*`、`wandb/` | 本机不存在且被忽略 | 本地数据、缓存、训练输出目录。 |
| `ml-egodex_dataset/` | 本机不存在且被忽略 | EgoDex 软链约定。 |
| `src/fast/pi` | 本机不存在且被忽略 | 官方 physical-intelligence FAST tokenizer 下载目标。 |
| `src/h_rdt/**/lang_embeddings/*.pt` | 文件是 Git LFS 指针 | 需要 LFS 拉取后才是实际张量。 |
| `assets/robots/g1/g1_body29_hand14.urdf` | 不存在 | 可视化脚本默认路径之一；实际 G1 URDF 在 `real/assets/g1/`。 |

### 8.3 数据加载接口约定

| 接口 | 期望数据布局 |
|---|---|
| `LerobotDataConfig` | `<root_dir>/<repo_id>/` 下的 LeRobot 数据；通常需要 `meta/stats_psi0.json` 或 `meta/stats.json`。 |
| `EgoDexDataConfig` | EgoDex root 下按 part/extra/test 等任务目录组织的 `.hdf5` 与视频文件。 |
| `HERawDataConfig` | HE raw root 下含 `task_description_dict.json`、`category/task/episode_*/data.json` 和图像。 |
| `MixedDataConfig` | 同时引用 EgoDex 与 HE root，并按比例混合采样。 |
| `LeRobotDatasetWrapper` | 单 repo 用 `LeRobotDataset`，多 repo 用 `MultiLeRobotDataset`。 |

## 9. 建议阅读与复现顺序

1. 先读 `README.md` 到 Fine-Tuning、Open-Loop Evaluation、Deployment 三节，明确真实 G1 fine-tune 的闭环。
2. 读 `src/psi/config/train/finetune_real_psi0_config.py`、`src/psi/config/data_lerobot.py`、`src/psi/config/transform.py`，理解 CLI 参数如何映射到数据字段。
3. 读 `scripts/train.py` 和 `src/psi/trainers/finetune.py`，理解训练主循环和 Psi0 fine-tune loss。
4. 读 `src/psi/models/psi0.py`，只抓住 Qwen3-VL backbone、hidden states、action header、diffusion/flow 推理这几条主线。
5. 用 `examples/visualize.md` 或 `examples/simple/openloop_eval.ipynb` 检查数据/模型输出，再进入 SIMPLE 或真机部署。
6. 需要对比时，再按 `baselines/README.md` 和对应 baseline quick start 阅读。

## 10. 当前风险与注意事项

1. 当前环境之前未完成完整 `uv sync`，因为 SIMPLE/curobo 子模块依赖曾缺失；后续复现前需要重新确认环境。
2. 大数据和 checkpoint 不在仓库里，不能直接训练复现实验结果。
3. 部分 H-RDT language embedding 是 Git LFS 指针，当前不是实际 `.pt` 张量。
4. `.gitignore` 排除了多数数据/缓存/输出资产，这是合理的；复现实验应明确 `PSI_HOME`、`DATA_HOME`、`HF_HOME`。
5. 可视化脚本存在默认路径与当前资产路径不完全一致的情况，使用前要检查参数或路径。
