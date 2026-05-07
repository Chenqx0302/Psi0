# SIMPLE MP 数据构造链路梳理

本文只基于项目文档与源码阅读整理，没有运行仿真、没有修改业务代码。

## 0. 结论先行

SIMPLE 确实提供了基于 cuRobo 的 MP 数据生成主链路：入口是 `third_party/SIMPLE/src/simple/cli/datagen.py`，核心组合是 `Gym Env + Task.decompose() + CuRoboPlanner + MotionPlannerAgent + LerobotRecorder`。这个链路可以复用来做新任务，但前提是新任务可以被现有 MP 子任务 DSL 表达，并且目标物体/容器具备稳定姿态、碰撞 mesh、可用抓取缓存或 Bodex/GSNet 抓取支持。

需要特别区分两类数据：

- `datagen.py` 直接产物：原始 SIMPLE LeRobot 数据，字段来自环境 observation/action space，例如 `observation.rgb_*`, `observation.joint_qpos`, `observation.amo_policy_*`, `action`。
- Psi0 发布的 `/simple/<task>` 训练数据：不是纯粹的 `datagen.py` 原始格式，而是经过 `third_party/SIMPLE/scripts/postprocess_psi0.py` 之类脚本转成 Psi0 训练 schema，含 `states`、36 维 `action`、`observation.images.egocentric`、拆分后的 hand/arm/leg joints、`stats_psi0.json` 等。

因此，如果目标是“像官方 SIMPLE 数据一样用于 Psi0 fine-tune”，完整链路应当是：

1. 定义/注册 MP task。
2. 用 `simple.cli.datagen` 生成成功 rollout 的原始 LeRobot 数据。
3. 用 Psi0/SIMPLE 的 postprocess 脚本转成 Psi0 训练格式。
4. 生成 eval-only 的 `level-0/1/2` environment_config 数据。
5. 用 `simple.cli.eval` 或相关 baseline eval 读取对应 level 的 eval dataset 做闭环验证。

## 1. 文档入口与官方暴露的命令

Psi0 README 只把 MP 数据生成指向 SIMPLE 文档，明确说 motion-planning based data generation 请参考 SIMPLE docs，并提供已经采集好的 6 个 SIMPLE 任务数据下载入口。相关位置：

- `README.md:336-343`：SIMPLE data generation 说明。
- `README.md:350-363`：下载 SIMPLE task data。
- `README.md:426-456`：SIMPLE eval 使用 `level-0/1/2`，MP 任务用 `eval.py` + `psi0`。

SIMPLE/Psi0 quick start 给出了可直接跑的 datagen 命令：

```bash
python -m simple.cli.datagen \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  --sim-mode mujoco_isaac \
  --headless \
  --data-format lerobot \
  --save-dir $PWD/third_party/SIMPLE/data/datagen \
  --num-episodes 1 \
  --shard-size 1 \
  --dr-level 0
```

见 `examples/quick_start/simple.md:81-115`。

SIMPLE 文档中的 `data_gen.md` 展示了同一条核心思路：创建 env、创建 `CuRoboPlanner`、创建 `MotionPlannerAgent`、包上 `LerobotRecorder`、执行 agent 动作直到 episode 结束，见 `third_party/SIMPLE/docs/source/tutorials/data_gen.md:32-89`。

## 2. MP 数据构造主链路

### 2.1 环境与任务创建

入口在 `third_party/SIMPLE/src/simple/cli/datagen.py:17-36`，主要参数包括：

- `env_id`：Gym env id，例如 `simple/G1WholebodyBendPickMP-v0`。
- `scene_uid`、`target_object`：可覆盖场景/目标物体。
- `sim_mode`：常用 `mujoco_isaac`，MuJoCo 做物理，Isaac Sim 做渲染。
- `data_format`：当前实现只支持 `lerobot`。
- `save_dir`、`num_episodes`、`shard_size`。
- `dr_level`：用于构造 task 的 domain randomization level。
- `plan_batch_size`、`ignore_target_collision`、`easy_motion_gen`：影响 cuRobo 规划。
- `eval`：只生成 eval environment configs，不生成完整动作 rollout。

源码里有一个实现细节需要注意：`datagen.py` 暴露了 `--plan-batch-size`，但构造 `CuRoboPlanner` 时 `plan_batch_size` 写死为 `1`；CLI 传入值进入的是 `MotionPlannerAgent(plan_batch_size=...)`，主要控制 GSNet/Bodex 取多少候选 grasp/pose，再交给 planner。后续如果要提升候选批量规划能力，需要单独确认 `CuRoboPlanner` 侧是否也要开放该参数，而不是只改命令行参数。

`datagen.py:37-55` 调用：

```python
env = gym.make(env_id, **make_kwargs)
task = env.unwrapped.task
```

Gym env 注册集中在 `third_party/SIMPLE/src/simple/envs/__init__.py`。例如：

- `simple/G1WholebodyBendPickMP-v0` -> `simple.envs.loco_manipulation:LocoManipulationEnv`，task uid `g1_wholebody_bend_pick_mp`，见 `envs/__init__.py:101-105`。
- `simple/G1WholebodyTabletopGraspMP-v0` -> task uid `g1_wholebody_tabletop_grasp_mp`，见 `envs/__init__.py:117-120`。
- `simple/G1WholebodyXMoveAndPickMP-v0` -> task uid `g1_wholebody_x_move_and_pick_mp`，见 `envs/__init__.py:164-168`。

`BaseDualSim` 在初始化时通过 `TaskRegistry.make(task, *args, **kwargs)` 实例化任务，见 `third_party/SIMPLE/src/simple/envs/base_dual_env.py:57-76`。任务类通过 `@TaskRegistry.register(...)` 注册，注册逻辑在 `third_party/SIMPLE/src/simple/core/registry.py:18-34`。

任务模块的导入聚合在 `third_party/SIMPLE/src/simple/tasks/__init__.py:24-108`，新增任务时如果不被导入，装饰器不会执行，`TaskRegistry` 也就找不到它。

### 2.2 Task reset：生成一局具体场景

抽象基类 `Task` 的核心职责在 `third_party/SIMPLE/src/simple/core/task.py`：

- `__init__` 保存 `dr`、`split`、`render_hz`、`dr_level`、`physics_dt` 到 metadata，见 `core/task.py:54-72`。
- `reset()` 负责调用 DR manager，构造 `Layout`，添加 robot、scene/table、container、target、distractors、articulated object、spatial pose、lighting、camera、material，见 `core/task.py:135-250`。
- `state_dict()` 保存当局完整 `environment_config`，包括 `uid/label/description/metadata/robot_cfg/sensor_cfgs/dr_cfgs/dr_state_dict/layout`，见 `core/task.py:252-269`。
- `decompose()` 是 MP 子任务序列接口，见 `core/task.py:298-304`。

代表性 MP 任务文件都按同一模式写：

- `metadata`：物理频率、渲染频率、最大步数、是否需要重力等。
- `robot_cfg`：例如 `uid="g1_wholebody"`。
- `sensor_cfgs`：相机配置。
- `dr_cfgs`：语言、目标、容器、干扰物、空间范围、场景、灯光、材质。
- `reset()`：调用基类 reset 后绑定 `self._target`、`self._container`、语言指令、初始高度、reward state、robot reset。
- `compute_reward()` / `check_success()`：成功判定。
- `decompose()`：输出 MP 子任务序列。

例子：

- Bend pick：`third_party/SIMPLE/src/simple/tasks/g1_wholebody_bend_pick_mp.py:42-365`。
- Tabletop grasp：`third_party/SIMPLE/src/simple/tasks/g1_wholebody_tabletop_grasp_mp.py:40-405`。
- X move and pick：`third_party/SIMPLE/src/simple/tasks/g1_wholebody_x_move_and_pick_mp.py:40-451`。
- X move bend pick：`third_party/SIMPLE/src/simple/tasks/g1_wholebody_x_move_bend_pick_mp.py:45-384`。
- Franka pick and place：`third_party/SIMPLE/src/simple/tasks/franka_tabletop_pick_n_place_mp.py:39-425`。

### 2.3 DR randomizers：一局环境的输入采样器

常见 DR 配置含义：

- `LanguageDRCfg`：返回任务指令模板，目前实现取第一个 instruction，见 `third_party/SIMPLE/src/simple/dr/language.py:6-18`。
- `TargetDRCfg`：从 AssetManager 读取固定目标 asset，例如 `graspnet1b:0`，见 `third_party/SIMPLE/src/simple/dr/target.py:18-50`。
- `DistractorDRCfg`：采样干扰物，可 include/exclude，见 `third_party/SIMPLE/src/simple/dr/distractor.py:14-90`。
- `SpatialDRCfg`：给 robot/target/container/distractors/articulated object 采样位置、稳定姿态、yaw，并做 collision manager 检查，见 `third_party/SIMPLE/src/simple/dr/spatial.py:81-318`。
- `TabletopSceneDRCfg`：选择 HSSD scene、table/table2，见 `third_party/SIMPLE/src/simple/dr/scene.py:35-228`。
- `LightingDRCfg`：采样灯光数量、温度、强度、位置、朝向，见 `third_party/SIMPLE/src/simple/dr/lighting.py:18-131`。
- `MaterialDRCfg`：采样 table/ground/object/robot shader 参数，见 `third_party/SIMPLE/src/simple/dr/material.py:21-136`。

创建新任务时，任务“难不难、会不会规划成功、分布是否合理”很大程度取决于 `dr_cfgs` 的空间区域、目标姿态、干扰物数量、桌面/场景选择，而不只取决于 `decompose()`。

## 3. Task.decompose 与 MP 子任务 DSL

`SubtaskSpec` 定义在 `third_party/SIMPLE/src/simple/datagen/subtask_spec.py:5-58`。它本质上只是：

- `phase`
- `description`
- `check`
- `meta` 字典

具体类型包括：

- `OpenGripperSpec`
- `CloseGripperSpec`
- `MoveEEFToPoseSpec`
- `GraspObjectSpec`
- `LiftSpec`
- `LowerSpec`
- `RetreatSpec`
- `PhaseBreakSpec`
- `StandSpec`
- `WalkSpec`
- `TurnSpec`
- `HeightAdjustSpec`

它们没有复杂逻辑，真正执行逻辑在 `MotionPlannerAgent.synthesize()`，见 `third_party/SIMPLE/src/simple/agents/mp.py:64-599`。

### 3.1 MotionPlannerAgent 如何消费子任务

`MotionPlannerAgent` 初始化保存 task、planner、batch size、debug state，见 `agents/mp.py:45-63`。每次 `synthesize()`：

1. 调 `task.decompose()` 获取全部子任务。
2. 根据 `_subtask_index` 支持 phase break 后继续规划。
3. 对 humanoid 根据 `hand_uid` 更新 robot ee link，并 `planner.reinit(robot)`。
4. 按 spec 类型把规划轨迹或 primitive command 塞进 `PrimitiveAgent` 的 `_action_queue`。

映射关系：

- `OpenGripperSpec` / `CloseGripperSpec`：各排 10 个 open/close eef action，见 `agents/mp.py:100-105`。
- `MoveEEFToPoseSpec`：读取 `position/orientation/grasp_type/eef_state`，把 world-frame 目标转到 robot base frame；普通夹爪走 `planner.batch_plan_for_move()`，Bodex 走 `planner.batch_plan_for_move_bodex()`；结果转为 qpos path 并排入队列，见 `agents/mp.py:106-219`。
- `GraspObjectSpec`：读取 target actor pose，普通夹爪用 `GSNet.load_cached_grasps*()`，Bodex 用 `Bodex.load_cached_grasps()`；再调用 robot 的 `get_grasp_pose_wrt_robot()` 转到 robot base frame；普通夹爪走 `batch_plan_for_approach()`，Bodex 走 `batch_plan_for_approach_bodex()`，见 `agents/mp.py:220-397`。
- `LiftSpec`：普通夹爪走 `batch_plan_for_lift()`，Bodex 走 `batch_plan_for_lift_bodex()`，见 `agents/mp.py:398-425`。
- `LowerSpec`：本质是负方向 lift，见 `agents/mp.py:427-449`。
- `RetreatSpec`：Bodex 走 `batch_plan_for_retreat_bodex()`，普通夹爪复用 lift，见 `agents/mp.py:451-481`。
- `StandSpec` / `WalkSpec` / `TurnSpec` / `HeightAdjustSpec`：不走 cuRobo，而是排 `loco_command`，用于 whole-body locomotion/姿态调整，见 `agents/mp.py:483-594`。
- `PhaseBreakSpec`：立即返回 `"phase_break"`，让 `datagen.py` 先执行当前队列动作，再继续规划后续阶段，见 `agents/mp.py:80-84` 和 `cli/datagen.py:149-167`。

`PrimitiveAgent` 的队列接口在 `third_party/SIMPLE/src/simple/agents/primitive_agent.py:37-142`：

- `queue_move_qpos`
- `queue_move_qpos_with_eef`
- `queue_follow_path`
- `queue_follow_path_with_eef`
- `queue_open_gripper`
- `queue_close_gripper`
- `queue_loco_command`
- `get_action()` 从队列 popleft，空队列抛 `StopIteration`

### 3.2 代表性 decompose 模式

Bend pick：

`g1_wholebody_bend_pick_mp.py:338-365`

```python
StandSpec("initialize")
HeightAdjustSpec("adjust_height", height=-0.3, keep_waist_pose=True)
PhaseBreakSpec("phase_break_before_pick", grasp_type="bodex")
GraspObjectSpec(..., grasp_type="bodex", hand_uid="dex3_right", lock_links=["left_hand_palm_link"])
HeightAdjustSpec("adjust_height", height=0, keep_waist_pose=True)
StandSpec("end")
```

Tabletop grasp：

`g1_wholebody_tabletop_grasp_mp.py:377-405`

```python
StandSpec("initialize")
PhaseBreakSpec("phase_break_before_handover", grasp_type="bodex")
GraspObjectSpec(..., grasp_type="bodex", hand_uid="dex3_right", lock_links=[...])
LiftSpec("lift", up=0.1, grasp_type="bodex", hand_uid="dex3_right")
```

X move and pick：

`g1_wholebody_x_move_and_pick_mp.py:405-451`

```python
StandSpec("initialize")
WalkSpec("walk to target", vx=0.35, target_yaw=0, target_distance=0.45)
StandSpec("initialize")
PhaseBreakSpec("phase_break_before_pick", grasp_type="bodex")
GraspObjectSpec(...)
LiftSpec("lift", up=0.15, grasp_type="bodex", hand_uid="dex3_right")
StandSpec("initialize")
```

Franka pick and place：

`franka_tabletop_pick_n_place_mp.py:396-425`

```python
OpenGripperSpec("init")
GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False)
CloseGripperSpec("grasp")
LiftSpec("lift", up=_LIFT_HEIGHT)
MoveEEFToPoseSpec("move_to_container", position=place_position, orientation=grasp_orientation)
LiftSpec("lower", up=_LOWER_HEIGHT, step_distance=-0.01, eef_state="close_eef")
OpenGripperSpec("place")
LiftSpec("retreat", up=_LIFT_HEIGHT, eef_state="open_eef")
```

## 4. CuRoboPlanner：输入、输出与关键环节

`CuRoboPlanner` 在 `third_party/SIMPLE/src/simple/mp/curobo.py`，继承抽象接口 `MotionPlanner`，接口在 `third_party/SIMPLE/src/simple/mp/planner.py:18-48`。

### 4.1 初始化

`CuRoboPlanner.__init__()` 见 `mp/curobo.py:51-79`：

- 输入 robot，要求 robot 满足 `BatchPlannable` 等协议。
- 设置 `plan_batch_size`、`plan_dt`、`approach_duration`、`pregrasp_distance`、`lift_height`、`plan_per_traj`、`easy_motion_gen`、`ignore_target_collisions`。
- 调 `_init_robot_related(robot)`。

`_init_robot_related()` 见 `mp/curobo.py:80-130`：

- 创建 `IKSolver`：`init_js_ik_solver`。
- 创建空 world 的 `MotionGen`：`motion_gen`。
- 创建 lift IK solver：`lift_ik_solver`。
- 从 robot cfg 读取 URDF、base_link、ee_link，创建 `CudaRobotModel`。

如果 humanoid 在不同手之间切换，`MotionPlannerAgent` 会调用 `robot.update_ee_link(hand_uid)` 并 `planner.reinit(robot)`，见 `agents/mp.py:86-98`。

### 4.2 碰撞世界

`create_collision_world_cfg(layout)` 见 `mp/curobo.py:899-948`：

- 输入 SIMPLE `Layout`。
- 读取 robot pose，将 world-frame object/table pose 转成 robot-frame。
- table/box 用 cuRobo `Cuboid`。
- 物体用 cuRobo `Mesh`，mesh path 来自 `obj.asset.collision_mesh_curobo`。
- `ignore_target_collisions=True` 时跳过 target。

`batch_plan_for_approach()`、`batch_plan_for_move()` 等在 `easy_motion_gen=False` 且传入 `world_layout` 时更新碰撞世界。

### 4.3 抓取/移动/抬升规划

普通 approach：

- `batch_plan_for_approach(goal_poses, world_layout=..., approach_axis=...)`，见 `mp/curobo.py:262-380`。
- 输入 goal poses，格式可以是 dict `{position, orientation}` 或数组。
- 从 robot 采样初始规划关节状态和初始四元数。
- 调 `robot.rotate_round_approach_dir_if_needed()` 调整抓取姿态。
- 调 `motion_gen.plan_batch()`。
- 从成功结果中按最短轨迹选出 trajectory。
- 输出 `(trajs, jNames)`，其中 `trajs` 是 list，每条是 numpy qpos 序列，`jNames` 是关节名。

move：

- `batch_plan_for_move(goal_poses, current_joint_states, world_layout=...)`，见 `mp/curobo.py:510-650`。
- 输入当前 joint state 和目标 EEF pose。
- 当前 joint state 可以是 dict 或 ndarray。
- 复制成 batch，当前 EE orientation 用 FK 获取。
- 调 `motion_gen.plan_batch()`，选最短成功轨迹。
- 输出 `(trajs, jNames)`。

lift/lower：

- `batch_plan_for_lift(current_joint_states, lift_height, step_distance)`，见 `mp/curobo.py:405-456`。
- 每一步通过 FK 找当前 EE pose，把 z 加上 `step_distance`，再用 IK solver 求下一步 joint state。
- 输出 list of qpos dict；`LowerSpec` 通过负 step_distance 实现。

Bodex 变体：

- `batch_plan_for_approach_bodex()`：见 `mp/curobo.py:134-260`。
- `batch_plan_for_lift_bodex()`：见 `mp/curobo.py:458-508`。
- `batch_plan_for_move_bodex()`：见 `mp/curobo.py:652-788`。
- `batch_plan_for_retreat_bodex()`：见 `mp/curobo.py:790-897`。

Bodex 路径会处理 dexterous hand qpos、link poses、lock_links，例如锁住另一只手或腰/躯干链接。当前实现里有一些 FIXME/HACK，说明这条链路更贴近作者当时的 G1 whole-body 数据构造，而不是完全抽象化的通用 planner。

## 5. 执行动作与保存数据

### 5.1 datagen rollout

`datagen.py:89-104` 创建 planner、agent、recorder：

```python
planner = CuRoboPlanner(...)
mp_agent = MotionPlannerAgent(task, planner, ...)
env = LerobotRecorder(env=env, agent=mp_agent, ...)
```

主循环在 `datagen.py:145-176`：

1. `env.reset(options={"state_dict": state_dict})`
2. `mp_agent.synthesize()`
3. 如果返回 `"phase_break"`，执行当前 action queue。
4. 如果返回 `True`，执行最终 action queue。
5. `mp_agent.get_action()` 逐步吐出 `ActionCmd`。
6. `env.step(action)` 推进 MuJoCo/Isaac。
7. 成功终止后 recorder 保存 episode。

执行环境：

- `LocoManipulationEnv.reset()`：task reset -> Mujoco update layout -> Isaac update layout -> 返回 obs/info，见 `envs/loco_manipulation.py:61-82`。
- `LocoManipulationEnv.step()`：`mujoco.apply_action(action)` -> mujoco step -> isaac step/render -> compute reward -> check success，见 `envs/loco_manipulation.py:84-100`。
- `TabletopGraspEnv` 对 Franka/Aloha/Vega 等桌面机械臂任务类似，见 `envs/tabletop_grasp.py:105-144`。

### 5.2 LerobotRecorder 原始保存格式

`LerobotRecorder` 在 `third_party/SIMPLE/src/simple/envs/lerobot.py`。

初始化：

- 保存路径为 `<root_dir>/<env.spec.id>/level-<dr_level>`，见 `lerobot.py:85-88`。
- 根据 task action/observation space 创建 `features_dict`，见 `lerobot.py:111-167`。
- Wholebody robot 额外写 AMO policy 相关字段，见 `lerobot.py:130-160`。

step 保存：

- 图像 obs 写为 `observation.rgb_<key>`。
- 一维 obs 写为 `observation.<key>`。
- action 从 robot 控制器/actuator/PD target/hand target qpos 中取成 1D float32。
- Wholebody 额外写 `observation.amo_policy_obs_prop`、`observation.amo_policy_output_torque`、`observation.amo_policy_command`、`observation.amo_policy_rpy`、`observation.amo_policy_turning_flag`、`observation.amo_policy_target_yaw`。
- 见 `lerobot.py:193-265`。

成功过滤：

- episode `terminated or truncated` 后，只有 `reward > 0.9` 才 `dataset.save_episode()` 并写 `environment_config`；否则清空 buffer 和已写图像。
- 见 `lerobot.py:267-282`。

environment_config：

- `write_env_config()` 把 `task.state_dict()` 序列化到 `meta/episodes.jsonl` 的 `environment_config` 字段，见 `lerobot.py:289-297`。

## 6. Psi0 发布训练数据的后处理链路

本地 `/data/chenqingxi/Psi0-data/simple/G1WholebodyBendPickMP-v0/meta/info.json` 显示发布训练数据字段是：

- `states`
- 36 维 `action`
- `observation.images.egocentric`
- `observation.hand_joints`
- `observation.arm_joints`
- `observation.leg_joints`
- `observation.prev_height`
- `observation.prev_torso_rpy`
- `next.done`
- `stats_psi0.json`

这与 `LerobotRecorder` 直接生成的 `observation.rgb_*`、`observation.joint_qpos`、`observation.amo_policy_*` schema 不同。

对应后处理脚本是：

- MP wholebody 数据：`third_party/SIMPLE/scripts/postprocess_psi0.py`
- Sonic/Teleop 数据：`third_party/SIMPLE/scripts/postprocess_psi0_sonic.py`

`postprocess_psi0.py` 关键逻辑：

- 读取原始 datagen 的 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`，见 `postprocess_psi0.py:282-310`。
- 读取每个 episode parquet 中的 `observation.joint_qpos`、`observation.amo_policy_command`、`observation.amo_policy_target_yaw`、`observation.amo_policy_turning_flag`、`action`，见 `postprocess_psi0.py:319-327`。
- `build_vectors()` 把 SIMPLE wholebody proprio/action 重排成 Psi0 的 `states` 和 36D `actions`，见 `postprocess_psi0.py:113-137`。
- `build_proprio_obs()` 拆出 hand/arm/leg/torso/height observation，见 `postprocess_psi0.py:140-170`。
- 写新 parquet 字段，见 `postprocess_psi0.py:357-371`。
- 复制/下采样视频为 `videos/.../egocentric/...mp4`，见 `postprocess_psi0.py:377-390`。
- 继承原始 `environment_config` 到新 episodes metadata，见 `postprocess_psi0.py:394-403`。
- 写 `info.json`、`stats.json`、`stats_psi0.json`、`relative_stats.json`、`lang_map.json`、`modality.json`，见 `postprocess_psi0.py:447-514`。

Psi0 fine-tune 脚本明确期待这种后处理格式：

- `scripts/train/psi0/finetune-simple-psi0.sh:48-60` 使用 `/hfm/data/simple/$task`，读 `meta/stats_psi0.json`，action/state 都 pad 到 36 维。
- `src/psi/config/transform.py:359-397` 的 `SimpleRepackTransform` 读取 `observation.images.egocentric`、`states`、`action`、`task`。
- `src/psi/config/data_lerobot.py:23-39` 会从 `root_dir/train_repo_ids[0]/meta/stats_psi0.json` 加载统计量。

所以，如果新 MP 数据要用于 Psi0，不能只停在 `datagen.py` 原始输出；还需要跑或改造 postprocess。

## 7. Eval config 与 L0/L1/L2

### 7.1 eval-only dataset

`datagen.py` 的 `--eval` 模式见 `third_party/SIMPLE/src/simple/cli/datagen.py:56-87`：

1. 创建 `LerobotRecorder(agent=None)`。
2. 每个 episode 只 `env.reset()`。
3. 把当前 `task.state_dict()` 写成 `environment_config`。
4. 写一个 dummy frame，保存为 LeRobot episode。

本地 eval 数据也符合这个结构：`/data/chenqingxi/Psi0-data/simple-eval/<task>/level-0|1|2/meta/episodes.jsonl` 中每个 episode 长度为 1，核心价值是 `environment_config`，不是演示动作。

`simple.cli.eval` 读取 eval dataset：

- 对 LeRobot 格式，`LeRobotDataset(repo_id=env_id, root=data_dir)` 或 `_LocalLeRobotDataset` 读取本地数据，见 `third_party/SIMPLE/src/simple/cli/eval.py:202-214` 和 `third_party/SIMPLE/src/simple/evals/env_runner.py:391-407`。
- `get_episode_lerobot()` 从 `dataset.meta.episodes[eps_idx]['environment_config']` 解析 env config，见 `third_party/SIMPLE/src/simple/datasets/lerobot.py:9-24`。
- eval 时 `env.reset(options={"state_dict": env_conf})` 复现环境，见 `simple/cli/eval.py:262-278` 和 `evals/env_runner.py:282-313`。

### 7.2 L0/L1/L2 的语义

SIMPLE README 明确给出三层语义，见 `third_party/SIMPLE/README.md:225-232`：

- Level 0：Visual & Distractors，随机 table material 和干扰物类型/初始位置。
- Level 1：Lighting，在 Level 0 基础上加入极端灯光变化。
- Level 2：Spatial pose，在 Level 1 基础上扰动目标物体初始位置。

代码里最接近这一语义的是 `DRManager.load_state_dict(state_dict, dr_level=...)`，见 `third_party/SIMPLE/src/simple/dr/manager.py:71-121`：

- `dr_level is None`：完整复现 state_dict。
- `dr_level == 0`：从原配置中移除 `distractors`、`material`，因此保留 scene/spatial/lighting/target，但重新随机干扰物和材质。
- `dr_level == 1`：进一步移除 `lighting`，因此重新随机干扰物、材质、灯光。
- `dr_level == 2`：进一步处理 `spatial`，只保留 robot spatial，目标/物体 spatial 重新随机，同时干扰物、材质、灯光也重新随机。

这和 README 的 Level 0/1/2 描述一致。`third_party/SIMPLE/src/simple/cli/dr_decoupled_wbc.py:155-158` 展示了用 `env.reset(options={"state_dict": env_conf, "dr_level": dr_level})` 从已有 episode config 派生不同 DR level 的方式。

本地样例也支持这个判断：

- `simple-eval/G1WholebodyBendPickMP-v0/level-0` 和 `level-1` 的 target xy 集合相同，但 lighting/material/distractor 不同。
- `level-2` 的 target xy 集合发生变化。
- 每个 level 都是 10 个 eval episode，每个 episode 只有 1 frame。

注意：当前 `datagen.py --eval` 是“直接 reset 并保存当前 task.state_dict()”，没有从一个 base config 显式调用 `load_state_dict(..., dr_level=...)`。如果你想严格复刻官方 L0/L1/L2 的“同一 base episode 派生不同 OOD 变化”，应按 `DRManager.load_state_dict` 的语义生成，而不是简单地对三个 `--dr-level` 各自独立随机采样后保存。

## 8. 创建新 MP 任务的数据构造方案

### 8.1 判断任务是否适合 MP

适合复用 MP 流程的任务：

- 可以分解成“站立/行走/转向/高度调整/抓取/抬升/移动到 pose/放置/后退”等阶段。
- 目标物体是可抓取物体，已有 stable poses、collision mesh、canonical/cached grasps，或者 Bodex 能加载对应抓取。
- 成功标准能通过仿真 state/contact/reward 判定。
- 不依赖需要人类探索、复杂接触、工具使用、多步非几何推理的动作。

不太适合直接 MP 的任务：

- 开抽屉、旋转开关、柔性物体、长时接触推动、需要力控闭环的任务。
- 抓取姿态/接触点不能由 GSNet/Bodex 稳定给出。
- 需要对失败接触进行在线补救，而现有 `MotionPlannerAgent` 基本是离线排队动作。

### 8.2 最小新增代码点

后续真正开发时，通常需要改这些点：

1. 新建一个 task 文件：`third_party/SIMPLE/src/simple/tasks/<new_task>_mp.py`。
2. 在类上加 `@TaskRegistry.register("<new_task>_mp")`。
3. 在 `third_party/SIMPLE/src/simple/tasks/__init__.py` import 新类，确保注册执行。
4. 在 `third_party/SIMPLE/src/simple/envs/__init__.py` 新增 Gym `register(id="simple/NewTaskMP-v0", entry_point=..., kwargs={"task": "<new_task>_mp"})`。
5. 如果只是 G1 wholebody loco-manip，优先复用 `LocoManipulationEnv`；如果是桌面机械臂，优先复用 `TabletopGraspEnv`。
6. 如果数据要用于 Psi0，确认 `postprocess_psi0.py` 是否支持新任务的 raw schema、视频 key、episode 长度、action/state 维度；必要时只改后处理脚本或新增一个专用后处理脚本。

### 8.3 新 task 类内部要设计的内容

新 task 类建议从最接近的现有 MP task 复制思想，而不是从零写：

- `uid/label/description`：保持和 Gym id、数据目录命名清晰对应。
- `metadata`：`physics_dt`、`render_hz`、`max_episode_steps` 要和控制/AMO 频率匹配。G1 wholebody 任务常见 `physics_dt=0.002`，`render_hz=30` 或 `50`。
- `robot_cfg`：先复用 `g1_wholebody`，避免同时引入新 robot 风险。
- `sensor_cfgs`：如果训练 Psi0，至少要保证 postprocess 使用的视频 key 存在。
- `dr_cfgs`：
  - `target`/`container` 选择 asset。
  - `distractors` 设置数量和 exclude。
  - `spatial` 设置 robot/target/container/distractor 区域、稳定姿态、旋转范围。
  - `scene` 固定或采样 HSSD 场景和 table/table2。
  - `lighting`/`material` 作为视觉 domain randomization。
  - `language` 给出指令模板。
- `reset()`：绑定 `_target`、`_container`，生成 instruction，重置 reward 状态和 robot。
- `compute_reward()` / `check_success()`：必须能真实反映成功，否则 recorder 会保存错误数据或丢掉正确数据。
- `decompose()`：把任务变成现有 `SubtaskSpec` 序列。

### 8.4 新任务的 L0/L1/L2 设计方法

建议把难度设计拆成两层：

第一层是训练数据 randomization：

- 训练集通常用 `dr_level=0` 或你自定义的 train 分布，保证 MP 成功率足够高。
- 先让 target/container 空间区域窄一些，调通规划和成功检测；再逐步增大 region。

第二层是 eval OOD levels：

- Base config：生成一组基础 eval episode，固定目标任务结构。
- Level 0：保留 scene/target spatial/lighting，重新随机 material 和 distractors。
- Level 1：在 Level 0 基础上重新随机 lighting。
- Level 2：在 Level 1 基础上重新随机 target/container/distractor spatial，但保留 robot 初始 spatial，防止任务完全变成不同导航问题。

实现上应复用 `DRManager.load_state_dict(..., dr_level=0/1/2)` 的逻辑。这样更接近官方 README 的 L0/L1/L2 定义。

### 8.5 推荐执行链路

开发新任务时建议按这个顺序：

1. 只写 task/env 注册，先让 `gym.make("simple/NewTaskMP-v0", sim_mode="mujoco")` 能 reset。
2. 检查 `task.layout` 中 robot、target、container、distractors、table/cameras 是否齐全。
3. 单 episode、headless、低 episode 数运行 datagen，观察是否能生成成功 episode：

```bash
python -m simple.cli.datagen \
  simple/NewTaskMP-v0 \
  --sim-mode mujoco_isaac \
  --headless \
  --data-format lerobot \
  --save-dir /path/to/raw_datagen \
  --num-episodes 1 \
  --shard-size 1 \
  --dr-level 0 \
  --debug
```

4. 如果规划失败，优先缩小空间随机范围、减少干扰物、固定 stable pose、检查目标物体抓取缓存和 collision mesh。
5. 如果规划成功但保存不到 episode，检查 `compute_reward()` / `check_success()`。
6. 批量生成 raw datagen。
7. 用 `postprocess_psi0.py` 转成 Psi0 训练格式。例如：

```bash
python third_party/SIMPLE/scripts/postprocess_psi0.py \
  --sim-root "/path/to/raw_datagen/simple/NewTaskMP-v0/level-0" \
  --out-dir "/path/to/psi0_simple/NewTaskMP-v0" \
  --total_episodes 100 \
  --fps 50 \
  --video-key observation.rgb_head_stereo_left
```

实际 `--video-key` 必须与 raw datagen `meta/info.json` 里的 feature key 对齐。

8. 生成 `simple-eval/NewTaskMP-v0/level-0|1|2`，每个 level 建议至少 10 个 episode，保存 `environment_config`。
9. 用 `simple.cli.eval` 复现 env config，先用 replay/MP 或 dummy policy 做 smoke test，再接 Psi0/baseline。

## 9. 风险与验证清单

必须优先验证：

- **注册链路**：`simple/NewTaskMP-v0` 是否能 `gym.make`，`TaskRegistry` 是否能找到 task uid。
- **资产链路**：目标/容器是否有 collision mesh、stable poses、grasps；`collision_mesh_curobo` 路径是否存在。
- **空间采样**：target/container/distractors 是否不穿模、不超出机械臂/手可达范围、不挡住必要路径。
- **cuRobo 可行性**：`GraspObjectSpec` 是否能找到 grasp，`batch_plan_for_approach/move/lift` 是否成功率足够高。
- **batch size 语义**：当前 CLI 的 `--plan-batch-size` 主要影响 `MotionPlannerAgent` 的候选 grasp 数量，`CuRoboPlanner` 初始化仍固定为 1；调参时不要误以为已经打开 planner 内部批量规划。
- **phase break**：wholebody 任务中 locomotion/height adjust 后再抓取时，是否需要 `PhaseBreakSpec` 先执行队列动作。
- **成功判定**：reward 是否能在真实完成时达到阈值，失败时不会误判。
- **数据保存**：raw LeRobot 的 `meta/episodes.jsonl` 是否包含 `environment_config`，parquet/video 是否完整。
- **后处理 schema**：Psi0 训练脚本需要的 `states`、`action`、`observation.images.egocentric`、`stats_psi0.json` 是否存在。
- **eval 复现**：`simple.cli.eval` 能否从 `data-dir level-x` reset 到相同 env config。
- **L0/L1/L2 对齐**：Level 0/1/2 的变化是否分别对应视觉/干扰物、灯光、空间扰动，而不是三个完全独立随机数据集。

## 10. 对“能否在官方代码基础上修改来生成不同任务 MP 数据”的回答

可以。官方代码已经把 MP 数据构造拆成可复用组件：task 定义负责场景/奖励/分解，`MotionPlannerAgent` 负责把 `SubtaskSpec` 翻译成 planner/primitive action，`CuRoboPlanner` 负责几何规划，`LerobotRecorder` 负责保存数据。

但二次开发最稳的路径不是改 `CuRoboPlanner` 或 `MotionPlannerAgent`，而是先新增一个与现有 G1 MP 任务同构的 task：改 `dr_cfgs`、`compute_reward()`、`decompose()` 和注册表。只有当新任务需要现有 DSL 无法表达的新 primitive 时，才考虑扩展 `SubtaskSpec` 和 `MotionPlannerAgent`。

如果最终要产出 Psi0 可训练数据，还必须把 raw datagen 产物通过 `postprocess_psi0.py` 这层转成发布数据的 schema；否则即使 MP rollout 生成成功，也不等价于官方 `psi-data/simple` 中用于 fine-tune 的数据。
