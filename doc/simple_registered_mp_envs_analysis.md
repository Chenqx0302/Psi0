# SIMPLE 已注册 MP 环境清单与可用性分析

本文只基于源码和本地已下载的 Psi0/SIMPLE 数据目录做静态阅读分析，没有运行仿真、没有生成数据、没有修改源码。

## 1. 结论

SIMPLE 代码里确实注册了大量 MP 环境。`third_party/SIMPLE/src/simple/envs/__init__.py` 一共注册了 82 个 Gym env，其中 72 个是 `*MP-v0`，10 个是 `*Teleop-v0`。

Psi0 公开 SIMPLE 数据目录里当前只有 6 个任务：

- `G1WholebodyBendPickMP-v0`
- `G1WholebodyTabletopGraspMP-v0`
- `G1WholebodyXMovePickTeleop-v0`
- `G1WholebodyXMoveBendPickTeleop-v0`
- `G1WholebodyHandoverTeleop-v0`
- `G1WholebodyLocomotionPickBetweenTablesTeleop-v0`

因此，按 Psi0 发布的 `psi-data/simple` 和 `psi-data/simple-eval` 六任务集合计算，72 个已注册 MP 环境里只有 2 个有公开训练/评测数据，另外 70 个 MP 环境没有对应公开 Psi0 SIMPLE 数据。

这些未公开数据的 MP 环境不能简单理解为“已经是可直接训练的 benchmark”。源码层面看，多数都有 `TaskRegistry.register()`、Gym env 注册、`decompose()`、`compute_reward()` 和 `check_success()`，可以作为 `simple.cli.datagen` 的候选入口；但是否能稳定生成训练数据，还必须逐个做 reset、规划成功率、reward 成功过滤、后处理 schema 和 eval config 验证。

## 2. 代码依据

主要依据：

- Env 注册：`third_party/SIMPLE/src/simple/envs/__init__.py`
- Task 导入：`third_party/SIMPLE/src/simple/tasks/__init__.py`
- Task 注册与实例化：`third_party/SIMPLE/src/simple/core/registry.py`
- Datagen 入口：`third_party/SIMPLE/src/simple/cli/datagen.py`
- MP agent：`third_party/SIMPLE/src/simple/agents/mp.py`
- MP task 文件：`third_party/SIMPLE/src/simple/tasks/*_mp.py`
- 本地公开数据目录：
  - `/data/chenqingxi/Psi0-data/simple`
  - `/data/chenqingxi/Psi0-data/simple-eval`

## 3. 数量统计

| 类别 | 数量 | 说明 |
|---|---:|---|
| Gym env 注册总数 | 82 | `envs/__init__.py` 中的 `register(...)` |
| MP env | 72 | env id 以 `MP-v0` 结尾 |
| Teleop env | 10 | env id 以 `Teleop-v0` 结尾 |
| Psi0 公开 SIMPLE 任务 | 6 | 本地 `simple` 和 `simple-eval` 均为这 6 个 |
| Psi0 公开 MP 任务 | 2 | `BendPickMP`、`TabletopGraspMP` |
| 已注册但无 Psi0 公开数据的 MP env | 70 | 72 个 MP env 减去 2 个公开 MP env |

## 4. 可直接性的判断

### 4.1 能否直接用来生成训练 rollout

大多数已注册 MP env 可以作为 `simple.cli.datagen` 的候选入口，因为它们具备：

- Gym env 注册。
- Task uid 注册。
- `decompose()` 输出 MP 子任务。
- `compute_reward()` / `check_success()` 用于 recorder 成功过滤。

但这只是“源码入口具备”，不是“数据质量已验证”。真正批量生成训练数据前需要逐个验证：

- `gym.make("simple/<EnvId>")` 能否 reset。
- 所需 robot、scene、asset、collision mesh、stable poses、GSNet/Bodex grasp cache 是否齐全。
- cuRobo 规划成功率是否可接受。
- `check_success()` 是否不会误判。
- `LerobotRecorder` 是否能保存完整 parquet/video/env config。
- 如果目标是 Psi0 fine-tune，raw datagen 数据是否能通过 `postprocess_psi0.py` 转成 Psi0 schema。

### 4.2 能否直接作为测试 benchmark

可以用 `datagen.py --eval` 为候选 env 生成 eval-only dataset，但这还不等价于官方论文式 L0/L1/L2 benchmark。

原因是官方 L0/L1/L2 语义来自 `DRManager.load_state_dict(..., dr_level=0/1/2)`：Level 0 变视觉/干扰物，Level 1 再变灯光，Level 2 再变目标空间位置。`datagen.py --eval` 当前只是独立 reset 并保存 `task.state_dict()`，如果要严格复刻官方 L0/L1/L2，应从同一批 base configs 派生三层环境，而不是三个 level 各自独立随机采样。

### 4.3 能否直接用于 Psi0 训练

不能一概而论。

G1 wholebody MP 任务更接近公开 SIMPLE MP 数据链路，因为 recorder 会保存 wholebody/AMO 相关字段，`postprocess_psi0.py` 也按 G1 wholebody 的 proprio/action 布局构造 `states` 和 36D `action`。

Franka/Aloha/Vega/G1 tabletop/G1 Inspire 等任务虽然是 MP env，但它们的 robot、action space、observation schema 不一定匹配 Psi0 的 G1 wholebody 后处理脚本。要用于 Psi0，通常需要单独检查或改造后处理脚本、训练 transform 和 stats schema。

## 5. 明显不能直接使用的 MP env

源码静态检查发现两个 `FindNGrasp` MP task 没有实现 `decompose()`，而基类 `Task.decompose()` 会直接 `raise NotImplementedError`：

- `AlohaTabletopFindNGraspMP-v0`
- `VegaTabletopFindNGraspMP-v0`

另外，`vega_tabletop_find_n_grasp_mp.py` 虽然有 `@TaskRegistry.register("vega_tabletop_find_n_grasp_mp")`，但没有在 `third_party/SIMPLE/src/simple/tasks/__init__.py` 中导入；按当前注册机制，未导入模块的装饰器不会执行，因此这个 env 还存在运行时 task registry 缺失风险。

## 6. 全量 MP env 清单

`Psi0 public data` 指是否出现在本地 `/data/chenqingxi/Psi0-data/simple` 和 `/data/chenqingxi/Psi0-data/simple-eval` 六任务集合中。

| MP env id | task uid | family | Psi0 public data | direct datagen status |
|---|---|---|---|---|
| `FrankaTabletopGraspMP-v0` | `franka_tabletop_grasp_mp` | tabletop arm | no | candidate: has decompose()+success check |
| `FrankaTabletopPickNPlaceMP-v0` | `franka_tabletop_pick_n_place_mp` | tabletop arm | no | candidate: has decompose()+success check |
| `AlohaTabletopGraspMP-v0` | `aloha_tabletop_grasp_mp` | tabletop arm | no | candidate: has decompose()+success check |
| `AlohaTabletopHandoverMP-v0` | `aloha_tabletop_handover_mp` | tabletop arm | no | candidate: has decompose()+success check |
| `VegaTabletopGraspMP-v0` | `vega_tabletop_grasp_mp` | tabletop arm | no | candidate: has decompose()+success check |
| `AlohaTabletopFindNGraspMP-v0` | `aloha_tabletop_find_n_grasp_mp` | tabletop arm | no | blocked: no decompose() |
| `VegaTabletopFindNGraspMP-v0` | `vega_tabletop_find_n_grasp_mp` | tabletop arm | no | blocked: no decompose(); also not imported in tasks/__init__.py |
| `G1TabletopGraspMP-v0` | `g1_tabletop_grasp_mp` | G1 tabletop | no | candidate: has decompose()+success check |
| `G1TabletopPickNPlaceMP-v0` | `g1_tabletop_pick_n_place_mp` | G1 tabletop | no | candidate: has decompose()+success check |
| `G1InspireTabletopGraspMP-v0` | `g1_inspire_tabletop_grasp_mp` | G1 tabletop | no | candidate: has decompose()+success check |
| `G1TabletopHandoverMP-v0` | `g1_tabletop_handover_mp` | G1 tabletop | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionMP-v0` | `g1_wholebody_locomotion_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyPickNPlaceMP-v0` | `g1_wholebody_pick_n_place_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesMP-v0` | `g1_wholebody_locomotion_pick_between_tables_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodySitMP-v0` | `g1_wholebody_sit_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickAndPlaceMP-v0` | `g1_wholebody_bend_pick_and_place_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickMP-v0` | `g1_wholebody_bend_pick_mp` | G1 wholebody | yes | candidate: has decompose()+success check |
| `G1WholebodyBendHandoverMP-v0` | `g1_wholebody_bend_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickAndPlaceOnSofaMP-v0` | `g1_wholebody_bend_pick_and_place_on_sofa_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTabletopGraspMP-v0` | `g1_wholebody_tabletop_grasp_mp` | G1 wholebody | yes | candidate: has decompose()+success check |
| `G1InspireWholebodyLocomotionMP-v0` | `g1_inspire_wholebody_locomotion_mp` | G1 Inspire wholebody | no | candidate: has decompose()+success check |
| `G1InspireWholebodyPickNPlaceMP-v0` | `g1_inspire_wholebody_pick_n_place_mp` | G1 Inspire wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant1MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant2MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyPickNPlaceVariant1MP-v0` | `g1_wholebody_pick_n_place_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant3MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant3_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnPickMP-v0` | `g1_wholebody_turn_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant4MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant4_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndPickMP-v0` | `g1_wholebody_x_move_and_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndPickNPlaceMP-v0` | `g1_wholebody_x_move_and_pick_n_place_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveBendPickMP-v0` | `g1_wholebody_x_move_bend_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveBendPickNPlaceMP-v0` | `g1_wholebody_x_move_bend_pick_n_place_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveAndPickMP-v0` | `g1_wholebody_y_move_and_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndPickVariant1MP-v0` | `g1_wholebody_x_move_and_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant5MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant5_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndHandoverMP-v0` | `g1_wholebody_x_move_and_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant6MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant6_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveAndHandoverMP-v0` | `g1_wholebody_y_move_and_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveBendPickMP-v0` | `g1_wholebody_y_move_bend_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyPickAndBendPlaceMP-v0` | `g1_wholebody_pick_and_bend_place_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant7MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant7_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndPickVariant2MP-v0` | `g1_wholebody_x_move_and_pick_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant8MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant8_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickVariant1MP-v0` | `g1_wholebody_bend_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant9MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant9_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnPickVariant1MP-v0` | `g1_wholebody_turn_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTabletopGraspVariant1MP-v0` | `g1_wholebody_tabletop_grasp_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickAndPlaceOnSofaVariant1MP-v0` | `g1_wholebody_bend_pick_and_place_on_sofa_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyBendPickAndPlaceOnSofaVariant2MP-v0` | `g1_wholebody_bend_pick_and_place_on_sofa_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveAndPickVariant1MP-v0` | `g1_wholebody_y_move_and_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant10MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant10_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant11MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant11_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndPickVariant3MP-v0` | `g1_wholebody_x_move_and_pick_variant3_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant12MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant12_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveAndHandoverVariant1MP-v0` | `g1_wholebody_x_move_and_handover_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveAndPickVariant2MP-v0` | `g1_wholebody_y_move_and_pick_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTabletopGraspVariant2MP-v0` | `g1_wholebody_tabletop_grasp_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndPickMP-v0` | `g1_wholebody_turn_x_move_and_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndPickVariant1MP-v0` | `g1_wholebody_turn_x_move_and_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndPickVariant2MP-v0` | `g1_wholebody_turn_x_move_and_pick_variant2_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyXMoveBendHandoverMP-v0` | `g1_wholebody_x_move_bend_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnYMoveAndPickMP-v0` | `g1_wholebody_turn_y_move_and_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnYMoveAndPickVariant1MP-v0` | `g1_wholebody_turn_y_move_and_pick_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndHandoverMP-v0` | `g1_wholebody_turn_x_move_and_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndHandoverVariant1MP-v0` | `g1_wholebody_turn_x_move_and_handover_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndBendPickMP-v0` | `g1_wholebody_turn_x_move_and_bend_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnYMoveAndBendPickMP-v0` | `g1_wholebody_turn_y_move_and_bend_pick_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTurnXMoveAndBendHandoverMP-v0` | `g1_wholebody_turn_x_move_and_bend_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyYMoveAndHandoverVariant1MP-v0` | `g1_wholebody_y_move_and_handover_variant1_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTabletopGraspVariant3MP-v0` | `g1_wholebody_tabletop_grasp_variant3_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyTabletopHandoverMP-v0` | `g1_wholebody_tabletop_handover_mp` | G1 wholebody | no | candidate: has decompose()+success check |
| `G1WholebodyLocomotionPickBetweenTablesVariant13MP-v0` | `g1_wholebody_locomotion_pick_between_tables_variant13_mp` | G1 wholebody | no | candidate: has decompose()+success check |

## 7. 建议优先级

如果后续目标是“不用 VR，尽快复用 MP 生成新 Psi0 风格数据”，建议优先从 G1 wholebody 且与公开 MP 任务最接近的环境开始：

- `G1WholebodyBendPickVariant1MP-v0`
- `G1WholebodyTabletopGraspVariant1/2/3MP-v0`
- `G1WholebodyXMoveAndPickMP-v0`
- `G1WholebodyXMoveBendPickMP-v0`
- `G1WholebodyYMoveAndPickMP-v0`
- `G1WholebodyTurnXMoveAndPickMP-v0`
- `G1WholebodyTurnYMoveAndPickMP-v0`

原因是这些任务和公开的 G1 wholebody MP schema 最接近，且已经有 `decompose()`、成功判定和 MP subtask 链路。`PickNPlace`、`Handover`、`LocomotionPickBetweenTables`、`Sofa`、`Sit` 等任务也有入口，但涉及更长时序、更多场景假设或更强的成功判定风险，建议放在第二批。

Franka/Aloha/Vega/G1 tabletop/G1 Inspire 任务可以用于研究 SIMPLE 自身的 MP 数据生成，但不应默认当作 Psi0 的 G1 wholebody 训练数据；这些任务需要先确认后处理和模型输入输出 schema。
