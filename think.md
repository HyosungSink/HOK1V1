## 一、整体变化概览

- 新增一套完整的自定义算法实现 `agent_diy/*`，包括 Agent、特征处理、奖励函数、模型与训练流程。
- 原有 `agent_ppo/*` 作为基线 PPO 实现基本保持不变，仅在配置中增加了一个 Dual-Clip 相关常量。
- 全局配置修改为默认使用 `diy` 算法，并调整模型保存/评估相关参数。
- 新增一份文档 `second_commit_detail.md`，用于记录第二次提交（`4ae829e..d989588`）的详细改动。

---

## 二、新增模块：agent_diy/*

### 2.1 agent_diy/agent.py

- 新增 `Agent` 类，继承自 `BaseAgent`，主要区别于 `agent_ppo.Agent`：
  - 使用 `agent_diy.model.Model` 作为模型。
  - 引入生存策略状态机：`normal / need_retreat / retreating / recalling / recovering`。
  - 维护多种与生存相关的运行时信息：`last_hp_ratio`、`recall_cooldown`、`in_combat`、`enemy_distance`、动作历史等。
- 在推理流程中增加多层生存策略逻辑：
  - `enhance_observation`：为观测补充生存特征（血量分段、战斗状态、回城冷却等）。
  - `apply_survival_strategy`：在模型输出动作后再做一层“安全过滤”，根据血量和战斗状态决定是否强制回城/撤退/防守。
  - `action_process` 中实现详细策略：
    - 极低血量时（例如 HP<20% 且已脱战）强制回城。
    - HP<40% 且附近有血包时，优先向血包移动。
    - HP<30% 时触发撤退状态，优先使用 2 技能位移撤退，其次向泉水方向移动。
    - “recovering” 状态下要求 HP>90% 且仍在泉水附近才允许离开，避免刚回城就走的问题。
    - 敌方英雄进入普攻距离（约 4500）时优先平 A 英雄；在血量劣势时采用保守拉扯策略。
    - 新增“技能后强制接一次普攻”的逻辑：若上一帧放出技能，本帧优先将主动作位设置为普通攻击。
- `observation_process / extract_hero_state / calculate_distances / extract_survival_features`：
  - 从原始观测中解析英雄当前血量、位置、技能冷却、敌方英雄和塔的位置，以及泉水/血包的距离与方向。
  - 统一封装为可被策略与生存逻辑使用的高层特征。
- 与 `agent_ppo.Agent` 的差异关键点：
  - `agent_ppo.Agent` 主要封装标准 PPO 推理与训练流程，几乎不包含显式“战术/生存规则”；
  - `agent_diy.Agent` 在模型前后增加大量启发式逻辑，更偏“策略+模型”结合。

> 注意：当前 `agent_diy.Agent.__init__` 中调用 `Algorithm(self.model, self.device, self.logger, self.monitor)`，  
> 而 `agent_diy.algorithm.Algorithm.__init__` 定义为 `(model, optimizer, scheduler, device=None, logger=None, monitor=None)`，  
> 与 `agent_ppo.Agent` 中的调用方式不同，尚未显式创建并传入 `optimizer/scheduler`，这一点与 init 版本相比是新增但尚未收敛的接口差异。

### 2.2 agent_diy/conf/conf.py

- 新增 `GameConfig`，重写奖励权重与生存相关配置：
  - 奖励项包括：`hp_point`、`money`、`exp`、`last_hit`、`minion_damage`、`kill`、`death`、`ep_rate`、`forward`、`tower_hp_point`、
    `damage_exchange`、`skill_hit`、`recall_decision`、`health_management`、`combat_timing`、`tower_defense`、`positioning` 等。
  - 增加分阶段动态权重 `ADAPTIVE_WEIGHT`：
    - early_game（<5000 帧）：放大奖励发育（money/exp）和生存（recall_decision/health_management），降低死亡惩罚系数。
    - mid_game：强化塔防、伤害交换和战斗时机。
    - late_game：显著放大推塔和前压奖励，适度增强击杀权重。
  - 回城/战斗相关阈值封装在 `RECALL_CONFIG`、`COMBAT_CONFIG` 中，用于 Agent 的生存状态机。
- 新增 `DimConfig` 与 `Config`：
  - 维持与 `agent_ppo.conf.Config` 相同的数据维度配置（DATA_SPLIT_SHAPE、SERI_VEC_SPLIT_SHAPE 等）。
  - 调整学习率：`INIT_LEARNING_RATE_START = 8e-4`, `TARGET_LR = 8e-5`（比初始 PPO 略小）。
  - 引入 `DUAL_CLIP_PARAM_C = 0.75` 作为 diy 算法的 Dual-Clip PPO 参数。
  - 增加动作映射 `ACTION_MAPPING` 和英雄技能配置 `HERO_SKILL_CONFIG`，为 Agent 中的策略逻辑（普攻/技能/回城/no_op 等）提供统一的离散动作索引。

### 2.3 agent_diy/feature/definition.py

- 定义与原 PPO 相似的数据结构：
  - `SampleData` / `ObsData` / `ActData` / `FrameCollector` 等。
  - 提供 `sample_process`、`build_frame`、`SampleData2NumpyData`、`NumpyData2SampleData` 等工具函数。
- 与 init 版本（`agent_ppo.feature.definition`）相比的关键差异：
  - 依赖 `agent_diy.conf.Config` 的 `data_shapes`、`GAMMA`、`LAMDA` 等配置；为 diy 算法单独维护一套超参。
  - `build_frame` 中将 `obs_data.feature` 作为主特征来源（若为空则回退到环境原始 observation），并使用 `Config.LABEL_SIZE_LIST` 推导 target-action 部分的合法动作掩码。
  - `FrameCollector` 负责：
    - 进行 GAE（广义优势估计）计算，填充每帧的 `advantage` 与 `reward_sum`。
    - 按 `LSTM_TIME_STEPS` 拼接样本，附带 LSTM 状态，最终输出为 `SampleData`（适配模型的输入期望）。

### 2.4 agent_diy/feature/reward_process.py

- 新增 `GameRewardManager`，基于 `GameConfig.REWARD_WEIGHT_DICT` 维护多路奖励：
  - 对英雄 HP、塔 HP、金币、经验、小兵伤害等基础指标打分。
  - 对击杀、死亡、推进、塔防、技能命中、生存决策等行为给予额外奖励或惩罚。
- 引入多种“生存/战术”相关的内部状态：
  - 追踪主角、敌方英雄、小兵和防御塔的 HP 变化，用于计算 `minion_damage`、`tower_hp_point` 等。
  - 保存前一帧位置、战斗状态、回城冷却等信息，为生存逻辑（例如回城奖励）提供上下文。
- `result(frame_data)`：
  - 先调用 `frame_data_process`、`get_reward` 计算基础奖励字典。
  - 根据当前 `frameNo` 应用 `ADAPTIVE_WEIGHT` 动态调整各奖励项权重。
  - 若配置了 `TIME_SCALE_ARG`，再做时间衰减。
- 新增的重点奖励逻辑包括（简要按功能归类）：
  - 清线与兵线管理：追踪小兵 HP 和距离，对合理的线权与站位给予奖励。
  - 推塔与塔防：根据敌塔 HP 下降量和站位（是否在兵线身后）计算额外奖励。
  - 回城与血量管理：根据回城是否及时、血量是否在安全区间等指标给出正负反馈。
  - 战斗节奏控制：通过 `damage_exchange`、`skill_hit`、`combat_timing` 等指标鼓励高效换血与合理开战时机。

### 2.5 agent_diy/model/model.py

- 基于 init 版本的 PPO 模型结构，实现一个多头策略+价值网络，主要变化在损失函数与配置来源：
  - 使用 `agent_diy.conf.Config` 中的维度、超参。
  - 在 `__init__` 中增加 `self.dual_clip_c = Config.DUAL_CLIP_PARAM_C`。
- 前向网络结构与原框架一致：
  - 将英雄/士兵/防御塔/全局信息分块编码，通过多层 MLP 与 pooling 得到汇总向量。
  - 经过公共 MLP + LSTM，输出多任务策略头与价值头。
- `compute_loss` 中实现 Dual-Clip PPO：
  - 在标准 PPO ratio-clipping 的基础上，引入 `dual_clip_surr = torch.clamp(ratio, 1 - c, 1 + c) * A_t`。
  - 对 `A_t < 0` 的样本使用 `max(standard_clip_surr, dual_clip_surr)`，以抑制过度更新同时兼顾收敛速度。
  - 保留原有值函数损失与熵正则项，并通过 `Config.USE_GRAD_CLIP` 配合 Algorithm 侧的梯度裁剪控制训练稳定性。

### 2.6 agent_diy/algorithm/algorithm.py

- 与 init 版本的 `agent_ppo.algorithm.Algorithm` 相同的基本职责：
  - 解析 `SampleData`，拆分成特征、标签、旧策略概率、权重、LSTM 状态等。
  - 前向计算、反向传播、梯度裁剪与学习率调度。
- 新增的监控逻辑：
  - `reward_stats`：维护多种奖励项的滑动窗口（长度 `stats_window_size`），用于统计 mean/std/max。
  - `update_reward_stats` / `get_reward_statistics`：供外部调用更新并汇总奖励统计。
  - 每隔 60 秒通过 `monitor` 上报损失与关键奖励统计，并用 `logger` 打印详细训练状态（包括当前学习率）。
  - `should_save_model`：一个基于奖励统计的简单“性能提升”判定逻辑（当前未与 workflow 强绑定，可作为后续模型保存策略的扩展点）。

### 2.7 agent_diy/workflow/train_workflow.py

- 新增 `workflow(envs, agents, logger, monitor)`：
  - 从 `agent_diy/conf/train_env_conf.toml` 读取并校验环境配置（使用 `read_usr_conf` / `check_usr_conf`）。
  - 通过 `ModelFileSync` 与 modelpool 同步最新模型文件。
  - 内部调用 `run_episodes`，循环收集样本并周期性保存模型（周期 `GameConfig.MODEL_SAVE_INTERVAL`）。
- `run_episodes` 与 init 版本的 workflow 结构相似，但使用 diy 的奖励函数和样本构建逻辑：
  - 每局对战前，按配置决定训练方与对手类型，并为训练方加载最新模型。
  - 使用 `GameRewardManager` 为每一帧生成 reward 字典，并累积用于日志/监控。
  - 通过 `FrameCollector` 收集对局数据，在对局结束后调用 `sample_process` 生成训练样本并 `yield` 给 learner。

### 2.8 agent_diy/conf/train_env_conf.toml

- 为 diy 算法单独提供的环境配置：
  - `monitor.monitor_side = -1`：默认自动换边监控。
  - `episode.opponent_agent = "selfplay"`，`eval_opponent_type = "common_ai"`。
  - 蓝红双方阵容均配置为单英雄鲁班七号（hero_id=112）。

---

## 三、已有模块的净变化（相对 init）

### 3.1 agent_ppo/conf/conf.py

- 在 `Config` 中新增常量：

```python
CLIP_PARAM = 0.2

# --- 已添加 ---
# 对应论文中的 c 值
DUAL_CLIP_PARAM_C = 3.0
# --- 修改结束 ---
```

- 该常量当前仅在 `agent_diy` 侧被实际使用（通过 `agent_diy.conf.Config`；`agent_ppo` 的模型/算法代码中未引用）。
- 其他字段（SERI_VEC_SPLIT_SHAPE、DATA_SPLIT_SHAPE、SAMPLE_DIM 等）与 init 版本保持一致。

### 3.2 .gitignore

- 删除了 `agent_diy` 忽略项：

```diff
 *.pyc
-agent_diy
 .vscode/
```

- 允许将 `agent_diy` 目录纳入版本控制，以承载自定义算法实现。

### 3.3 conf/configure_app.toml

- 将默认算法从 `ppo` 改为 `diy`：

```diff
-algo = "ppo"
+algo = "diy"
```

- 表示当前工程默认以新加入的 `agent_diy` 为主算法。

### 3.4 conf/kaiwudrl/configure.toml

- 调整用户模型保存策略：

```diff
-user_save_mode_max_count = 0
-user_save_model_max_frequency_per_min = 0
+user_save_mode_max_count = 400
+user_save_model_max_frequency_per_min = 2
```

- 含义：
  - 从“无限制”改为“最多 400 次保存，每分钟最多 2 次”；
  - 有助于在频繁试验自定义算法时控制模型文件数量与保存频率。

### 3.5 kaiwu.json

- 从空模型池变为包含一个固定模型 ID：

```diff
-{"model_pool": []}
+{"model_pool": [144259]}
```

- 便于在评估或对战配置中引用指定的对手模型。

### 3.6 train_test.py

- 默认测试算法从 `ppo` 改为 `diy`：

```diff
-algorithm_name = "ppo"
+algorithm_name = "diy"
```

- `algorithm_name_list` 仍为 `["ppo", "diy"]`，便于在两种算法间切换。

### 3.7 文档：second_commit_detail.md

- 新增 markdown 文档，详细记录第二次提交（`4ae829e..d989588`）期间：
  - `agent_ppo/conf/conf.py` 奖励权重/动态权重的多轮调整；
  - `agent_ppo/feature/reward_process.py` 与 `agent_ppo/workflow/train_workflow.py` 的改动；
  - 若只关注当前 HEAD 的有效逻辑，需要注意：这些针对 `agent_ppo` 的改动在最新提交中大部分已被撤销或迁移到 `agent_diy` 体系，仅作为历史记录保留。

---

## 四、行为与使用层面的总结

- 工程从“仅包含基线 PPO（agent_ppo）”演化为“PPO 基线 + 自定义 diy 算法双轨并存”的结构：
  - `agent_ppo/*` 基本回到 init 状态，可作为官方示例或对照实验使用。
  - `agent_diy/*` 承载全部自定义逻辑（生存策略、奖励重构、Dual-Clip PPO 等）。
- 配置默认切换到 `diy`：
  - `conf/configure_app.toml` 与 `train_test.py` 均指向 `diy`，训练/测试脚本会优先走 `agent_diy` 的代码路径。
- 新增的代码主要集中在：
  - 生存状态机与启发式策略（回城/撤退/吃血包/对线拉扯）；
  - 奖励项的精细拆分与分阶段动态权重；
  - Dual-Clip PPO 以及奖励统计与训练过程监控。

若后续你希望：
- 单独对比第二次提交 vs 第一次提交，可直接复用已有的 `second_commit_detail.md`；  
- 或只关注第三次提交相对于第二次提交的变化，可以基于本文中 “新增 agent_diy/* + 配置切换为 diy + 还原 agent_ppo 部分改动” 这一主线进行阅读和调参。

