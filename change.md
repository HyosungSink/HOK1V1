# tqq 分支变更说明（4ae829e → 7747d4c）

本说明基于 `tqq` 分支首尾两个提交：

- 起始：`4ae829e init`
- 最新：`7747d4c DCP-32`

完整代码 diff 已写入根目录 `code.diff`，此处仅梳理改动思路与结构性变化。

---

## 一、整体目标

- 将原 `agent_ppo` 路径整体对齐 / 复用 `agent_diy` 的模型、特征和配置。
- 在策略侧引入显式的生存状态机（撤退 / 回城 / 恢复等），修复“回城不果断”“泉水站一秒就走”等问题。
- 在奖励侧引入更细粒度的密集奖励和动态权重，强化对清线、塔防、生存与战斗质量的引导。
- 在训练入口和平台配置上，将默认算法切换到 `diy`，并完善模型保存与评估配置。

---

## 二、训练入口与平台配置调整

1. 算法切换为 `diy`
   - `conf/configure_app.toml:24-27`  
     - `algo = "ppo"` → `algo = "diy"`，框架层默认跑 `diy` 算法。
   - `train_test.py:31-32`  
     - `algorithm_name = "ppo"` → `algorithm_name = "diy"`，本地 `train_test` 脚本默认跑 `diy`。

2. 模型保存与上传策略
   - `conf/kaiwudrl/configure.toml:273-276`  
     - `user_save_mode_max_count: 0` → `400`：限制用户最大模型保存次数，防止无限增长。
     - `user_save_model_max_frequency_per_min: 0` → `2`：限制用户手动保存频率（每分钟 2 次）。

3. 评估模型池配置
   - `kaiwu.json:1`  
     - `{"model_pool": []}` → `{"model_pool": [144259]}`：显式配置评估可用的模型 id，配合 workflow 中的 `get_valid_model_pool` 使用。

4. 其他小调整
   - `.gitignore:1-3`  
     - `agent_diy` → `agent_diy/`：明确忽略整个目录，避免误提交生成文件。

---

## 三、agent_ppo 侧重构与生存策略逻辑

1. 依赖整体切换到 agent_diy
   - `agent_ppo/agent.py:10-22`  
     - 原来从 `agent_ppo.*` 导入 `Model / Config / GameConfig / Algorithm / feature`；  
     - 现在全部切换为从 `agent_diy` 导入：`agent_diy.model.model.Model`、`agent_diy.feature.definition.*`、`agent_diy.conf.conf.{Config, GameConfig}`、`agent_diy.algorithm.algorithm.Algorithm`。  
   - `agent_ppo/feature/definition.py:14-21`、`agent_ppo/model/model.py:18-19`、`agent_ppo/algorithm/algorithm.py:8`、`agent_ppo/feature/reward_process.py:6` 等文件也统一改为引用 `agent_diy.conf.conf` 中的配置。
   - 思路：让 `agent_ppo` 这一条链路完全复用 `agent_diy` 已经调好的网络结构与配置，只在策略逻辑和奖励设计上做定制化。

2. Agent 类结构与状态管理
   - `agent_ppo/agent.py:25-67`  
     - 在 `Agent` 中新增大量与生存相关的状态：
       - `survival_state`（`normal / need_retreat / retreating / recalling / recovering`）
       - `last_hp_ratio`、`continuous_low_hp_frames`、`recall_cooldown`、`last_distance_to_spring`
       - 战斗状态：`in_combat`、`combat_frames`、`last_damage_frame`、`enemy_distance`
       - 行为历史：`action_history`、`max_history_length`
       - `[修改点1]` 新增 `pending_normal_attack` 标记，用于“技能后强制平 A”。
     - 原来在 `Agent` 内部自己创建 `optimizer / scheduler / Algorithm` 的逻辑被删除，现在只负责持有 `Model` 并调用 `agent_diy` 的 `Algorithm`。

3. 观测增强与生存特征
   - `agent_ppo/agent.py:69-113`  
     - `_model_inference` 中先调用 `enhance_observation` 为 `ObsData` 增加额外的 `survival_features`：
       - 当前生存状态 one-hot 编码。
       - 低血量阈值标记（<35%, <25%）。
       - 战斗状态与战斗持续帧数。
       - 回城冷却可用性。
     - 这些特征挂在 `obs_data.survival_features` 上，供模型或后续处理使用。

4. 生存策略状态机与决策优先级
   - `agent_ppo/agent.py:115-209, 267-347, 513-567`  
     - 新增一套基于 `hp_ratio`、战斗状态、泉水距离的生存状态机：
       - `should_force_recall` / `should_retreat`：判断极低血量强制回城、低血量撤退的条件；将撤退阈值从 0.35 降到 0.30（[修改点4]），避免过于保守。
       - `apply_retreat_action`：优先用位移技能（如鲁班 2 技能击退）逃生，否则朝泉水方向移动。
       - `apply_defensive_action`：在低血量近战情况下，选择反方向移动和控制技能。
       - `update_combat_state`：根据敌人距离和受伤情况维护 `in_combat` / `combat_frames` 等战斗状态。
     - 回城与泉水逻辑：
       - 极低血量且脱战并且无回城冷却时，强制执行回城动作，设置 `recall_cooldown`，将 `survival_state` 置为 `recalling`。
       - 在 `recovering` 状态下，如果在泉水附近但未恢复到 90% 血量，则强制 `no_op`，避免“刚站泉水就又走出去”的问题。

5. 动作后处理中的关键策略（action_process）
   - `agent_ppo/agent.py:267-347`  
     核心是对模型输出的 `action` 做后处理，优先满足生存 / 资源 / 目标选择逻辑：
     - `[修改点1] 技能后强制平 A`  
       - 如果上一帧使用了技能，本帧强制输出普攻（`Config.ACTION_MAPPING["attack"]`），并清除 `pending_normal_attack`，保证技能 → 普攻的连招稳定触发。
     - `[修改点2] 低血量优先吃塔下血包`  
       - 当 `hp_ratio < 0.40` 且存在可用血包且不在战斗中时，强制朝血包方向移动，而不立即回城或继续前压。
     - `[修改点4] 低血量撤退阈值调整`  
       - 撤退阈值从 0.35 降为 0.30，进入撤退态时优先尝试位移技能，否则向泉水移动。
     - 泉水恢复策略：
       - 在 `recovering` 状态，要求血量恢复到 90% 才离开泉水；若离泉水太远，则强制朝泉水移动；泉水范围内未恢复够则强制待机。
     - 正常战斗下的激进度控制（`survival_state == "normal"`）：
       - 若敌方英雄在普攻范围内（4500），即使有小兵，仍强制攻击英雄（[修改点3]），提高换血质量。
       - 根据血量和敌我距离，选择“压线 / 保守拉扯 / 后撤”不同策略。

6. 其它
   - `agent_ppo/agent.py:588-615`  
     - `update_status` / `reset` 中同步维护生存状态、血量历史、泉水距离和 `pending_normal_attack`，保证跨局与跨帧状态一致。

---

## 四、奖励函数与配置升级

1. 奖励整体设计升级
   - `agent_ppo/conf/conf.py:4-56`  
     - 原本简单的 `hp_point / tower_hp_point / money / exp / death / kill / last_hit / forward` 等少量项，升级为完整的生存与推进奖励体系：
       - 基础：`hp_point`、`money`、`exp`、`last_hit`
       - 新增：`minion_damage`（对小兵造成伤害）用于鼓励清线
       - 战斗：`kill`、`death`、`ep_rate`
       - 推进：`forward`、`tower_hp_point`
       - 密集：`damage_exchange`、`skill_hit`
       - 生存策略：`recall_decision`、`health_management`、`combat_timing`、`tower_defense`、`positioning`
   - 动态权重：
     - `ADAPTIVE_WEIGHT` 中按早期 / 中期 / 后期区分，对 `money/exp/death/recall_decision/tower_defense/forward` 等权重做时间分段调整，体现对不同对局阶段的不同偏好。

2. 回城与战斗配置
   - `agent_ppo/conf/conf.py:58-75`  
     - 新增 `RECALL_CONFIG`：低血量阈值、必回城血量、安全血量、泉水距离、战斗距离、撤退评估帧数。
     - 新增 `COMBAT_CONFIG`：理想战斗距离、血量优势 / 劣势阈值。
   - 这些配置被生存策略和奖励函数同时使用，形成从配置 → 决策 → 奖励的闭环。

3. PPO / 模型相关配置
   - `agent_ppo/conf/conf.py:91-187`  
     - 轻微调整学习率：`1e-3` → `8e-4`，目标学习率 `1e-4` → `8e-5`。
     - 新增 `DUAL_CLIP_PARAM_C = 0.75`，供 Dual-Clip PPO 使用。
     - 新增 `ACTION_MAPPING` 与 `HERO_SKILL_CONFIG`，显式定义鲁班 7 号的技能 / 攻击 / 射程等信息，为策略层的手工逻辑（如技能后接普攻、距离判断等）提供依据。

4. 奖励计算逻辑增强
   - `agent_ppo/feature/reward_process.py`  
     - 改为从 `agent_diy.conf.conf.GameConfig` 读取配置，并在 `GameRewardManager` 中增加大量状态缓存和统计字段（英雄 / 敌人 / 塔血量、位置信息、小兵血量、战斗状态等）。
     - `result` 中新增：
       - 调用 `get_adaptive_weight_multiplier` 按对局时间段动态放大 / 缩小不同奖励分量。
     - 新增 / 强化的奖励维度：
       - `calculate_recall_reward(..., frame_no)`：基于生存状态机的回城奖励，包括“何时开始撤退”“是否成功回城”“是否在泉水恢复够再离开”等，多处用到 `GameConfig.RECALL_CONFIG` 中的阈值（[修改点4] 对低血 / 迟离泉水等情况有不同奖励 / 惩罚）。
       - `calculate_minion_damage_reward`：通过跟踪敌方小兵血量变化，给“清线伤害”以密集奖励，用于验证并驱动清线策略（[修改点2]）。
       - `calculate_damage_exchange_reward`：根据当前帧敌我伤害和血量比例计算伤害交换质量，鼓励有利换血、惩罚亏血。
     - 思路：奖励从“结果导向（塔血 / 人头）”扩展为“过程导向（换血质量、清线、回城决策、站位等）”，从而为生存状态机和动作后处理提供正向信号。

---

## 五、模型与算法层改动

1. 模型参数与 Dual-Clip PPO
   - `agent_ppo/model/model.py`  
     - 配置来源改为 `agent_diy.conf.conf.{DimConfig, Config}`，保证和 `agent_diy` 完全对齐（数据切分、维度等）。
     - 在 `Model.__init__` 中新增 `self.dual_clip_c = Config.DUAL_CLIP_PARAM_C`。
     - 在策略损失计算部分，引入 Dual-Clip PPO：
       - 先计算标准 PPO 的 `surr1 / surr2` 与裁剪后的 `standard_clip_surr`。
       - 再计算 `dual_clip_surr = c * A_t`，并在 `A_t < 0` 时取 `max(standard_clip_surr, dual_clip_surr)`，抑制负优势样本更新幅度，降低策略崩坏风险。

2. 算法封装与训练监控
   - `agent_ppo/algorithm/algorithm.py:10-43`  
     - 仍保持原来 `learn` 的数据拆分与前向 / 反向流程，但配置改为使用 `agent_diy.conf.conf.Config`。
     - 新增 `reward_stats`、`update_reward_stats`、`get_reward_statistics`、`log_training_status`、`should_save_model` 等：
       - 按窗口统计 `damage_exchange / tower_defense / skill_hit / forward / kill / death / tower_hp_point / minion_damage` 等奖励的均值、方差、最大值。
       - 周期性将 loss 与关键奖励统计写入 `monitor`，并在日志中打印。
       - `should_save_model` 根据奖励总和提升判断是否值得保存模型，构建基于性能的保存策略。
   - 思路：训练侧不仅看 loss，还关注关键行为指标的变化，为后续自动调参与模型选择打基础。

3. 特征定义对齐
   - `agent_ppo/feature/definition.py:14-21`  
     - 仅更换配置来源为 `agent_diy.conf.conf.Config`，其余采样逻辑（GAE、LSTM 拼帧等）保持不变，确保 `ppo` 链路上的样本格式与 `diy` 一致。

---

## 六、workflow 与评估流程

1. 训练 workflow 对齐 diy
   - `agent_ppo/workflow/train_workflow.py:14-25, 40-52, 63-66`  
     - 特征和配置从 `agent_diy.feature.definition`、`agent_diy.conf.conf.GameConfig` 导入。
     - 读取的环境配置文件改为 `agent_diy/conf/train_env_conf.toml`，并在校验失败时直接返回。
     - 模型保存频率改为使用 `GameConfig.MODEL_SAVE_INTERVAL`。

2. 模型加载与评估对手
   - `agent_ppo/workflow/train_workflow.py:81-88, 132-145, 169-199`  
     - 保留原有“交替训练阵营、按频率插入 eval 对局”的逻辑。
     - 对手类型为数字 id 时，通过 `get_valid_model_pool` 和 `kaiwu.json` 校验 id 合法性，否则抛异常；与新配置的 `model_pool` 联动。

---

## 七、小结

- 这个提交从“配置 + agent_ppo + 奖励 + 模型 + workflow”全链路上，将 `ppo` 路径与 `agent_diy` 强绑定，并在此基础上增强：
  - 生存与回城相关的策略逻辑（状态机 + 手工规则）。
  - 以清线、塔防和换血质量为核心的密集奖励设计。
  - Dual-Clip PPO 与奖励统计监控，加强训练稳定性与可观测性。
- 对外行为上，默认跑 `diy` 算法、使用指定的模型池进行评估，并允许更频繁地进行模型保存，方便在平台上快速迭代与回滚。

