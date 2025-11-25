
## 1. agent_ppo/conf/conf.py

### 1.1 修改的代码行

```python
class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 0.010,  #进一步提高金钱
        "exp": 0.008,
        "ep_rate": 0.08,  #尝试不要蓝 再次降低 0.10->0.08
        "death": -2.0,  #保命
        "kill": 1.0,   #杀人
        "last_hit": 1.2,  #提高 1.0->1.2
        "forward": 0.01,
        # 新增奖励项
        "minion_clear": 1.5,           # 清兵线奖励
        "attack_tower": 3.2,           # 推塔奖励   3.0->3.2
        "health_pack_usage": 1.2,      # 使用血包奖励
    }
    # 新增的动态权重调整配置
    DYNAMIC_WEIGHT_CONFIG = {
        "early_game_threshold": 2400,    # 前期结束的帧号（1分20秒=80秒=2400帧）
        "late_game_threshold": 7200,     # 后期开始的帧号（4分钟=240秒=7200帧）

        # 前期的权重 - 发育类>人头类>推塔类
        "early_game_weights": {
            "hp_point": 2.0,
            "tower_hp_point": 8.0,    # 前期推塔权重最低
            "money": 0.015,           # 前期金钱权重最高  提高0.013-0.015
            "exp": 0.010,             # 前期经验权重最高
            "ep_rate": 0.05,
            "death": -2.5,            # 前期死亡惩罚较重  调整-3->-2.5
            "kill": 0.5,              # 前期击杀权重较低
            "last_hit": 1.5,          # 补刀权重提高
            "forward": 0.0085,   ####  前期走慢点  0.010-0.0085
            "minion_clear": 1.8,      # 清兵线奖励最高
            "attack_tower": 2.0,      # 前期推塔奖励较低
            "health_pack_usage": 1.2, 
        },

        # 中期的权重 - 发育类≈人头类≈推塔类
        "mid_game_weights": {
            "hp_point": 2.0,
            "tower_hp_point": 10.0,   # 中期推塔权重适中
            "money": 0.010,
            "exp": 0.008,
            "ep_rate": 0.05,
            "death": -2.2,
            "kill": 0.8,              # 中期击杀权重适中
            "last_hit": 1.2,
            "forward": 0.009,  ###0.012-0.010
            "minion_clear": 1.5,
            "attack_tower": 3.2,
            "health_pack_usage": 1.2,
        },

        # 后期的权重 - 发育类<人头类<推塔类
        "late_game_weights": {
            "hp_point": 2.0,
            "tower_hp_point": 11.0,   
            "money": 0.010,           
            "exp": 0.008,            
            "ep_rate": 0.05,
            "death": -2.0,     
            "kill": 1.0,              # 后期击杀权重提升
            "last_hit": 1.0,          # 后期补刀权重降低
            "forward": 0.010,         # 后期前进更积极  14-10
            "minion_clear": 1.3,      # 后期清兵权重降低
            "attack_tower": 4.0,      # 后期推塔奖励最高
            "health_pack_usage": 1.2,
        }
    }
```

### 1.2 逐行/逐项说明

- `tower_hp_point: 10.0`：由 `5.0` 提升到 `10.0`，显著放大推塔相关奖励。
- `money: 0.010`：由 `0.006` 提升到 `0.010`，并增加注释“进一步提高金钱”，强调经济发育。
- `exp: 0.008`：由 `0.006` 提升到 `0.008`，提高经验获取的奖励强度。
- `ep_rate: 0.08`：从原先较高的 `0.75` 调整为 `0.08`，并加上“尝试不要蓝”的注释，表示弱化蓝量管理在奖励中的权重。
- `death: -2.0`：由 `-1.0` 调整为 `-2.0`，加重死亡惩罚，鼓励更保守的生存策略。
- `kill: 1.0`：由原来的负值惩罚（`-0.6`）改为正向奖励 `1.0`，明确击杀为鼓励行为。
- `last_hit: 1.2`：由 `0.5` 提升到 `1.2`，并在注释中说明“提高 1.0->1.2”，进一步强调补刀的重要性。
- `# 新增奖励项`：用注释单独划出新增奖励类目，便于与基础奖励区分。
- `minion_clear: 1.5`：新增清兵线奖励基础权重，用于奖励在安全距离下积极清线。
- `attack_tower: 3.2`：新增推塔奖励基础权重（并注明从 3.0 提升到 3.2），让推塔在整体奖励中更突出。
- `health_pack_usage: 1.2`：新增血包使用奖励，鼓励在低血量时合理吃血包。
- `early_game_threshold` / `late_game_threshold`：以帧号划分前期（≤2400）、中期、后期（≥7200），作为动态权重切换的时间边界。
- `early_game_weights`：为前期单独配置各项权重：
  - 提高 `money`/`exp`/`minion_clear`，降低 `attack_tower` 与前压 `forward`，并加重 `death` 惩罚，体现“优先安全发育”的策略。
  - 适度降低 `kill` 权重，避免前期过度追求击杀。
- `mid_game_weights`：将发育、推塔、击杀权重拉近（例如 `tower_hp_point: 10.0`, `kill: 0.8`），体现“发育≈团战≈推塔”的中期均衡状态。
- `late_game_weights`：显著提高推塔与击杀权重（如 `tower_hp_point: 11.0`, `attack_tower: 4.0`, `kill: 1.0`），并降低 `minion_clear`、`last_hit`，体现后期以团战和拆塔为核心目标。
- 整个 `DYNAMIC_WEIGHT_CONFIG` 新增块：为后续在运行时根据当前帧号动态切换权重提供配置基础。

---

## 2. agent_ppo/feature/reward_process.py

### 2.1 修改的代码行

**（1）在 `result` 中接入动态权重更新：**

```python
    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        frame_no = frame_data["frameNo"]

        # 在每帧更新奖励计算前，动态调整权重
        self.update_dynamic_weights(frame_no)

        # 正常计算奖励
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        # 时间衰减
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value
```

**（2）新增 `update_dynamic_weights` 方法：**

```python
    # 动态调整奖励权重，根据当前帧号自动切换阶段
    def update_dynamic_weights(self, frame_no):
        cfg = GameConfig.DYNAMIC_WEIGHT_CONFIG
        early_th = cfg["early_game_threshold"]
        late_th = cfg["late_game_threshold"]

        # 判断当前阶段
        if frame_no <= early_th:
            phase_weights = cfg["early_game_weights"]
            phase_name = "early_game"
        elif frame_no >= late_th:
            phase_weights = cfg["late_game_weights"]
            phase_name = "late_game"
        else:
            phase_weights = cfg["mid_game_weights"]
            phase_name = "mid_game"

        # 更新 m_cur_calc_frame_map 中所有奖励项的权重
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            # 如果该阶段定义了新的权重，则替换；否则使用默认权重
            new_weight = phase_weights.get(reward_name, GameConfig.REWARD_WEIGHT_DICT.get(reward_name, 1.0))
            if reward_struct.weight != new_weight:
                reward_struct.weight = new_weight

        # 同步到主角和敌方的 frame_map
        for m in [self.m_main_calc_frame_map, self.m_enemy_calc_frame_map]:
            for reward_name, reward_struct in m.items():
                reward_struct.weight = phase_weights.get(reward_name, GameConfig.REWARD_WEIGHT_DICT.get(reward_name, 1.0))
```

**（3）在 `set_cur_calc_frame_vec` 中新增三类奖励项：**

```python
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            
            ########## 新增########################################
            # =============== 清兵线奖励 ===============
            elif reward_name == "minion_clear":
                try:
                    hero_pos = (
                        main_hero["actor_state"]["location"]["x"],
                        main_hero["actor_state"]["location"]["z"],
                    )

                    nearby_minion_count = 0
                    total_minion_distance = 0.0

                    for npc in npc_list:
                        if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                            npc.get("camp") != camp and 
                            npc.get("hp", 0) > 0):
                            
                            npc_pos = (npc["location"]["x"], npc["location"]["z"])
                            dist = math.dist(hero_pos, npc_pos)
                            
                            if dist <= 10.0:
                                nearby_minion_count += 1
                                total_minion_distance += dist

                    # 清兵线奖励计算
                    if nearby_minion_count > 0:
                        avg_distance = total_minion_distance / nearby_minion_count
                        base_reward = nearby_minion_count * 0.3
                        
                        # 距离奖励：保持适当距离（3-3.5米）
                        distance_reward = 0.0
                        if 3.0 <= avg_distance <= 3.5:
                            distance_reward = 0.5    # 理想攻击距离
                        elif avg_distance < 2.0:
                            distance_reward = -0.2   # 太近有危险
                        elif avg_distance > 8.0:
                            distance_reward = -0.01  # 太远只有轻微惩罚（可能兵线还没到）
                            
                        reward_struct.cur_frame_value = base_reward + distance_reward
                    else:
                        reward_struct.cur_frame_value = 0.0  # 没有小兵时无奖励

                except Exception:
                    reward_struct.cur_frame_value = 0.0

            # =============== 推塔奖励 ===============
            elif reward_name == "attack_tower":
                try:
                    reward_struct.cur_frame_value = 0.0
                    
                    if enemy_tower and main_hero:
                        hero_pos = (
                            main_hero["actor_state"]["location"]["x"],
                            main_hero["actor_state"]["location"]["z"],
                        )
                        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
                        dist = math.dist(hero_pos, tower_pos)
                        
                        # 在塔攻击范围内
                        if dist <= 8.0:
                            # 检查塔血量是否减少
                            current_tower_hp = enemy_tower["hp"]
                            last_tower_hp = getattr(self, 'last_enemy_tower_hp', current_tower_hp)
                            
                            if current_tower_hp < last_tower_hp:
                                # 塔血量减少，给予推塔奖励
                                hp_reduced = last_tower_hp - current_tower_hp
                                reward_struct.cur_frame_value = hp_reduced / 50.0  # 每减少50点血量给1分
                            
                            self.last_enemy_tower_hp = current_tower_hp
                            
                            # 在塔附近且有兵线时给予基础奖励
                            if nearby_minion_count > 2 and dist <= 6.0:
                                reward_struct.cur_frame_value += 0.5

                except Exception:
                    reward_struct.cur_frame_value = 0.0

            # =============== 血包奖励 ===============
            elif reward_name == "health_pack_usage":
                try:
                    hero_state = main_hero["actor_state"]
                    hero_pos = (hero_state["location"]["x"], hero_state["location"]["z"])
                    hero_hp = hero_state["hp"]
                    hero_max_hp = hero_state["max_hp"]
                    hp_ratio = hero_hp / max(hero_max_hp, 1)
                    
                    health_pack_reward = 0.0
                    cakes = frame_data.get("cakes", [])
                    
                    # 检查血包
                    nearest_cake_dist = float('inf')
                    for cake in cakes:
                        # 根据数据协议，血包有 collider 字段
                        collider = cake.get("collider", {})
                        if collider:
                            # 假设血包位置在碰撞体中心
                            center = collider.get("center", {})
                            if center:
                                cake_pos = (center.get("x", 0), center.get("z", 0))
                                dist = math.dist(hero_pos, cake_pos)
                                if dist < nearest_cake_dist:
                                    nearest_cake_dist = dist
                    
                    # 血量低于20%时的奖励逻辑
                    if hp_ratio < 0.2:
                        if nearest_cake_dist <= 3.0:
                            # 低血量且靠近血包，高奖励
                            health_pack_reward = 2.0
                        elif nearest_cake_dist <= 6.0:
                            # 低血量且在血包附近，中等奖励
                            health_pack_reward = 1.0
                        else:
                            # 低血量但远离血包，轻微惩罚
                            health_pack_reward = -0.3
                    
                    # 检测血量恢复（可能使用了血包）
                    current_hp = hero_hp
                    last_hp = getattr(self, 'last_hp', current_hp)
                    if current_hp > last_hp + 80:  # 血量恢复较多
                        # 检查恢复时是否在血包附近
                        if nearest_cake_dist <= 3.0:
                            health_pack_reward += 1.5  # 使用血包恢复的额外奖励
                    self.last_hp = current_hp
                    
                    reward_struct.cur_frame_value = health_pack_reward
                    
                except Exception:
                    reward_struct.cur_frame_value = 0.0
```

**（4）前压奖励阈值调整：**

```python
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        ...
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.73 and dist_hero2emy > dist_main2emy:  #保守点0.7-0.73
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value
```

**（5）将新增奖励项纳入 `get_reward` 汇总：**

```python
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            ##### 新增##############################################
            elif reward_name in ["minion_clear", "attack_tower", "health_pack_usage"]:
                # 这些奖励项直接使用当前帧的值
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                ...
```

### 2.2 逐行/逐块说明

- `frame_no = frame_data["frameNo"]`：提前在 `result` 的开头取出帧号，供动态权重更新和时间衰减共同使用。
- `self.update_dynamic_weights(frame_no)`：在每帧奖励计算前调用动态权重更新函数，使奖励权重随对局进程（前/中/后期）自动变化。
- `# 时间衰减` 注释及其后循环：保持原有时间衰减逻辑，只是将 `frame_no` 的计算移到函数开头。
- `update_dynamic_weights` 整个方法：
  - 从 `GameConfig.DYNAMIC_WEIGHT_CONFIG` 中读取阈值与各阶段权重。
  - 根据 `frame_no` 判定当前阶段：早期、中期、后期，并设置 `phase_weights`。
  - 遍历 `m_cur_calc_frame_map`，如果当前阶段为某奖励项定义了新权重，则覆盖原有权重；否则回退到全局 `REWARD_WEIGHT_DICT`。
  - 将结果同步到 `m_main_calc_frame_map` 和 `m_enemy_calc_frame_map`，保证主角与对手使用同一套阶段性权重。
- `minion_clear` 相关代码块：
  - 计算英雄当前位置 `hero_pos`。
  - 遍历 `npc_list`，筛选敌方小兵（`ACTOR_SUB_SOLDIER` 且存活），统计 10 米范围内的小兵数量与平均距离。
  - `base_reward = nearby_minion_count * 0.3`：根据周围小兵数量给基础清线奖励。
  - 通过 `avg_distance` 加入距离修正：理想距离（3–3.5 米）加奖励、太近扣分、太远轻微扣分。
  - 无小兵或异常时将奖励置 0，避免异常数据影响训练。
- `attack_tower` 相关代码块：
  - 初始将推塔奖励置 0，仅在满足条件时累加。
  - 当英雄在 8 米塔射程内时，比较当前塔血与上一次记录的塔血 `last_enemy_tower_hp`，如果塔血下降则根据下降量 `hp_reduced / 50.0` 发放推塔奖励。
  - 当塔附近小兵较多（`nearby_minion_count > 2` 且英雄距离塔 ≤6 米）时，额外给 0.5 的基础推塔奖励，鼓励“带线推塔”而非单人裸塔。
  - 异常场景统一回落到 0 奖励。
- `health_pack_usage` 相关代码块：
  - 通过 `hp_ratio` 判断当前血量占比；只在低血量（<20%）时启用血包相关奖励。
  - 遍历 `cakes`，用碰撞体中心位置计算最近血包距离 `nearest_cake_dist`。
  - 低血量 + 距血包距离不同，分配不同奖励：近距离高奖励，中等距离中等奖励，远离血包给轻微惩罚。
  - 使用 `last_hp` 与当前血量对比，如果本帧血量显著提升（>80），且附近有血包，再给额外奖励，鼓励在危险时主动吃血包恢复。
  - 整体通过 `health_pack_reward` 聚合，并在异常时归零。
- `calculate_forward` 中阈值从 0.99 调整为 0.73，并在注释中标注“保守点 0.7-0.73”：允许英雄在血量高于 73% 时就可以前压，而非几乎满血才前压，提升进攻性。
- `elif reward_name in ["minion_clear", "attack_tower", "health_pack_usage"]`：
  - 将三类新增奖励项纳入 `get_reward` 汇总流程，直接取主角侧 `cur_frame_value` 作为 `value`，不做零和差分。

---

## 3. agent_ppo/workflow/train_workflow.py

### 3.1 修改的代码行

```python
        if is_eval:
            #opponent_agent = usr_conf["env_conf"]["episode"]["eval_opponent_type"]
            # 2. 使用对手模型 
            opponent_agent = "147741"
            # 3. 从 common_ai 和对手模型列表中随机选择
            #opponent_agent_list = ["140865", "140518"]
            #opponent_agent = random.choice(opponent_agent_list)

        usr_conf["env_conf"]["episode"]["opponent_agent"] = opponent_agent
```

### 3.2 逐行说明

- 注释掉原有的 `eval_opponent_type` 配置读取，避免继续使用统一配置的评估对手。
- 新增注释“使用对手模型”，并将 `opponent_agent` 强制指定为 `"147741"`，方便针对该模型进行固定对手评估。
- 预留了 `opponent_agent_list` 与 `random.choice` 的注释示例，暗示未来可以改成在多个对手模型之间随机选择。
- 保持最后一行 `usr_conf["env_conf"]["episode"]["opponent_agent"] = opponent_agent` 不变，只是传入的 agent id 来源发生改变。

---

## 4. conf/kaiwudrl/configure.toml

### 4.1 修改的配置行

```toml
# 用户保存模型最大次数, 设置为小于等于0代表不限制
user_save_mode_max_count = 400
# 用户保存模型的频率, 设置为小于等于0代表不限制
user_save_model_max_frequency_per_min = 2
```

### 4.2 逐行说明

- `user_save_mode_max_count = 400`：原先为 `0`（不限制），改为最多可保留 400 次用户保存的模型快照，避免无限制占用磁盘。
- `user_save_model_max_frequency_per_min = 2`：原先为 `0`（不限制），改为每分钟最多保存 2 次模型，防止过于频繁地保存造成 I/O 压力。

---

## 5. kaiwu.json

### 5.1 修改的配置行

```json
{"model_pool": [147741]}
```

### 5.2 逐行说明

- 将 `model_pool` 从空列表 `[]` 改为包含单个模型 id `147741`，与训练/评估中使用的对手模型 id 保持一致，表示当前环境下只启用该模型作为对手或候选模型。

---

## 6. train_test.py

### 6.1 修改的代码行

```python
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "diy"
```

### 6.2 逐行说明

- `algorithm_name_list = ["ppo", "diy"]`：保持算法候选列表不变，仍支持 `ppo` 与 `diy` 两套算法实现。
- `algorithm_name = "diy"`：将默认训练/测试算法从 `"ppo"` 切换为 `"diy"`，本地运行 `train_test.py` 时会优先走 `diy` 算法的训练流程。

---

## 7. 小结

- 本次提交围绕“奖励权重重构 + 新增奖励项 + 固定对手评估 + 模型保存策略 + 默认算法切换”五个方向进行了修改。
- 文档中已将所有涉及变更的代码行以代码块方式完整列出，并对每处变更的设计目的和影响进行了对应说明，便于后续回溯和进一步调参。

