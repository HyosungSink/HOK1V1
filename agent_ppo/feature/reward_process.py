#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import numpy as np
from agent_ppo.conf.conf import GameConfig

class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True

def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map

class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        
        # 历史数据追踪
        self.last_hero_hp = 1.0
        self.last_enemy_hp = 1.0
        self.last_tower_hp = 1.0
        self.last_hero_pos = None
        self.skill_cooldowns = [0, 0, 0]
        
        # [修改点2] 小兵血量追踪
        self.last_enemy_minion_hp = {}
        
        # 生存状态机
        self.survival_state = "normal"  # normal, need_retreat, retreating, recalling, recovering
        self.state_counter = 0
        self.last_hp_ratio = 1.0
        self.recall_cooldown = 0
        self.combat_frames = 0
        self.last_damage_taken = 0
        self.last_distance_to_spring = float('inf')

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        exp_levels = [160, 298, 446, 524, 613, 713, 825, 950, 1088, 1240, 1406, 1585, 1778, 1984]
        for i, exp in enumerate(exp_levels, 1):
            self.m_each_level_max_exp[i] = exp

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        
        frame_no = frame_data["frameNo"]
        
        # 动态权重调整
        weight_multiplier = self.get_adaptive_weight_multiplier(frame_no)
        for key in self.m_reward_value:
            if key in weight_multiplier:
                self.m_reward_value[key] *= weight_multiplier[key]
        
        # 时间衰减
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        
        return self.m_reward_value

    def get_adaptive_weight_multiplier(self, frame_no):
        """根据游戏阶段动态调整权重"""
        weight_multiplier = {}
        
        if frame_no < 5000:  # 早期
            weight_multiplier = GameConfig.ADAPTIVE_WEIGHT.get("early_game", {})
        elif frame_no < 10000:  # 中期
            weight_multiplier = GameConfig.ADAPTIVE_WEIGHT.get("mid_game", {})
        else:  # 后期
            weight_multiplier = GameConfig.ADAPTIVE_WEIGHT.get("late_game", {})
        
        return weight_multiplier

    # [修改点4] 增加 frame_no 参数
    def calculate_recall_reward(self, main_hero, main_spring, enemy_hero, frame_no):
        """计算回城奖励 - 核心生存机制"""
        if not main_spring:
            return 0.0
        
        hero_hp = main_hero["actor_state"]["hp"]
        hero_max_hp = main_hero["actor_state"]["max_hp"]
        hero_hp_ratio = hero_hp / hero_max_hp if hero_max_hp > 0 else 1.0
        
        hero_pos = (main_hero["actor_state"]["location"]["x"],
                   main_hero["actor_state"]["location"]["z"])
        spring_pos = (main_spring["location"]["x"],
                     main_spring["location"]["z"])
        
        distance_to_spring = math.dist(hero_pos, spring_pos)
        
        # 更新回城冷却
        if self.recall_cooldown > 0:
            self.recall_cooldown -= 1
        
        reward = 0.0
        
        # 检查是否在战斗中
        in_combat = False
        if enemy_hero:
            enemy_pos = (enemy_hero["actor_state"]["location"]["x"],
                        enemy_hero["actor_state"]["location"]["z"])
            enemy_distance = math.dist(hero_pos, enemy_pos)
            in_combat = enemy_distance < 6000
        
        # 动态获取阈值 (确保使用修改后的 0.30)
        low_hp_threshold = GameConfig.RECALL_CONFIG["low_hp_threshold"]

        # 状态机逻辑
        if self.survival_state == "normal":
            # 正常状态 -> 检查是否需要撤退
            if hero_hp_ratio < low_hp_threshold:  # [修改点4] 使用 0.30 阈值
                self.survival_state = "need_retreat"
                self.state_counter = 0
                reward += 0.5  # 识别危险的小奖励
                
        elif self.survival_state == "need_retreat":
            # 需要撤退状态
            self.state_counter += 1
            
            if hero_hp_ratio < 0.25:  # 血量持续下降到25%
                if in_combat:
                    # 战斗中低血量，强烈惩罚
                    reward -= 3.0
                else:
                    # 脱战了，应该回城
                    if distance_to_spring > 1000:
                        reward -= 2.0  # 惩罚不回城
                    else:
                        self.survival_state = "recovering"
                        reward += 3.0  # 奖励成功回城
            
            # 检查是否正在撤退
            if distance_to_spring < self.last_distance_to_spring:
                reward += 1.0  # 奖励向泉水移动
                self.survival_state = "retreating"
                
        elif self.survival_state == "retreating":
            # 撤退中
            if distance_to_spring < 2000:  # 接近泉水
                self.survival_state = "recovering"
                reward += 2.5  # 奖励成功撤退到安全区
            elif in_combat:
                reward -= 1.0  # 撤退中被追击
                
        elif self.survival_state == "recovering":
            # 恢复中
            self.state_counter += 1 # 计数

            # [修改点4] 后期取消血量比逻辑
            is_late_game = frame_no >= 10000 
            
            if is_late_game:
                 # 后期：只要待够时间 (e.g. 50帧) 即算成功，不强求回满血
                 if self.state_counter >= 50:
                    self.survival_state = "normal"
                    reward += 4.0
                    self.recall_cooldown = 300
                 elif distance_to_spring > 2000:
                    reward -= 1.5
            else:
                # 前中期：需要回血
                if hero_hp_ratio > 0.8:  # 血量恢复到80%
                    self.survival_state = "normal"
                    reward += 4.0  # 奖励完成整个生存循环
                    self.recall_cooldown = 300  # 设置回城冷却
                elif distance_to_spring > 2000:  # 恢复未完成就离开
                    reward -= 1.5  # 惩罚过早离开泉水
        
        # 存储距离用于下一帧比较
        self.last_distance_to_spring = distance_to_spring
        
        # 额外奖励：满血不要待在泉水
        if hero_hp_ratio > 0.9 and distance_to_spring < 1000:
            reward -= 0.8  # 满血赖泉水的惩罚
        
        return reward
    
    # [修改点2] 新增函数：计算小兵伤害奖励
    def calculate_minion_damage_reward(self, frame_data, main_camp):
        """计算对敌方小兵造成伤害的奖励"""
        reward = 0.0
        
        npc_list = frame_data["npc_states"]
        
        for npc in npc_list:
            if npc["camp"] != main_camp and npc["sub_type"] == "ACTOR_SUB_SOLDIER":
                minion_id = npc["runtime_id"]
                current_hp = npc["hp"]
                
                if minion_id in self.last_enemy_minion_hp:
                    # 造成伤害 = 上一帧血量 - 当前血量
                    damage_dealt = max(0, self.last_enemy_minion_hp[minion_id] - current_hp)
                    
                    # 简单奖励：每100点伤害给 0.01 奖励 (基础值, 会被权重放大)
                    reward += damage_dealt / 1000.0 
                
                self.last_enemy_minion_hp[minion_id] = current_hp
                
        # 清理已阵亡小兵的记录
        current_minion_ids = {npc["runtime_id"] for npc in npc_list if npc["sub_type"] == "ACTOR_SUB_SOLDIER" and npc["camp"] != main_camp}
        self.last_enemy_minion_hp = {k: v for k, v in self.last_enemy_minion_hp.items() if k in current_minion_ids}
        
        return reward

    def calculate_damage_exchange_reward(self, main_hero, enemy_hero):
        """计算即时伤害交换奖励"""
        if not enemy_hero:
            return 0.0
            
        if not hasattr(self, 'last_main_hp'):
            self.last_main_hp = main_hero["actor_state"]["hp"]
            self.last_enemy_hp = enemy_hero["actor_state"]["hp"]
            return 0.0
        
        # 计算本帧伤害
        damage_dealt = max(0, self.last_enemy_hp - enemy_hero["actor_state"]["hp"])
        damage_taken = max(0, self.last_main_hp - main_hero["actor_state"]["hp"])
        
        # 更新历史血量
        self.last_main_hp = main_hero["actor_state"]["hp"]
        self.last_enemy_hp = enemy_hero["actor_state"]["hp"]
        self.last_damage_taken = damage_taken
        
        # 血量比例
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        enemy_hp_ratio = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
        
        # 基础交换比
        if damage_dealt + damage_taken > 0:
            exchange_ratio = (damage_dealt - damage_taken) / 100.0
        else:
            exchange_ratio = 0
        
        # 根据双方血量调整奖励
        hp_advantage = hero_hp_ratio - enemy_hp_ratio
        
        reward = 0.0
        
        # 低血量决策
        if hero_hp_ratio < 0.3:
            if damage_taken > 0:
                reward -= damage_taken * 0.02  # 低血量受伤严重惩罚
            elif damage_dealt > 0 and damage_taken == 0:
                reward += damage_dealt * 0.01  # 安全输出奖励
        
        # 血量优势时的交换
        elif hp_advantage > 0.2:
            reward += exchange_ratio * 0.3  # 有优势时鼓励换血
        
        # 血量劣势时的交换
        elif hp_advantage < -0.2:
            if exchange_ratio > 0:
                reward += exchange_ratio * 0.5  # 劣势但打出优势交换，额外奖励
            else:
                reward += exchange_ratio * 0.1  # 劣势换血失败，轻微惩罚
        
        # 均势交换
        else:
            reward += exchange_ratio * 0.2
        
        return reward

    def calculate_health_management_reward(self, main_hero):
        """血量管理奖励"""
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        
        reward = 0.0
        
        # 维持健康血线奖励
        if 0.5 < hero_hp_ratio < 0.8:
            reward += 0.3  # 健康血量区间
        elif hero_hp_ratio > 0.8:
            reward += 0.5  # 高血量奖励
        elif hero_hp_ratio < 0.3:
            reward -= 0.5  # 危险血量惩罚
        
        # 血量回复奖励
        if hasattr(self, 'last_hp_ratio'):
            hp_change = hero_hp_ratio - self.last_hp_ratio
            if hp_change > 0 and self.last_hp_ratio < 0.5:
                reward += hp_change * 2.0  # 低血量回复奖励
        
        self.last_hp_ratio = hero_hp_ratio
        
        return reward

    def calculate_combat_timing_reward(self, main_hero, enemy_hero):
        """战斗时机选择奖励"""
        if not enemy_hero:
            return 0.0
            
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        enemy_hp_ratio = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
        
        hero_pos = (main_hero["actor_state"]["location"]["x"],
                   main_hero["actor_state"]["location"]["z"])
        enemy_pos = (enemy_hero["actor_state"]["location"]["x"],
                    enemy_hero["actor_state"]["location"]["z"])
        
        distance = math.dist(hero_pos, enemy_pos)
        
        reward = 0.0
        
        # 进入战斗距离
        if distance < 5000:
            self.combat_frames += 1
            
            # 有利时机进攻
            if hero_hp_ratio > enemy_hp_ratio * 1.3:
                reward += 0.5  # 血量优势时进攻
            
            # 不利时机进攻惩罚
            elif hero_hp_ratio < enemy_hp_ratio * 0.7:
                reward -= 1.0  # 血量劣势还要打
            
            # 持续战斗时长控制
            if self.combat_frames > 100:
                if hero_hp_ratio < 0.5:
                    reward -= 0.5  # 长时间战斗且血量不健康
        else:
            self.combat_frames = 0
        
        return reward

    def calculate_tower_defense_reward(self, main_hero, main_tower, enemy_hero):
        """塔防奖励"""
        if not main_tower:
            return 0.0
        
        hero_pos = (main_hero["actor_state"]["location"]["x"],
                   main_hero["actor_state"]["location"]["z"])
        tower_pos = (main_tower["location"]["x"],
                    main_tower["location"]["z"])
        
        distance_to_tower = math.dist(hero_pos, tower_pos)
        tower_hp_ratio = main_tower["hp"] / main_tower["max_hp"]
        
        # 危险等级
        danger_level = (1 - tower_hp_ratio) ** 2
        
        reward = 0.0
        
        # 塔受威胁时
        if tower_hp_ratio < 0.7:
            if enemy_hero:
                enemy_pos = (enemy_hero["actor_state"]["location"]["x"],
                           enemy_hero["actor_state"]["location"]["z"])
                enemy_to_tower = math.dist(enemy_pos, tower_pos)
                
                if enemy_to_tower < 6000:  # 敌人威胁塔
                    if distance_to_tower < 5000:
                        reward += danger_level * 1.5  # 守护塔
                    else:
                        reward -= danger_level * 2.0  # 塔被攻击但不在
        
        return reward

    def calculate_skill_hit_reward(self, frame_action):
        """技能命中奖励"""
        reward = 0.0
        
        if "skill_action" not in frame_action:
            return 0.0
        
        skill_actions = frame_action.get("skill_action", [])
        for skill in skill_actions:
            if skill.get("hit", False):
                skill_type = skill.get("skill_type", 1)
                base_reward = [0.3, 0.5, 0.8][min(skill_type - 1, 2)]
                
                target_hp_ratio = skill.get("target_hp_ratio", 1.0)
                if target_hp_ratio < 0.3:
                    base_reward *= 1.5
                
                reward += base_reward
        
        return reward

    def calculate_positioning_reward(self, main_hero, enemy_hero, main_tower):
        """位置奖励"""
        if not enemy_hero:
            return 0.0
            
        hero_pos = (main_hero["actor_state"]["location"]["x"],
                   main_hero["actor_state"]["location"]["z"])
        enemy_pos = (enemy_hero["actor_state"]["location"]["x"],
                    enemy_hero["actor_state"]["location"]["z"])
        
        distance = math.dist(hero_pos, enemy_pos)
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        enemy_hp_ratio = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
        
        reward = 0.0
        
        # 根据血量调整理想距离
        if hero_hp_ratio < 0.3:
            ideal_distance = 8000  # 低血量保持距离
            if distance < 5000:
                reward -= 1.0  # 太近了
        elif hero_hp_ratio > enemy_hp_ratio * 1.5:
            ideal_distance = 4000  # 优势可以压进
            if distance > 7000:
                reward -= 0.3  # 太保守
        else:
            ideal_distance = 5500  # 正常对线距离
        
        # 距离偏差惩罚
        distance_diff = abs(distance - ideal_distance) / ideal_distance
        reward += max(0, 1 - distance_diff) * 0.2
        
        return reward

    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        # 获取英雄和防御塔
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        
        if not main_hero:
            return
            
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # 获取防御塔和泉水
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    enemy_spring = organ

        frame_no = frame_data["frameNo"]

        # 计算所有奖励
        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            
            # 基础奖励
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            elif reward_name == "tower_hp_point":
                if main_tower:
                    reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
                else:
                    reward_struct.cur_frame_value = 0
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"):
                            reward_struct.cur_frame_value += 1.0
                        elif enemy_hero and (dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                              and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"):
                            reward_struct.cur_frame_value -= 1.0
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward_improved(
                    main_hero, main_tower, enemy_tower)
            
            # 新增密集奖励
            elif reward_name == "damage_exchange":
                reward_struct.cur_frame_value = self.calculate_damage_exchange_reward(
                    main_hero, enemy_hero)
            elif reward_name == "recall_decision":
                # [修改点4] 传递 frame_no
                reward_struct.cur_frame_value = self.calculate_recall_reward(
                    main_hero, main_spring, enemy_hero, frame_no)
            elif reward_name == "health_management":
                reward_struct.cur_frame_value = self.calculate_health_management_reward(main_hero)
            elif reward_name == "combat_timing":
                reward_struct.cur_frame_value = self.calculate_combat_timing_reward(
                    main_hero, enemy_hero)
            elif reward_name == "tower_defense":
                reward_struct.cur_frame_value = self.calculate_tower_defense_reward(
                    main_hero, main_tower, enemy_hero)
            elif reward_name == "skill_hit":
                reward_struct.cur_frame_value = self.calculate_skill_hit_reward(
                    frame_data.get("frame_action", {}))
            elif reward_name == "positioning":
                reward_struct.cur_frame_value = self.calculate_positioning_reward(
                    main_hero, enemy_hero, main_tower)
            # [修改点2] 新增小兵伤害奖励调用
            elif reward_name == "minion_damage":
                reward_struct.cur_frame_value = self.calculate_minion_damage_reward(
                    frame_data, camp)

    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    def calculate_forward_improved(self, main_hero, main_tower, enemy_tower):
        """改进的前进奖励计算"""
        if not main_tower or not enemy_tower:
            return 0.0
        
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (main_hero["actor_state"]["location"]["x"],
                   main_hero["actor_state"]["location"]["z"])
        
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        
        # 血量要求调整为40%
        if hero_hp_ratio > 0.4 and dist_hero2emy < dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
            # 根据血量调整奖励
            forward_value *= hero_hp_ratio
        
        return forward_value

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            # 原有奖励计算
            if reward_name == "hp_point":
                if (self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value)
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value)
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value)
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value)
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
            # [修改点2] 包含 minion_damage
            elif reward_name in ["forward", "last_hit", "damage_exchange", "recall_decision",
                                "health_management", "combat_timing", "tower_defense", 
                                "skill_hit", "positioning", "minion_damage"]:
                # 直接使用计算值
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                # 零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value)
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value)
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        
        reward_dict["reward_sum"] = reward_sum
