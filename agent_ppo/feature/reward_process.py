#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
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
        # 缓存上一次关键量，用于构造增量奖励
        self._last_enemy_tower_hp = None
        self._last_main_hero_hp = None

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        frame_no = frame_data["frameNo"]
        self.m_last_frame_no = frame_no

        # 动态调整奖励权重（按对局阶段）
        self.update_reward_weights(frame_no)

        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Compute phase mixing weights (early / mid / late) with a smooth transition
    # 计算前中后期的混合权重，实现阶段之间的线性平滑过渡
    def _get_phase_mix_weight(self, frame_no):
        early_end = getattr(GameConfig, "EARLY_GAME_END_FRAME", 2400)
        mid_end = getattr(GameConfig, "MID_GAME_END_FRAME", 7200)
        window = getattr(GameConfig, "TRANSITION_WINDOW", 0)

        weights = {"early": 0.0, "mid": 0.0, "late": 0.0}
        if window <= 0:
            if frame_no <= early_end:
                weights["early"] = 1.0
            elif frame_no <= mid_end:
                weights["mid"] = 1.0
            else:
                weights["late"] = 1.0
            return weights

        # early -> mid transition
        if frame_no <= early_end - window:
            weights["early"] = 1.0
        elif frame_no < early_end + window:
            ratio = (frame_no - (early_end - window)) / (2.0 * window)
            ratio = max(0.0, min(1.0, ratio))
            weights["early"] = 1.0 - ratio
            weights["mid"] = ratio
        # mid (stable)
        elif frame_no <= mid_end - window:
            weights["mid"] = 1.0
        # mid -> late transition
        elif frame_no < mid_end + window:
            ratio = (frame_no - (mid_end - window)) / (2.0 * window)
            ratio = max(0.0, min(1.0, ratio))
            weights["mid"] = 1.0 - ratio
            weights["late"] = ratio
        else:
            weights["late"] = 1.0
        return weights

    # Update per-item reward weights according to game phase
    # 根据当前对局阶段，动态调整各奖励项权重
    def update_reward_weights(self, frame_no):
        phase_mix = self._get_phase_mix_weight(frame_no)
        phase_cfg = getattr(GameConfig, "PHASE_WEIGHT_MULTIPLIERS", {})

        # 计算每个奖励项的最终乘数
        reward_scale = {}
        for name in GameConfig.REWARD_WEIGHT_DICT.keys():
            scale = 0.0
            for phase, alpha in phase_mix.items():
                phase_dict = phase_cfg.get(phase, {})
                scale += alpha * phase_dict.get(name, 1.0)
            if scale <= 0.0:
                scale = 1.0
            reward_scale[name] = scale

        # 应用到各个 calc_frame_map 中
        for calc_map in (self.m_cur_calc_frame_map, self.m_main_calc_frame_map, self.m_enemy_calc_frame_map):
            for reward_name, reward_struct in calc_map.items():
                base_weight = GameConfig.REWARD_WEIGHT_DICT.get(reward_name, reward_struct.weight)
                reward_struct.weight = base_weight * reward_scale.get(reward_name, 1.0)

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
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

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(
                    main_hero, main_tower, enemy_tower, frame_data["frameNo"]
                )
            # 清兵线奖励（只在有己方英雄信息时计算）
            elif reward_name == "minion_clear":
                reward_struct.cur_frame_value = self.calculate_minion_clear(
                    main_hero, enemy_hero, npc_list, frame_data, camp
                )
            # 推塔奖励
            elif reward_name == "attack_tower":
                reward_struct.cur_frame_value = self.calculate_attack_tower(
                    main_hero, npc_list, enemy_tower, camp
                )
            # 血量管理 / 吃血包奖励
            elif reward_name == "health_management":
                reward_struct.cur_frame_value = self.calculate_health_management(main_hero, frame_data, camp)

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower, frame_no):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        hp_ratio = main_hero["actor_state"]["hp"] / max(1.0, float(main_hero["actor_state"]["max_hp"]))
        # 根据当前阶段平滑插值得到前压血量阈值
        thresholds = getattr(GameConfig, "FORWARD_HP_THRESHOLDS", {"early": 0.65, "mid": 0.55, "late": 0.45})
        phase_mix = self._get_phase_mix_weight(frame_no)
        hp_threshold = 0.0
        total_alpha = 0.0
        for phase, alpha in phase_mix.items():
            if phase in thresholds:
                hp_threshold += alpha * thresholds[phase]
                total_alpha += alpha
        if total_alpha > 0:
            hp_threshold /= total_alpha
        else:
            hp_threshold = thresholds.get("early", 0.65)

        if hp_ratio > hp_threshold and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # 清兵线奖励：鼓励在安全距离下高效清理敌方小兵，同时惩罚离兵线过近或过远
    def calculate_minion_clear(self, main_hero, enemy_hero, npc_list, frame_data, camp):
        if main_hero is None:
            return 0.0

        cfg = getattr(GameConfig, "MINION_CLEAR_CONFIG", {})
        radius = cfg.get("detection_radius", 10.0)
        optimal_min = cfg.get("optimal_distance_min", 3.0)
        optimal_max = cfg.get("optimal_distance_max", 3.8)
        danger_distance = cfg.get("danger_distance", 2.0)
        far_distance = cfg.get("far_distance", 8.5)

        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )

        nearby_minion_distances = []
        for npc in npc_list:
            try:
                if npc.get("sub_type") != "ACTOR_SUB_SOLDIER":
                    continue
                if npc.get("camp") == camp:
                    continue
                if npc.get("hp", 0) <= 0:
                    continue
                npc_pos = (npc["location"]["x"], npc["location"]["z"])
                dist = math.dist(hero_pos, npc_pos)
                if dist <= radius:
                    nearby_minion_distances.append(dist)
            except Exception:
                continue

        if not nearby_minion_distances:
            return 0.0

        base_reward_per_minion = 0.25
        ideal_distance_bonus = 0.6
        too_close_penalty = -0.3
        close_penalty_slope = 0.2
        far_reward_max = 0.6
        too_far_penalty = -0.02
        enemy_hero_danger_radius = 8.0
        enemy_hero_danger_penalty = -0.15

        count = len(nearby_minion_distances)
        avg_distance = sum(nearby_minion_distances) / float(count)

        # 基础奖励：周围敌方小兵越多，奖励越高
        reward = count * base_reward_per_minion

        # 距离修正：保持在理想清兵距离附近
        if optimal_min <= avg_distance <= optimal_max:
            reward += ideal_distance_bonus
        elif avg_distance < danger_distance:
            reward += too_close_penalty
        elif avg_distance < optimal_min:
            # 略近于理想距离，按线性方式给予惩罚
            gap = optimal_min - avg_distance
            reward -= close_penalty_slope * gap
        elif avg_distance > far_distance:
            reward += too_far_penalty
        else:
            # 略远于理想距离，鼓励“站在兵线后方”但不宜过远
            span = max(1e-6, far_distance - optimal_max)
            ratio = (avg_distance - optimal_max) / span
            ratio = max(0.0, min(1.0, ratio))
            reward += ratio * far_reward_max

        # 敌方英雄压制风险：敌方英雄太近时适当扣分
        if enemy_hero is not None:
            enemy_pos = (
                enemy_hero["actor_state"]["location"]["x"],
                enemy_hero["actor_state"]["location"]["z"],
            )
            dist_enemy = math.dist(hero_pos, enemy_pos)
            if dist_enemy <= enemy_hero_danger_radius:
                reward += enemy_hero_danger_penalty

        return reward

    # 推塔奖励：根据塔血下降量及带兵线推塔情况给予奖励/惩罚
    def calculate_attack_tower(self, main_hero, npc_list, enemy_tower, camp):
        if main_hero is None or enemy_tower is None:
            return 0.0

        cfg = getattr(GameConfig, "TOWER_ATTACK_CONFIG", {})
        attack_range = cfg.get("attack_range", 8.0)
        safe_attack_range = cfg.get("safe_attack_range", 6.5)
        min_minion_count = cfg.get("min_minion_count", 2)
        hp_damage_scale = cfg.get("hp_damage_scale", 45.0)

        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        dist = math.dist(hero_pos, tower_pos)

        # 统计己方兵线数量
        friend_minion_count = 0
        for npc in npc_list:
            try:
                if npc.get("sub_type") != "ACTOR_SUB_SOLDIER":
                    continue
                if npc.get("camp") != camp:
                    continue
                if npc.get("hp", 0) <= 0:
                    continue
                npc_pos = (npc["location"]["x"], npc["location"]["z"])
                if math.dist(tower_pos, npc_pos) <= attack_range:
                    friend_minion_count += 1
            except Exception:
                continue

        in_range = dist <= attack_range
        if not in_range:
            return 0.0

        damage_reward_with_minions_scale = 1.2
        damage_reward_without_minions_scale = 0.4
        safe_range_stay_bonus = 0.4
        naked_tower_penalty = -0.2

        reward = 0.0
        current_hp = enemy_tower.get("hp", 0)
        last_hp = self._last_enemy_tower_hp
        if last_hp is None:
            last_hp = current_hp
        hp_reduced = max(0.0, float(last_hp - current_hp))
        self._last_enemy_tower_hp = current_hp

        if hp_reduced > 0 and hp_damage_scale > 0:
            base = hp_reduced / hp_damage_scale
            if friend_minion_count >= min_minion_count:
                reward += base * damage_reward_with_minions_scale
            else:
                reward += base * damage_reward_without_minions_scale

        # 有兵线、在安全距离内靠塔，给额外稳定奖励
        if friend_minion_count >= min_minion_count and dist <= safe_attack_range:
            reward += safe_range_stay_bonus

        # 无兵线硬抗塔，给少量惩罚
        if friend_minion_count == 0 and current_hp > 0:
            reward += naked_tower_penalty

        return reward

    # 血量管理奖励：根据当前血量、血包位置和回血行为构造奖励
    def calculate_health_management(self, main_hero, frame_data, camp):
        if main_hero is None:
            return 0.0

        cfg = getattr(GameConfig, "HEALTH_MGMT_CONFIG", {})
        critical_hp_ratio = cfg.get("critical_hp_ratio", 0.25)
        low_hp_ratio = cfg.get("low_hp_ratio", 0.4)
        near_dist = cfg.get("health_pack_near", 3.5)
        mid_dist = cfg.get("health_pack_mid", 7.0)
        hp_recovery_threshold = cfg.get("hp_recovery_threshold", 100.0)

        hero_state = main_hero["actor_state"]
        hero_pos = (hero_state["location"]["x"], hero_state["location"]["z"])
        hero_hp = hero_state["hp"]
        hero_max_hp = max(1.0, float(hero_state["max_hp"]))
        hp_ratio = hero_hp / hero_max_hp

        # 最近血包距离
        cakes = frame_data.get("cakes", [])
        nearest_cake_dist = float("inf")
        for cake in cakes:
            try:
                collider = cake.get("collider", {})
                center = collider.get("center", {})
                cake_pos = (center.get("x", 0.0), center.get("z", 0.0))
                dist = math.dist(hero_pos, cake_pos)
                if dist < nearest_cake_dist:
                    nearest_cake_dist = dist
            except Exception:
                continue

        reward = 0.0

        # 危急血量状态
        critical_near_reward = 2.5
        critical_mid_reward = 1.3
        critical_far_penalty = -0.4
        # 低血量状态
        low_hp_near_reward = 1.5
        low_hp_mid_reward = 0.7
        low_hp_far_penalty = -0.2
        # 回血行为奖励
        recovery_near_bonus = 1.8

        if hp_ratio <= critical_hp_ratio:
            if nearest_cake_dist <= near_dist:
                reward += critical_near_reward
            elif nearest_cake_dist <= mid_dist:
                reward += critical_mid_reward
            else:
                reward += critical_far_penalty
        elif hp_ratio <= low_hp_ratio:
            if nearest_cake_dist <= near_dist:
                reward += low_hp_near_reward
            elif nearest_cake_dist <= mid_dist:
                reward += low_hp_mid_reward
            else:
                reward += low_hp_far_penalty

        # 检测本帧是否发生了显著回血（例如吃到血包）
        last_hp = self._last_main_hero_hp
        if last_hp is None:
            last_hp = hero_hp
        if hero_hp - last_hp >= hp_recovery_threshold and nearest_cake_dist <= near_dist:
            reward += recovery_near_bonus
        self._last_main_hero_hp = hero_hp

        return reward

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
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

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
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
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
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
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in ("minion_clear", "attack_tower", "health_management"):
                # 新增非零和奖励：直接使用主角侧的当前帧值
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
