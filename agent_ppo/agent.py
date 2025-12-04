#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import math
import numpy as np

from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from agent_ppo.conf.conf import Config, GameConfig
from kaiwu_agent.utils.common_func import attached
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        # predict local or remote
        # 本地预测或远程预测
        self.is_predict_remote = True

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        # ===== 生存策略相关状态 =====
        # 生存状态机：normal / need_retreat / retreating / recalling / recovering
        self.survival_state = "normal"
        self.state_counter = 0
        self.last_hp_ratio = 1.0
        self.recall_cooldown = 0
        self.last_recall_frame = 0
        self.continuous_low_hp_frames = 0
        self.last_distance_to_spring = float("inf")
        self.combat_frames = 0

        # 战斗状态
        self.in_combat = False
        self.last_damage_frame = 0
        self.enemy_distance = float("inf")

        # 动作历史与技能后普攻（[修改点1]）
        self.action_history = []
        self.max_history_length = 10
        self.pending_normal_attack = False

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, observation):
        # The remote prediction will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 远程预测不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            player_id = observation["player_id"]
            camp = observation["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the observation returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(observation)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, False)
        return [ActData(action=action)]

    @exploit_wrapper
    def exploit(self, observation):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            player_id = observation["player_id"]
            camp = observation["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the observation returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(observation)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return [ActData(d_action=d_action)]

    def train_predict(self, observation):
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        if self.is_predict_remote:
            act_data, model_version = self.predict(observation)
            return act_data[0].action

        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def eval_predict(self, observation):
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        if self.is_predict_remote:
            act_data, model_version = self.exploit(observation)
            return act_data[0].d_action

        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def action_process(self, observation, act_data, is_stochastic):
        """
        对模型输出的动作进行后处理，加入生存 / 撤退 / 回城等规则。
        """
        # 基础动作：训练用随机采样，评估用贪心
        action = act_data.action if is_stochastic else act_data.d_action

        # --------------------------------------------------------
        # [修改点1] 技能后强制平 A（最高优先级）
        # 上一帧释放技能，则本帧强制普攻
        # --------------------------------------------------------
        if self.pending_normal_attack:
            action[0] = Config.ACTION_MAPPING["attack"]
            self.pending_normal_attack = False
            self._record_action(action[0])
            return action

        # 提取生存相关信息
        hero_state = self._extract_hero_state(observation)
        hp_ratio = hero_state.get("hp_ratio", 1.0)
        spring_distance = hero_state.get("spring_distance", float("inf"))
        enemy_distance = hero_state.get("enemy_distance", float("inf"))

        # 血包信息（[修改点2]）
        pack_available = hero_state.get("pack_available", False)
        pack_distance = hero_state.get("pack_distance", float("inf"))
        pack_direction = hero_state.get("pack_direction", 4)

        # 更新战斗状态
        self._update_combat_state(hero_state)

        # 更新回城冷却
        if self.recall_cooldown > 0:
            self.recall_cooldown -= 1

        # --------------------------------------------------------
        # 1. 极低血量强制回城
        # --------------------------------------------------------
        critical_hp = GameConfig.RECALL_CONFIG["critical_hp_threshold"]
        if (
            hp_ratio < critical_hp
            and not self.in_combat
            and self.recall_cooldown == 0
        ):
            action[0] = Config.ACTION_MAPPING["recall"]
            self.recall_cooldown = 300
            self.survival_state = "recalling"
            if self.logger:
                self.logger.info(f"Emergency recall at {hp_ratio:.1%} HP")

        # --------------------------------------------------------
        # 2. 低血量(<40%)优先吃塔下血包  [修改点2]
        # --------------------------------------------------------
        elif (
            hp_ratio < 0.40
            and pack_available
            and pack_distance < 6000
            and not self.in_combat
        ):
            action[0] = pack_direction

        # --------------------------------------------------------
        # 3. 低血量撤退 [修改点4：阈值从0.35改为0.30]
        # --------------------------------------------------------
        elif hp_ratio < 0.30 and self.survival_state == "normal":
            self.survival_state = "need_retreat"
            # 优先使用位移技能（如鲁班 2 技能）
            if hero_state.get("skill_2_ready", False):
                action[0] = Config.ACTION_MAPPING["skill_2"]
            else:
                self._move_towards_spring(action, hero_state)

        # --------------------------------------------------------
        # 4. 泉水恢复状态：要求血量到 90% 才能离开泉水
        # --------------------------------------------------------
        elif self.survival_state == "recovering":
            if hp_ratio > 0.9:
                self.survival_state = "normal"
            elif spring_distance > GameConfig.RECALL_CONFIG["spring_distance"]:
                # 离泉水太远，强制朝泉水移动
                self._move_towards_spring(action, hero_state)
            else:
                # 泉水范围内且血量不足 90%，强制待机，避免“站一秒就走”
                action[0] = Config.ACTION_MAPPING["no_op"]

        # --------------------------------------------------------
        # 5. 正常战斗状态下，根据血量控制激进程度
        # --------------------------------------------------------
        elif self.survival_state == "normal":
            # [修改点3] 有小兵和敌方英雄时，优先攻击敌方英雄
            normal_attack_range = Config.HERO_SKILL_CONFIG["normal_attack_range"]
            if enemy_distance < normal_attack_range:
                action[0] = Config.ACTION_MAPPING["attack"]
            # 低血量近战时采取保守拉扯策略
            elif hp_ratio < 0.5 and enemy_distance < 5000:
                self._apply_conservative_strategy(action, hero_state)

        # --------------------------------------------------------
        # 技能释放检测：本帧如果施放技能，下帧强制普攻  [修改点1]
        # --------------------------------------------------------
        skill_actions = [
            Config.ACTION_MAPPING["skill_1"],
            Config.ACTION_MAPPING["skill_2"],
            Config.ACTION_MAPPING["skill_3"],
        ]
        if action[0] in skill_actions:
            self.pending_normal_attack = True

        self._record_action(action[0])
        return action

    def observation_process(self, observation):
        """
        将环境 observation 转为 ObsData，并附加生存相关特征（不改动原始特征向量维度）。
        """
        feature_vec = observation["observation"]
        legal_action = observation["legal_action"]

        obs_data = ObsData(
            feature=feature_vec,
            legal_action=legal_action,
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

        # 解析英雄与环境信息，为生存策略提供输入
        hero_state = self._extract_hero_state(observation)
        obs_data.hp_ratio = hero_state.get("hp_ratio", 1.0)
        obs_data.spring_distance = hero_state.get("spring_distance", float("inf"))
        obs_data.enemy_distance = hero_state.get("enemy_distance", float("inf"))
        obs_data.spring_direction = hero_state.get("spring_direction", 4)
        obs_data.enemy_direction = hero_state.get("enemy_direction", 0)
        obs_data.skill_2_ready = hero_state.get("skill_2_ready", False)

        # 血包信息（[修改点2]）
        obs_data.pack_available = hero_state.get("pack_available", False)
        obs_data.pack_distance = hero_state.get("pack_distance", float("inf"))
        obs_data.pack_direction = hero_state.get("pack_direction", 4)

        return obs_data

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(player_id)

        # 生存策略状态重置
        self.survival_state = "normal"
        self.state_counter = 0
        self.last_hp_ratio = 1.0
        self.recall_cooldown = 0
        self.last_recall_frame = 0
        self.continuous_low_hp_frames = 0
        self.last_distance_to_spring = float("inf")
        self.combat_frames = 0
        self.in_combat = False
        self.last_damage_frame = 0
        self.enemy_distance = float("inf")
        self.action_history = []
        self.pending_normal_attack = False

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

        # 记录最近一帧的血量与泉水距离，供状态机使用
        if hasattr(obs_data, "hp_ratio"):
            self.last_hp_ratio = obs_data.hp_ratio
        if hasattr(obs_data, "spring_distance"):
            self.last_distance_to_spring = obs_data.spring_distance

    # ===== 生存策略辅助函数 =====

    def _extract_hero_state(self, observation):
        """提取当前帧英雄及环境的关键信息。"""
        hero_state = {}

        frame_state = observation.get("frame_state", {})
        hero_states = frame_state.get("hero_states", [])

        # 我方英雄信息
        for hero in hero_states:
            if hero.get("player_id") == self.player_id:
                actor_state = hero.get("actor_state", {})
                hp = actor_state.get("hp", 1)
                max_hp = actor_state.get("max_hp", 1)
                hero_state["hp_ratio"] = hp / max_hp if max_hp > 0 else 1.0
                hero_state["position"] = (
                    actor_state.get("location", {}).get("x", 0),
                    actor_state.get("location", {}).get("z", 0),
                )

                # 技能 CD 状态（假定索引 1 为位移 / 控制技能）
                skills = hero.get("skills", [])
                if len(skills) > 1:
                    hero_state["skill_2_ready"] = skills[1].get("cd", 0) == 0
                break

        # 距离信息（泉水 / 敌人 / 血包）
        hero_state.update(self._calculate_distances(observation, hero_state.get("position")))

        # 战斗信息
        hero_state["taking_damage"] = observation.get("taking_damage", False)
        hero_state["frame_no"] = frame_state.get("frameNo", 0)

        return hero_state

    def _calculate_distances(self, observation, hero_pos):
        """计算英雄到泉水、敌方英雄以及血包的距离与方向。"""
        distances = {
            "spring_distance": float("inf"),
            "enemy_distance": float("inf"),
            "tower_distance": float("inf"),
            "spring_direction": 4,
            "enemy_direction": 0,
            "pack_distance": float("inf"),
            "pack_direction": 0,
            "pack_available": False,
        }

        if not hero_pos:
            return distances

        frame_state = observation.get("frame_state", {})

        # 泉水与血包
        npc_states = frame_state.get("npc_states", [])
        for npc in npc_states:
            camp = npc.get("camp")
            sub_type = npc.get("sub_type")
            loc = npc.get("location", {})
            pos = (loc.get("x", 0), loc.get("z", 0))

            # 我方泉水
            if camp == self.hero_camp and sub_type == "ACTOR_SUB_CRYSTAL":
                distances["spring_distance"] = self._calculate_distance(hero_pos, pos)
                distances["spring_direction"] = self._calculate_direction(hero_pos, pos)

            # 我方血包（[修改点2]）
            elif camp == self.hero_camp and sub_type == "ACTOR_SUB_HP_MOD":
                dist = self._calculate_distance(hero_pos, pos)
                if dist < distances["pack_distance"]:
                    distances["pack_distance"] = dist
                    distances["pack_direction"] = self._calculate_direction(hero_pos, pos)
                    distances["pack_available"] = True

        # 敌方英雄
        hero_states = frame_state.get("hero_states", [])
        for hero in hero_states:
            actor_state = hero.get("actor_state", {})
            if actor_state.get("camp") != self.hero_camp:
                loc = actor_state.get("location", {})
                enemy_pos = (loc.get("x", 0), loc.get("z", 0))
                distances["enemy_distance"] = self._calculate_distance(hero_pos, enemy_pos)
                distances["enemy_direction"] = self._calculate_direction(hero_pos, enemy_pos)
                break

        return distances

    def _calculate_distance(self, pos1, pos2):
        """计算二维平面距离。"""
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def _calculate_direction(self, from_pos, to_pos):
        """将 from_pos 指向 to_pos 的方向量化到 0-7。"""
        dx = to_pos[0] - from_pos[0]
        dz = to_pos[1] - from_pos[1]
        angle = math.atan2(dz, dx)
        # 将角度划分为 8 个扇区
        direction = int((angle + math.pi) / (math.pi / 4)) % 8
        return direction

    def _update_combat_state(self, hero_state):
        """根据敌人距离和受击情况更新战斗 / 脱战状态。"""
        enemy_distance = hero_state.get("enemy_distance", float("inf"))
        taking_damage = hero_state.get("taking_damage", False)

        if enemy_distance < GameConfig.RECALL_CONFIG["combat_distance"] or taking_damage:
            self.in_combat = True
            self.combat_frames += 1
            if taking_damage:
                self.last_damage_frame = hero_state.get("frame_no", 0)
        else:
            # 超过 3 秒未受伤则视为脱战
            frames_since_damage = hero_state.get("frame_no", 0) - self.last_damage_frame
            if frames_since_damage > 90:
                self.in_combat = False
                self.combat_frames = 0

        self.enemy_distance = enemy_distance

    def _move_towards_spring(self, action, hero_state):
        """将动作修改为朝泉水方向移动。"""
        spring_direction = hero_state.get("spring_direction", 4)
        action[0] = spring_direction
        return action

    def _apply_conservative_strategy(self, action, hero_state):
        """在低血量情况下面对近距离敌人时采取后撤拉扯策略。"""
        enemy_direction = hero_state.get("enemy_direction", 0)
        enemy_distance = hero_state.get("enemy_distance", float("inf"))

        if enemy_distance < 4000:
            opposite_direction = (enemy_direction + 4) % 8
            action[0] = opposite_direction

        return action

    def _record_action(self, action_id):
        """记录最近的动作序列，便于后续策略调试与扩展。"""
        self.action_history.append(action_id)
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def load_model_local(self, model_file_path, idx):
        # When loading the local model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载本地模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        self.is_predict_remote = False
        self.model.load_state_dict(
            torch.load(
                model_file_path,
                map_location=torch.device("cpu"),
            )
        )
        self.logger.info(f"agent {idx} load model {model_file_path} successfully")
