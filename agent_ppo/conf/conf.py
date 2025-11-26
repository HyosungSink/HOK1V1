#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 9.0,
        "money": 0.012,
        "exp": 0.009,
        "ep_rate": 0.06,
        "death": -2.5,
        "kill": 0.8,
        "last_hit": 1.3,
        "forward": 0.009,
        # Extra shaping rewards (only used on agent side, not zero-sum)
        "minion_clear": 1.6,
        "attack_tower": 3.5,
        "health_management": 1.3,
    }
    # Phase-aware weight multipliers. Final weight = REWARD_WEIGHT_DICT[key] * phase_multiplier
    # 分阶段奖励权重系数：最终权重 = 基础权重 * 对应阶段乘数
    PHASE_WEIGHT_MULTIPLIERS = {
        "early": {
            "money": 1.35,
            "exp": 1.3,
            "last_hit": 1.25,
            "minion_clear": 1.2,
            "death": 1.15,
            "kill": 0.65,
            "attack_tower": 0.6,
            "forward": 0.85,
            "tower_hp_point": 0.85,
        },
        "mid": {
            "money": 1.05,
            "exp": 1.05,
            "last_hit": 1.0,
            "minion_clear": 1.0,
            "death": 1.0,
            "kill": 1.0,
            "attack_tower": 1.05,
            "forward": 1.0,
            "tower_hp_point": 1.0,
        },
        "late": {
            "money": 0.7,
            "exp": 0.7,
            "last_hit": 0.75,
            "minion_clear": 0.8,
            "death": 0.9,
            "kill": 1.4,
            "attack_tower": 1.5,
            "forward": 1.15,
            "tower_hp_point": 1.35,
        },
    }
    # Frame index (time) configuration for phase splitting and smooth transition
    # 用帧号划分前中后期，并在相邻阶段之间做线性平滑过渡
    EARLY_GAME_END_FRAME = 2400
    MID_GAME_END_FRAME = 7200
    TRANSITION_WINDOW = 600

    # Forward reward hp thresholds for different phases
    # 不同对局阶段前压所需的血量阈值
    FORWARD_HP_THRESHOLDS = {
        "early": 0.65,
        "mid": 0.55,
        "late": 0.45,
    }

    # Minion clearing heuristics
    # 清兵逻辑相关阈值
    MINION_CLEAR_CONFIG = {
        "detection_radius": 10.0,
        "optimal_distance_min": 3.0,
        "optimal_distance_max": 3.8,
        "danger_distance": 2.0,
        "far_distance": 8.5,
    }

    # Tower attack related configuration
    # 推塔相关阈值
    TOWER_ATTACK_CONFIG = {
        "attack_range": 8.0,
        "safe_attack_range": 6.5,
        "min_minion_count": 2,
        "hp_damage_scale": 45.0,
    }

    # Health / blood management configuration (health packs etc.)
    # 血量管理相关配置（吃血包等）
    HEALTH_MGMT_CONFIG = {
        "critical_hp_ratio": 0.25,
        "low_hp_ratio": 0.4,
        "health_pack_near": 3.5,
        "health_pack_mid": 7.0,
        "hp_recovery_threshold": 100.0,
    }

    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    DIM_OF_ORGAN_1_2 = [18, 18]
    DIM_OF_ORGAN_3_4 = [18, 18]
    DIM_OF_HERO_FRD = [235]
    DIM_OF_HERO_EMY = [235]
    DIM_OF_HERO_MAIN = [14]
    DIM_OF_GLOBAL_INFO = [25]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        810,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(725 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for ppo is 15584,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中ppo的维度是15584
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
