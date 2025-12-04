#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class GameConfig:
    # 改进的奖励权重设计 - 完整生存策略系统
    REWARD_WEIGHT_DICT = {
        # ===== 基础奖励 =====
        "hp_point": 2.5,            # 降低，避免过于保守
        "money": 0.1,               # 金币奖励
        "exp": 0.1,                 # 经验奖励
        "last_hit": 2.5,            # 补刀奖励提升
        
        # [修改点2] 新增：对小兵造成伤害奖励 (鼓励清线)
        "minion_damage": 0.15,
        
        # ===== 战斗奖励 =====
        "kill": 6.0,                # 击杀奖励
        "death": -5.0,              # 死亡惩罚
        "ep_rate": -0.2,            # 能量管理
        
        # ===== 推进奖励 =====
        "forward": 0.5,             
        "tower_hp_point": 12.0,     # 塔血保护
        
        # ===== 密集奖励系统 =====
        "damage_exchange": 0.5,      # 伤害交换效率
        "skill_hit": 1.5,           # 技能命中
        
        # ===== 生存策略奖励（核心） =====
        "recall_decision": 4.0,      # 回城决策
        "health_management": 2.0,    # 血量管理
        "combat_timing": 1.5,       # 战斗时机选择
        "tower_defense": 3.0,       # 塔防奖励
        "positioning": 1.0,         # 位置奖励
    }
    
    # 动态权重调整配置
    ADAPTIVE_WEIGHT = {
        "early_game": {  # 前5000帧
            "money": 1.5,
            "exp": 1.5,
            "death": 0.7,           # 早期降低死亡惩罚
            "recall_decision": 1.5,  # 早期强化生存意识
            "health_management": 1.5,
        },
        "mid_game": {    # 5000-10000帧
            "tower_defense": 1.5,
            "damage_exchange": 1.5,
            "combat_timing": 1.5,
        },
        "late_game": {   # 10000帧后
            "tower_hp_point": 8.0,
            "forward": 5.0,
            "kill": 1.5,
        }
    }
    
    # 回城配置
    RECALL_CONFIG = {
        "low_hp_threshold": 0.30,        # 需要考虑回城的血量
        "critical_hp_threshold": 0.1 ,   # 必须回城的血量
        "safe_hp_threshold": 0.7,        # 安全血量
        "spring_distance": 2000,         # 泉水判定距离
        "combat_distance": 6000,         # 战斗距离判定
        "retreat_frames": 30,            # 撤退评估帧数
    }
    
    # 战斗配置
    COMBAT_CONFIG = {
        "ideal_distance_aggressive": 4000,  # 激进距离
        "ideal_distance_normal": 5500,      # 正常对线距离
        "ideal_distance_defensive": 8000,   # 防守距离
        "hp_advantage_threshold": 0.3,      # 血量优势阈值
        "hp_disadvantage_threshold": -0.2,  # 血量劣势阈值
    }
    
    TIME_SCALE_ARG = 0
    MODEL_SAVE_INTERVAL = 1800

# 维度配置
class DimConfig:
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    DIM_OF_ORGAN_1_2 = [18, 18]
    DIM_OF_ORGAN_3_4 = [18, 18]
    DIM_OF_HERO_FRD = [235]
    DIM_OF_HERO_EMY = [235]
    DIM_OF_HERO_MAIN = [14]
    DIM_OF_GLOBAL_INFO = [25]

# 模型和算法配置
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
    
    # 学习率配置（略微调整）
    INIT_LEARNING_RATE_START = 8e-4  # 略微降低学习率
    TARGET_LR = 8e-5
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
    
    # ===== Dual-Clip参数 =====
    DUAL_CLIP_PARAM_C = 0.75  # 从3.0降低到0.75，平衡探索与利用
    
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

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
    
    # 动作映射（鲁班7号相关）
    ACTION_MAPPING = {
        "move": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 移动方向
        "attack": 9,        # 普攻
        "skill_1": 10,      # 技能1
        "skill_2": 11,      # 技能2
        "skill_3": 12,      # 技能3（大招）
        "recall": 13,       # 回城
        "no_op": 14,        # 不操作
    }
    
    # 鲁班技能配置
    HERO_SKILL_CONFIG = {
        "skill_1_range": 5000,   # 技能1射程
        "skill_2_range": 6000,   # 技能2射程
        "skill_3_range": 8000,   # 大招射程
        "normal_attack_range": 4500,  # 普攻距离
    }