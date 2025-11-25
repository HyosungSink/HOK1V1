# HOK1V1 强化学习项目（基于腾讯开悟框架）

本项目是在腾讯开悟强化学习开发框架之上，面向王者荣耀 1v1 场景（`hok1v1`）的强化学习示例工程。  
代码遵循开悟框架的标准目录与编程接口，包含：

- 官方 PPO 示例智能体：`agent_ppo`（完整可训练）
- DIY 自定义智能体模板：`agent_diy`（留给开发者二次开发）
- 统一的训练 / 评估配置：`conf/`
- 一步训练自检脚本：`train_test.py`

项目依赖腾讯 Kaiwu 分布式强化学习系统（`kaiwudrl`、`kaiwu_agent` 等），用于完成智能体与环境交互、样本采集、分布式训练、模型下发和评估。

---

## 一、整体训练流程（结合开悟框架）

根据《腾讯开悟强化学习框架》文档，一个完整训练流程包括：

1. **智能体 – 环境循环交互**
   - 环境通过 `reset()` / `step()` 输出观测 `obs`、奖励 `reward` 等原始数据；
   - 智能体将原始观测处理为特征（`ObsData`）；
   - 调用预测函数（本地或远程），生成动作 `ActData`；
   - 将动作转换为环境协议格式，调用 `env.step(actions)` 推进一帧。

2. **样本处理**
   - 一局从开始到结束称为一个 episode；
   - 每一帧的交互构成一条样本，整局轨迹组成样本序列；
   - 对轨迹做优势函数/回报计算，打包为标准训练样本 `SampleData`，送入 learner。

3. **模型迭代优化**
   - Learner 从 Reverb（或 ZMQ）拉取 `SampleData`；
   - 调用算法实现（如 PPO），计算 loss，反向传播并更新模型参数。

4. **智能体模型更新**
   - 训练好的模型文件写入 model pool；
   - Aisrv / Actor 周期性从 model pool 拉取最新模型；
   - 智能体加载新模型，与环境继续交互，实现「交互–训练–更新」闭环。

本项目的 `agent_ppo` 对上述流程给出了完整实现；`agent_diy` 提供骨架模板，方便你对特征、奖励、模型、算法和工作流进行自定义开发。

---

## 二、项目目录结构

项目根目录结构（省略无关文件）：

```text
HOK1V1/
├── agent_diy/                 # DIY 智能体模板
│   ├── agent.py               # 智能体核心入口（预测 / 训练接口骨架）
│   ├── algorithm/
│   │   └── algorithm.py       # 算法逻辑骨架（loss、优化流程留空）
│   ├── conf/
│   │   ├── conf.py            # GameConfig / Config 模板
│   │   └── train_env_conf.toml# 环境配置模板（阵营、对手、评估间隔等）
│   ├── feature/
│   │   └── definition.py      # ObsData / ActData / SampleData 等数据结构模板
│   ├── model/
│   │   └── model.py           # 模型骨架（留给开发者实现网络结构）
│   └── workflow/
│       └── train_workflow.py  # 训练工作流骨架（如何与环境循环、收集样本）
│
├── agent_ppo/                 # PPO 示例智能体（完整实现，可直接训练）
│   ├── agent.py               # 智能体主类，封装预测/评估/训练入口
│   ├── algorithm/
│   │   └── algorithm.py       # PPO 算法实现（loss 计算与优化逻辑）
│   ├── conf/
│   │   ├── conf.py            # PPO 相关超参数、维度配置
│   │   └── train_env_conf.toml# PPO 训练环境配置（对手、自对弈/通用 AI、阵营等）
│   ├── feature/
│   │   ├── definition.py      # ObsData / ActData / SampleData 定义与样本打包逻辑
│   │   └── reward_process.py  # HOK1V1 奖励设计（血量、推塔、经验等多项奖励组合）
│   ├── model/
│   │   └── model.py           # 实际的网络结构（多分支 MLP + LSTM + 多头策略 + 值函数）
│   └── workflow/
│       └── train_workflow.py  # 训练工作流（多局循环、样本收集、模型同步等）
│
├── conf/
│   ├── algo_conf_hok1v1.toml         # 算法–模型–工作流映射（ppo / diy）
│   ├── app_conf_hok1v1_selfplay.toml # 自对弈模式下的 app 配置
│   ├── app_conf_hok1v1_noselfplay.toml# 非自对弈模式 app 配置
│   ├── configure_app.toml            # 应用级总配置（app 名称、algo、框架模式等）
│   └── kaiwudrl/
│       ├── actor.toml                # Actor 进程配置
│       ├── aisrv.toml                # Aisrv 进程配置
│       ├── client.toml               # Client 配置
│       └── learner.toml              # Learner 进程配置（训练超参、Reverb 等）
│
├── kaiwu.json                 # 评估/对战所用模型池配置（对手模型 id 等）
├── train_test.py              # 一步训练自检脚本（用于验证整体流程是否打通）
├── setup.sh                   # Codex CLI 本地环境配置脚本（与训练逻辑无关）
└── 腾讯开悟强化学习框架.pdf      # 官方框架说明文档（本 README 主要参考的文档）
```

---

## 三、核心模块说明（对照 PDF）

### 1. 智能体目录：`agent_ppo` / `agent_diy`

#### 1.1 `agent.py` —— 智能体入口

- 继承 `kaiwu_agent.agent.base_agent.BaseAgent`，并通过 `@attached` 装饰器注册到框架。
- 主要职责：
  - 维护环境相关状态：`hero_camp`、`player_id`、`game_id` 等；
  - 维护 LSTM 状态、学习率调度器、优化器等（PPO 实现中）；
  - 实现以下关键接口（PDF 中重点提到）：
    - `observation_process(obs, state=None)`：将环境 `obs` 转为 `ObsData`；
    - `action_process(observation, act_data, is_train)`：将智能体动作转为环境协议动作；
    - `predict(observation)` / `exploit(observation)`：预测 / 评估用；  
    - `learn(list_sample_data)`：发送/消费样本，触发训练。

PPO 版本中 `_model_inference` 会将 `ObsData` 中的特征、LSTM 状态整理为张量，调用 `Model.forward` 输出 logits / value / 新 LSTM 状态，并采样动作。

DIY 版本中 `_model_inference` 目前只是示例代码（返回默认 `ActData()`），需要你根据自己的模型实现补充完整。

#### 1.2 `feature/definition.py` —— 特征与样本处理

PDF 对「特征处理」部分有详细描述，本项目对应实现主要集中于：

- **数据结构定义**
  - `ObsData`：智能体预测输入，例如 `feature`、`legal_action`、`lstm_cell`、`lstm_hidden`；
  - `ActData`：预测输出，例如 `action`、`prob`、`value`、LSTM 状态等；
  - `SampleData`：训练样本，内部 `npdata` 对应 learner 读取的一维数组。

- **样本构建与轨迹处理（PPO）**
  - `build_frame(agent, state_dict)`：
    - 从当前帧状态中提取 feature、legal_action、行动 `action`、奖励 `reward`、LSTM 信息等；
    - 组装为框架统一的 `Frame` 对象，方便后续 GAE 等计算。
  - `FrameCollector`：
    - 按帧号顺序缓存每个智能体的 `Frame`；
    - 在 episode 结束后计算 `reward_sum` 和 `advantage`（GAE）；
    - 将一条条轨迹打包为 `SampleData`，作为 PPO 的训练输入。

DIY 版本中，同样提供了 `ObsData` / `ActData` / `SampleData` 模板，你可以根据自己的网络输入输出自由扩展字段。

#### 1.3 `feature/reward_process.py` —— 奖励设计（PPO）

- `GameRewardManager` 根据 HOK1V1 的帧数据计算多种奖励子项，包括：
  - 血量相关（英雄 / 塔）；
  - 经验增长；
  - 伤害 / 击杀；
  - 推进程度（防御塔距离变化）等。
- 提供 `get_reward(frame_data, reward_dict)` 等接口，将多项奖励加权汇总为 `reward_sum`，同时保留子项用于分析。

你在 DIY 智能体中可以参考此文件，自行设计奖励形态。

#### 1.4 `model/model.py` —— 模型结构

- PPO 版本模型：
  - 使用多路 MLP 提取：
    - 己方 / 敌方英雄特征；
    - 双方小兵特征；
    - 双方防御塔特征；
    - 主英雄特征和全局信息；
  - 将各分支拼接后，再经过 MLP + LSTM；
  - 输出：
    - 多头离散动作 logits（按钮、移动、技能、目标等）；
    - 值函数 `V(s)`；
    - 目标嵌入向量，用于动作目标建模。
- DIY 版本模型文件仅保留结构骨架，具体网络由你自行实现。

#### 1.5 `algorithm/algorithm.py` —— 算法实现

- PPO 版本：
  - `learn(list_sample_data)`：
    - 将多条 `SampleData.npdata` 堆叠为 batch；
    - 按 `Config.DATA_SPLIT_SHAPE` 还原各部分数据（特征 / legal_action / LSTM 等）；
    - 调用 `Model.forward` 和 `Model.compute_loss` 完成前向与 loss 计算；
    - 梯度反向传播 + 梯度裁剪 + 学习率调度；
    - 定期向监控上报损失指标（value_loss、policy_loss、entropy_loss）。

- DIY 版本：
  - 该文件目前为空实现，需要你根据自己的算法（DQN、A2C、SAC 等）补充：
    - 如何从 `SampleData` 解析输入；
    - 如何组织前向计算；
    - 如何计算 loss 并更新模型。

#### 1.6 `workflow/train_workflow.py` —— 训练工作流

- PPO 工作流做的事情：
  - 从 `agent_ppo/conf/train_env_conf.toml` 读取并校验环境配置；
  - 通过 `ModelFileSync` 与 model pool 同步模型文件；
  - 在一个无限循环中，不断：
    - 启动新 episode（拉模型、reset 环境和智能体）；
    - 按帧执行 `env.step(actions)`，同时：
      - 调用 `agent.train_predict()` / `agent.eval_predict()`；
      - 通过 `FrameCollector` 保存训练所需帧；
    - episode 结束后，生成 `SampleData`，调用 `agent.learn()` 发送样本；
    - 定期调用 `agent.save_model()` 将模型写入 model pool。

- DIY 工作流目前只有骨架和注释，留给你完全自定义训练逻辑（例如单机训练循环、独立评估流程等）。

---

## 四、配置说明：`conf/`

### 1. 算法配置：`conf/algo_conf_hok1v1.toml`

- 描述「算法名 → 模型类 → 工作流」的映射关系：

```toml
[ppo]
actor_model   = "agent_ppo.agent.Agent"
learner_model = "agent_ppo.agent.Agent"
aisrv_model   = "agent_ppo.agent.Agent"
train_workflow = "agent_ppo.workflow.train_workflow.workflow"

[diy]
actor_model   = "agent_diy.agent.Agent"
learner_model = "agent_diy.agent.Agent"
aisrv_model   = "agent_diy.agent.Agent"
train_workflow = "agent_diy.workflow.train_workflow.workflow"
```

- 当 `configure_app.toml` 中 `algo = "ppo"` 或 `algo = "diy"` 时，框架会据此加载相应智能体和工作流。

### 2. 应用配置：`conf/configure_app.toml`

关键字段：

- `[app]`
  - `app = "hok1v1"`：应用名，对应 HOK 1v1 场景；
  - `self_play = true`：是否自对弈；
  - `algo = "ppo"`：当前使用的算法（可改为 `"diy"` 切换到自定义智能体）；
  - `selfplay_app_conf` / `noselfplay_app_conf`：指向不同的 app 配置文件；
  - `algo_conf = "conf/algo_conf_hok1v1.toml"`：算法映射配置；
  - `use_which_deep_learning_framework = "pytorch"`：使用 PyTorch；
  - `copy_dir = "conf,agent_ppo,agent_diy"`：保存模型时一并打包的目录。

### 3. App 级配置：`conf/app_conf_hok1v1_*.toml`

- `app_conf_hok1v1_selfplay.toml` / `app_conf_hok1v1_noselfplay.toml`：
  - `run_handler`：环境封装器，负责将游戏仿真器转为统一 RL 接口；
  - `rl_helper`：Aisrv 侧 RL 流程辅助模块；
  - `probs_handler`：用于处理 action 概率的工具；
  - `[hok1v1.policies.*]`：定义训练使用的 policy 及对应算法（如 `algo = "ppo"`）。

### 4. 分布式组件配置：`conf/kaiwudrl/*.toml`

- `learner.toml`：learner 的训练超参、Reverb 配置、采样维度等；
- `actor.toml`：actor 侧批次大小、CPU 亲和性等；
- `aisrv.toml`、`client.toml`：分别对应 aisrv 和 client 行为配置。

这些文件通常由平台生成或运维同学维护，开发智能体时一般只需关注 `configure_app.toml`、`algo_conf_hok1v1.toml` 以及各智能体目录下的 `conf`。

---

## 五、train_test.py：一步训练自检

`train_test.py` 用于快速验证当前工程是否能跑通一次完整训练流程：

- 通过环境变量临时覆盖部分训练参数：
  - 减小 `replay_buffer_capacity`、`train_batch_size`；
  - 设置 `max_frame_no`、`dump_model_freq` 等；
- 调用外部脚本：
  - `change_algorithm_all.sh`：切换算法（`ppo` / `diy`）；
  - `change_sample_server.sh`：根据平台架构选择 Reverb 或 ZMQ；
  - 启动 `aisrv`、`learner`、`actor` 等进程；
- 循环检查：
  - 通过日志和 `process_stop.done` 文件判断异常；
  - 监控 `/data/ckpt/{app}_{algo}` 目录下是否产生 `model.ckpt-*.pkl`；
  - 一旦检测到模型文件，即认为训练成功，并清理进程。

> 在运行前需要保证：平台环境、工具脚本（`tools/*.sh`）以及相关容器/服务已按开悟平台要求部署完毕。

---

## 六、如何基于 `agent_diy` 进行自定义开发

建议参考 `agent_ppo` 的实现步骤如下：

1. **设计数据结构**
   - 在 `agent_diy/feature/definition.py` 中：
     - 定义适合你算法的 `ObsData`、`ActData`、`SampleData` 字段；
     - 确保 `SampleData.npdata` 与 `Config.SAMPLE_DIM` 一致。

2. **实现特征与动作处理**
   - 在 `agent_diy/agent.py` 中：
     - 实现 `observation_process`：把环境 `obs` / `state` 转成你定义的 `ObsData`；
     - 实现 `action_process`：把 `ActData` 转成环境协议动作；
     - 实现 `_model_inference`：调用你自己的 `Model` 输出动作和必要信息。

3. **构建模型**
   - 在 `agent_diy/model/model.py` 中实现自定义网络结构；
   - 注意输入维度与 `ObsData.feature` 对齐。

4. **实现算法**
   - 在 `agent_diy/algorithm/algorithm.py` 中实现：
     - `learn(list_sample_data)`：前向 + loss + 反向 + 优化；
     - 必要时增加 target 网络、replay buffer 等逻辑。

5. **自定义工作流**
   - 在 `agent_diy/workflow/train_workflow.py` 中补充训练逻辑：
     - 如何 reset 环境 / 智能体；
     - 如何循环 `env.step`、构建样本、调用 `learn`；
     - 如何保存 / 加载模型。

6. **配置切换到 DIY**
   - 修改 `conf/configure_app.toml` 中：
     - `algo = "diy"`
   - 或在平台的任务配置中选择 `diy` 算法。
   - 运行 `train_test.py` 或直接启动正式训练任务验证流程。

---

## 七、运行与开发建议

- 开发前建议通读《腾讯开悟强化学习框架》PDF 中「特征处理」「模型开发」「算法开发」「工作流开发」四个章节，并对照本工程的 `agent_ppo` 实现理解各接口的使用方式。
- 推荐从以下顺序理解代码：
  1. `agent_ppo/agent.py` → 理解智能体对外接口；
  2. `agent_ppo/feature/definition.py` / `reward_process.py` → 理解数据与奖励；
  3. `agent_ppo/model/model.py` → 理解状态表示与网络结构；
  4. `agent_ppo/algorithm/algorithm.py` → 理解训练细节；
  5. `agent_ppo/workflow/train_workflow.py` → 理解工程级训练流程。
- 完成 DIY 实现后，优先通过 `train_test.py` 进行一次短流程自检，再提交平台长时间训练任务。


