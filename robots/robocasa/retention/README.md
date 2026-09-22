# RoboCasa365 π0.5 能力保持实验

从官方 **π0.5 Human300** 权重出发，对选定的 Target Human 任务独立或联合微调，
评测下游任务与旧任务的性能变化。训练不读取旧任务示范。

完整的设定分析、数据来源、依赖安装、运行命令和统计解释见
[中文实验说明](../../../docs/source-zh/rst_source/usage/robocasa_retention.rst)
或 [English guide](../../../docs/source-en/rst_source/usage/robocasa_retention.rst)。

**第一次上服务器，请先看 [从零部署与训练（中文）](../../../docs/source-zh/rst_source/usage/robocasa_retention_setup.rst)**
（[English walkthrough](../../../docs/source-en/rst_source/usage/robocasa_retention_setup.rst)）。
它按单卡 A800 80GB → 单机8卡 A800 的顺序，给出系统检查、三个独立环境、场景/权重/数据下载、
真实接口测试、训练恢复、Qwen 服务、分片评测和排错步骤。

本机代码目录为 `/home/ma-user/work/algorithm/hyy_vla/rpent`。
按指南配置 `cluster/cluster.env`，三个运行 Conda 环境和独立工具环境位于
`/home/ma-user/work/dataset/Common_wl/envs/`，分别记录本机实际的绝对 prefix。
工作目录、uv/pip/Conda 缓存、检查点和输出默认放到
`/home/ma-user/work/dataset/Common_wl/hyy_vla_retention/`；
代码包内不创建环境、缓存、数据或训练结果。

在本地已激活仿真环境后，从项目根目录执行：

```bash
CLUSTER="$PWD/robots/robocasa/retention/cluster"
bash "$CLUSTER/run_local.sh" init pilot1 pilot1
bash "$CLUSTER/run_local.sh" cli pilot1 download-data --execute
bash "$CLUSTER/run_local.sh" cli pilot1 doctor --stage all --runtime
bash "$CLUSTER/run_local.sh" evaluate pilot1 direct --checkpoint base
bash "$CLUSTER/run_local.sh" train pilot1 pilot1
bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
bash "$CLUSTER/run_local.sh" cli pilot1 summarize
```

`init` 不覆盖配置；`train PRESET NAME` 对同名配置自动使用 `--resume`。
中断后保持相同实验名并重跑训练，恢复已保存的优化器状态，跳过已完成模型。
`save_interval` 控制中间保存频率；pilot 默认100步，完整50任务模板默认1000步。
上游没有保存数据加载器游标，不能保证恢复后采样顺序与不中断训练完全一致。

只改了 Harness 或评测时，不必重训相同模型：

```bash
bash "$CLUSTER/run_local.sh" cli pilot1 status
# 停止旧作业与 planner；确认 training_matches=true 后执行。
bash "$CLUSTER/run_local.sh" cli pilot1 refresh-plan
bash "$CLUSTER/run_local.sh" train pilot1 pilot1
bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
```

`refresh-plan` 核对训练身份，保留 training/checkpoints，把旧评测、服务记录、汇总和计划
归档到 `history/<旧协议ID>`。改变训练任务、学习率、batch、FSDP 或训练适配代码需要新实验。
旧版缺少 training_id 的计划不自动迁移。评测自动跳过完整回合；只有确认原进程已停止，
才能清理崩溃遗留的 `.running` 标记。

MTP 由你在平台提交 `hyy_vla` 算法包，包含 `rpent/vendor/openpi` 及其完整 Git 元数据、
`rpent/robots/robocasa/retention/cluster/cluster.env`。单机8卡训练的启动命令为：

```bash
bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh train joint50 joint50
```

不要在本地执行 MTP 入口。它先验证平台信息，再配置软链、CUDA12.8、NCCL 和同一 Conda 环境。
一个 JAX 进程协同使用8卡；不支持多节点。相同提交命令恢复相同实验；本地改代码后需重新提交快照。
评测用 `evaluate joint50 direct`、`evaluate joint50 qwen3_vl_4b`、
`evaluate joint50 qwen35_4b`，每个模式单独提交，planner 与 worker 由同一作业管理。
完整命令、下载、版本保存和故障处理以部署指南为准。

- 训练：全参数 SFT / LoRA；单任务、多个任务联合或逐任务独立训练，均从同一基础权重开始。
- 评测：直接执行 π0.5、Qwen3-VL-4B-Instruct + Harness、Qwen3.5-4B + Harness。
- 范围：可只测指定任务，也可评测完整 Target50 + Pretrain300；子集结果明确标注。
- 控制：固定归一化、配对场景、总动作预算、独立空记忆和基础设施错误处理。
- 指标：被适应任务、Target50、旧300、旧266、重合34及 Target 分组，输出逐任务变化。
- 边界：衡量遗忘和 agent 补偿，不保证权重绝不遗忘；未实现 RL 训练。

建议先做小规模试验，再联合50任务，最后按研究需要做独立50次；50次独立微调不是前置要求。
以下数量均按一种训练方法、一个训练 seed 计算，评测回合数包含基础权重：

| 配置 | 训练结果 | 评测回合 |
| --- | --- | --- |
| [单任务 pilot](examples/pilot1.json) | 1个模型 | 24 |
| [三任务联合 pilot](examples/pilot3-joint.json) | 1个模型 | 36 |
| [三任务独立 pilot](examples/pilot3-independent.json) | 3个模型 | 72 |
| [联合50任务](examples/joint50.json) | 1个模型 | 21,000 |
| [独立50任务](example.json) | 50个模型 | 535,500 |

pilot 使用 direct 模式、3个评测 seed、1,000步训练，并加入三个旧任务探针；
完整配置使用三种模式、10个评测 seed、10,000步训练。
请使用上述 init 入口生成配置；表中的 JSON 是预置参数参考，不要直接复制旧的服务器路径执行。
`adaptation_mode: "joint"` 混合指定任务训练一个模型，`"independent"` 分别训练。
`adaptation_tasks: null` 选择全部50任务；`evaluation_tasks` 独立控制评测范围，
例如 `{"target": ["OpenDrawer"]}` 只测一个下游任务，`null` 评测完整350个任务。
只测新任务不能判断遗忘。`--tasks` / `--max-cells` 仅限制执行，不改变声明分母。

每阶段使用独立输出目录并从同一基础权重开始。联合训练步数不会自动乘50，
混合采样沿用官方数据集长度的0.4次幂权重，评测对任务等权。
50个专用模型的对角线平均值不代表一个通用模型的 Target50 成绩。
本项目实现 SFT/LoRA 遗忘测量与 agent 补偿基线，不承诺权重绝不遗忘；尚未实现 RL 训练。

代码入口为 [__main__.py](__main__.py)，恢复逻辑为 [recovery.py](recovery.py)，
启动脚本位于 [cluster](cluster)，验证记录见 [VALIDATION.md](VALIDATION.md)。
