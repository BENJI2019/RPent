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

新增准备命令（在指南中的 Linux 仿真环境执行）：

```bash
python -m robots.robocasa.retention --config /data/experiment/configs/pilot1.json \
  init --preset pilot1 --workspace /data/experiment
python -m robots.robocasa.retention --config /data/experiment/configs/pilot1.json download-data
python -m robots.robocasa.retention --config /data/experiment/configs/pilot1.json download-data --execute
CUDA_VISIBLE_DEVICES=0 python -m robots.robocasa.retention \
  --config /data/experiment/configs/pilot1.json doctor --stage all --runtime
```

`init` 不覆盖配置；同一 preset 再做一次实验可加 `--output-root /data/experiment/runs/new-pilot`。
`download-data` 默认预览，仅 `--execute` 才下载选定的 Target Human 数据。
`doctor` 逐项报告文件和环境问题，不启动训练，也不能替代真实模型/仿真测试。
准备命令不会生成锁定协议的 `plan.json`。

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
在具备模型、数据和场景资产的 Linux 服务器上，复制所需配置到 `logs/retention.json`，
修改路径和 GPU 编号，再从 RPent 根目录运行：

```bash
python -m robots.robocasa.retention --config logs/retention.json plan
python -m robots.robocasa.retention --config logs/retention.json evaluate --checkpoint base --mode all
python -m robots.robocasa.retention --config logs/retention.json train --all --resume
python -m robots.robocasa.retention --config logs/retention.json evaluate --checkpoint all --mode all
python -m robots.robocasa.retention --config logs/retention.json summarize
```

`adaptation_mode: "joint"` 将 `adaptation_tasks` 中的任务混合训练成一个模型；
`"independent"` 则分别训练。`adaptation_tasks: null` 选择全部50任务。
`evaluation_tasks` 独立控制评测任务，例如 `{"target": ["OpenDrawer"]}` 只测一个新任务，
`null` 评测完整350个任务。只测新任务不能判断遗忘。
`--tasks` 仅限制当次执行，不改变配置声明的评测分母。

每阶段使用独立输出目录并从同一基础权重开始。训练步数按每次运行计，联合模式不会自动乘50；
混合采样沿用官方数据集长度的0.4次幂权重，评测则对任务等权。
Qwen 服务需要按完整指南单独启动；三个环境的依赖不能混装。
50个专用模型的对角线平均值不代表一个通用模型的 Target50 成绩。

代码入口为 [__main__.py](__main__.py)，π0.5 RPC 接口为
[pi05_server.py](../pi05_server.py)。验证情况见 [VALIDATION.md](VALIDATION.md)。
