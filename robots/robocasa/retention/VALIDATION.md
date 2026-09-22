# 验证记录（2026-09-21）

本次修改位于 `codex/robocasa-pi05-retention`，基线为 `cb2eb7d`。
这些记录验证软件契约，不表示已获得任何机器人任务成功率。

## 环境与运行步骤完善

新增 `init`、按实验配置下载 Target Human 数据的 `download-data` 和训练前 `doctor`，
并增加 A800 单卡/8卡中英文操作指南。仿真 extra 补充显式 Torch 依赖；Qwen 服务支持
配置显存占比、上下文和并发数。生成配置使用 GPU0、单请求并发、32,768上下文和0.4显存占比，
尚需在 A800 上验证实际峰值与吞吐。

| 检查 | 本次结果 |
| --- | --- |
| Linux RoboCasa、配置/注册相关检查 | 91 passed、1 skipped、28 deselected；真实 GPU 用例因缺少资产跳过 |
| Windows retention、准备入口、VLA、Target50 检查 | 73 passed；最后的 planner 参数调整后，准备入口13项再次通过 |
| 中英文 Sphinx，`-W --keep-going` | 两种语言通过 |
| 指南命令检查 | 每种语言25组 Bash 块语法通过，内嵌 Python 可编译，两种语言的执行命令一致；未执行安装/下载/训练块 |
| 实际命令入口 | 五种 preset 的 init、下载预览和 plan 通过；缺少资源时 doctor 返回1且逐项报告；准备命令未创建 plan 或下载数据 |
| 全库 Ruff 检查、格式检查和 `git diff --check` | 通过，246个 Python 文件格式符合要求 |

新增测试覆盖配置不覆盖、Target 数据范围、注册表路径不符、上游静默下载失败、
已有部分下载保护、按 total_episodes 检查文件数量、训练/评测的不同前置条件，
以及8卡 FSDP、全局 batch 和实际可见 GPU 数的匹配关系。
下载器与 CUDA 探针使用离线替身测试；不能据此声称已经完成 Box 下载或 A800 训练。

日志为 `logs/retention-setup-focused-tests.log`、`logs/retention-setup-docs-{en,zh}.log`、
`logs/retention-setup-commands.log`；入口检查产物在 `logs/retention-setup-cli-h0po4sav/`。
本次没有重跑下文完整离线套件或安装新的 GPU 环境。官方包的下载入口、依赖文件和
tokenizer 来源已在线核对；实际服务器安装仍以指南的 pip check、doctor 和真实接口测试为准。

## 分阶段实验功能

新增独立/联合训练、声明式评测子集，以及单任务、三任务、联合50任务配置。
验证覆盖每个方法/训练 seed 的模型数量、固定基础权重、混合任务数据路径、
缺失数据拒绝、评测分片、子集标签、排除已适应旧任务、缺失配对分母和预算。

| 检查 | 本次结果 |
| --- | --- |
| RoboCasa、配置/注册及可选 GPU 用例（Linux、Python 3.12） | 78 passed、1 skipped、28 deselected；跳过的是需要真实资产的 GPU 用例 |
| retention、VLA、Target50 协议检查（Windows、Python 3.12） | 60 passed，其中 retention 35 项 |
| 五份预置配置的实际 CLI `plan` | 单任务/三任务联合/三任务独立/联合50/独立50的模型数分别为1/1/3/1/50，回合数为24/36/72/21,000/535,500 |
| 三份 pilot 的空结果 `summarize` | 均返回2，所有分数为空，未宣称完整评测完成 |
| 中英文 Sphinx 构建，`-W --keep-going` | 两种语言通过 |
| 全库 Ruff 检查、格式检查及 `git diff --check` | 通过；243个 Python 文件格式符合要求 |

Linux 日志位于 `logs/retention-staged-focused-tests.log` 和
`logs/retention-staged-docs-{en,zh}.log`；命令入口检查产物位于
`logs/retention-staged-cli-zhyna1bu/`。这些本地验证产物不纳入版本控制。
本次未重跑全库单元测试，之前的完整检查及既有失败保留在下表。
联合数据配置通过离线测试；没有将模拟配置测试称为真实 OpenPI 多任务训练成功。

## 此前基础功能验证

| 检查 | 结果 |
| --- | --- |
| RoboCasa、机器人配置/注册相关检查（Linux、Python 3.12） | 58 passed；真实 GPU 用例 1 skipped；其余机器人 28 deselected |
| 新实验、原 VLA 与 Target50 协议检查（Windows、Python 3.12） | 40 passed |
| 全库离线检查（Linux 快照，shell 文件规范化为 LF） | 601 passed、4 failed、3 skipped |
| 原始 HEAD 上复查上述失败所在文件 | 相同 4 项失败，31 项通过 |
| 官方协议命令入口 | 成功生成 50 次独立适应、535,500 回合的计划；空结果汇总返回状态码 2 且没有伪造分数 |
| 中英文 Sphinx 构建，`-W --keep-going` | 两种语言通过 |
| Ruff 0.15.22，全库检查及格式检查 | 通过，243 个 Python 文件格式符合要求 |
| `git diff --check` | 通过 |
| `pre-commit run --all-files` | 获取 GitHub hook 仓库时连接失败；改用同版本 Ruff 执行对应全库检查 |

四项既有失败分别是两个 LIBERO 启动器检查顺序断言、一个离线 planner
5 秒超时，以及 LIBERO memory-ablation 渲染缺少 `output_dir`。
已在未修改的 HEAD 快照上复现，没有修改其他机器人的代码或削弱测试。
离线环境还缺少部分 Flywheel 可选依赖，因此不能声称完整 CI 矩阵通过。

未执行：真实 OpenPI checkpoint 加载、GPU SFT/LoRA 训练、RoboCasa 真实动作链、
Qwen 图像和工具调用。当前主机没有本实验所需的权重、示范和场景资产。
GPU 测试保留显式开关 `RPENT_RUN_PI05_INTEGRATION=1`；部署步骤见中英文指南。
OpenPI 配置、动作接口和 checkpoint 来源已按固定官方源码版本核对，
但源码核对不能替代上述运行验证。

主要检查命令：

```bash
python -m pytest tests/unit_tests/robots/robocasa \
  tests/unit_tests/rpent/robots/test_config_contracts.py \
  tests/unit_tests/rpent/robots/test_registry_contracts.py \
  tests/e2e_tests/robocasa/test_pi05_retention.py -k robocasa -q
python -m pytest tests/unit_tests -q
ruff check --preview .
ruff format --check .
sphinx-build -W --keep-going docs/source-zh docs/build/html-zh
sphinx-build -W --keep-going docs/source-en docs/build/html-en
```
