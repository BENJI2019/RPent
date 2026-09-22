从零开始：A800 上的 π0.5 训练与评测
===================================

本流程面向 **单卡 A800 80GB 的前期试验** 和 **单机8卡 A800 80GB 的正式实验**。
先跑 direct，再接入两个 Qwen。研究设定、任务划分和统计解释见
:doc:`robocasa_retention`。本页命令在 Linux Bash 中执行，不能直接粘贴到 Windows PowerShell。

0. 确认服务器与当前代码
-----------------------

把包含本次新增文件的完整 RPent 源码同步到服务器，例如 ``~/RPent``。
只重新克隆上游仓库不一定包含这些尚未发布的功能。下文始终在源码根目录运行，
``robots`` 集成不作为独立 wheel 安装。

.. code-block:: bash

   cd "$HOME/RPent"
   cat /etc/os-release
   nvidia-smi
   test -f robots/robocasa/retention/prepare.py
   df -h .

确认能看到 A800，GPU 未被其他训练占满，磁盘能容纳示范视频、模型缓存及训练 checkpoint。
官方 OpenPI 建议训练使用至少80GB显存；这不是当前 batch 和每种方法都一定能运行的保证。
Linux 发行版未知时先看上面的输出。若是 Ubuntu/Debian 且缺少系统依赖，可执行：

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git git-lfs curl build-essential cmake pkg-config \
     ffmpeg libegl1 libgl1 libgl1-mesa-dri libglib2.0-0
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   uv --version

其他发行版请安装等价系统包。NVIDIA 驱动须支持下面 OpenPI 固定的 CUDA12 JAX
运行时；检查以 ``doctor --runtime`` 实际识别 GPU 为准。
不需要为这个实验安装 RLDX-1 权重或 RPent 的 LIBERO OpenPI fork。

1. 统一目录和环境变量
---------------------

下列命令在同一 Bash 会话顺序执行。建议把 export 行保存为自己的环境文件，
之后每次登录和启动新终端都加载。磁盘不够时可把 ``EXP_HOME`` 换成数据盘上的绝对路径。

.. code-block:: bash

   export RPENT_DIR="$HOME/RPent"
   export EXP_HOME="$HOME/robocasa-retention"
   export OPENPI_DIR="$EXP_HOME/openpi"
   export SIM_PY="$EXP_HOME/sim-env/bin/python"
   export OPENPI_PY="$OPENPI_DIR/.venv/bin/python"
   export QWEN_PY="$EXP_HOME/qwen-env/bin/python"
   export ROBOCASA_ASSETS_PATH="$EXP_HOME/assets"
   export HF_HOME="$EXP_HOME/cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   mkdir -p "$EXP_HOME/configs" "$EXP_HOME/env" "$RPENT_DIR/logs"
   cd "$RPENT_DIR"

数据与代码分开放置：

.. code-block:: text

   RPent/                          当前项目源码
   robocasa-retention/
     sim-env/                      仿真、Harness 和实验命令
     openpi/.venv/                 官方 OpenPI、JAX 训练与策略推理
     qwen-env/                     可选：vLLM planner 服务
     assets/                       仿真场景和对象
     datasets/v1.0/target/          仅所选 Target Human 示范
     checkpoints/robocasa365/       固定版本的 Human300 基础权重
     configs/                      各阶段 JSON 和 doctor 报告
     runs/pilot1/                  训练记录、checkpoint、评测和汇总
     env/                          软件版本记录

2. 安装仿真和 Harness 环境
--------------------------

.. code-block:: bash

   uv venv "$EXP_HOME/sim-env" --python 3.11
   uv pip install --python "$SIM_PY" -e "$RPENT_DIR[test,robocasa-pi05]" \
     -c "$RPENT_DIR/robots/robocasa/retention/runtime-constraints.txt" \
     --torch-backend auto
   uv pip check --python "$SIM_PY"
   "$SIM_PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   "$SIM_PY" -m robots.robocasa.retention --help

Torch 检查应显示 CUDA 可用及实际可见的 GPU 数量。``robocasa-pi05`` 现在显式声明
仿真启动需要的 Torch；不要把 CPU 版 Torch 当作已完成 GPU 配置。
安装器根据驱动选择 backend，解析后的实际版本需保存，见最后一节。

下载仿真资产：

.. code-block:: bash

   "$EXP_HOME/sim-env/bin/robocasa-download-assets" \
     --assets-path "$ROBOCASA_ASSETS_PATH" --no-macros --skip-existing -y

保留所有 attribution 文件和下载清单。中断后重跑该命令；资源完整性仍需通过第6步的
真实环境测试确认。``--no-macros`` 适用于下面的新数据下载入口，它会自行设置数据根目录。

3. 安装独立的官方 OpenPI 环境
-----------------------------

.. code-block:: bash

   git clone https://github.com/robocasa-benchmark/openpi "$OPENPI_DIR"
   git -C "$OPENPI_DIR" checkout 5a6beda9ff99da30b4e1b59320f6a32971d7c397
   cd "$OPENPI_DIR"
   GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
   uv pip freeze --python "$OPENPI_PY" | sed '/^-e /d' \
     > "$EXP_HOME/env/openpi-before-rpent.constraints.txt"
   uv pip install --python "$OPENPI_PY" -e "$RPENT_DIR[robocasa-pi05]" \
     -c "$EXP_HOME/env/openpi-before-rpent.constraints.txt"
   uv pip check --python "$OPENPI_PY"
   cd "$RPENT_DIR"
   CUDA_VISIBLE_DEVICES=0 "$OPENPI_PY" -c "import jax; print(jax.__version__, jax.devices())"

OpenPI 固定了 JAX、Torch、Transformers 等依赖。这里保留其解析结果，过滤不能作为
constraints 的 editable 行后再安装 RPent；不要把仿真环境的 constraints 强加给 OpenPI，
也不要在它里面升级 Qwen 所需的 Transformers。安装 RPent 后不要再次运行会移除额外包的
``uv sync``；若重建 OpenPI 环境，请重做这一整节。

4. 下载基础权重并生成单卡配置
-----------------------------

HF 下载工具放在独立工具环境中，不改 OpenPI 的依赖：

.. code-block:: bash

   uv tool install huggingface_hub
   hf download robocasa/robocasa365_checkpoints \
     --revision c484448aba1a9b60a04c9b0ca117241518ea69f3 \
     --include "pi05_pretrain_human300/multitask_learning/75000/**" \
     --local-dir "$EXP_HOME/checkpoints/robocasa365"

这里需要包含 ``params/`` 和 ``assets/**/norm_stats.json`` 的 JAX checkpoint，
不是只含 safetensors 的其他模型目录。数据是公开的；若本机的 HF 配置要求认证，
在自己的终端使用 ``hf auth login``，不要把凭据写进实验 JSON。

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     init --preset pilot1 --workspace "$EXP_HOME"

``init`` 根据本节目录生成绝对路径，记录当前 ``SIM_PY``，把 env/VLA GPU 都设为0，
不会覆盖已有配置，也不会开始训练或下载。检查生成的 JSON：pilot1 是 OpenDrawer
单任务、1,000步、batch16、direct 模式、3个评测 seed，加三个旧任务探针。
如需 LoRA、改变 batch 或学习率，应在第7步 plan 前修改 JSON。
重复做同一 preset 的新实验时，配置文件和 ``output_root`` 都必须换成新路径。
``init --output-root /绝对路径/新实验`` 可直接指定新输出目录。

如果从一开始就要比较两个 Qwen，可在上述 init 命令加入 ``--with-planners``，
并完成第9步后再评测。不要训练完成后直接修改 ``modes``，否则协议身份会变化。

5. 按配置下载 Target Human 数据
-------------------------------

.. code-block:: bash

   # 只列出任务与保存位置，不下载，不锁定实验协议
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" download-data
   # 确认列表后执行实际下载
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" download-data --execute

pilot1 只下载 OpenDrawer 的 Target Human 示范；旧任务探针只用于仿真评测，不下载示范。
此命令复用 RoboCasa 下载器，在子进程中设置 ``DATASET_BASE_PATH``，不修改 site-packages。
它核对安装版本的注册表与固定任务路径，禁止自动覆盖不完整目录，并在上游下载失败但
正常返回时仍报告失败。默认下载该任务完整的500条示范；``demo_fraction`` 控制训练取样，
不会减少下载量。不会下载 Pretraining Human/MimicGen，也不要使用上游 ``--all``。

存在但不完整的目录不会被当作成功：先检查报错路径与剩余磁盘空间，保存需要的文件，
把该具体目录移到备份位置后再重试；不要删除整个 datasets 根目录。
检查覆盖基本元数据，以及 parquet 和三个相机的视频文件数量是否与 total_episodes 一致，
不是全量校验和或解码证明。

OpenPI 首次使用还需要 PaliGemma tokenizer，可提前用同一用户预取：

.. code-block:: bash

   "$OPENPI_PY" -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

这是 OpenPI 的独立缓存，不包含在上面的 HF 权重子目录里；首次需要访问官方 tokenizer 存储。

6. 训练前检查与真实接口测试
---------------------------

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/pilot1.json" doctor --stage all --runtime

所有检查应为 PASS。结果保存为配置旁的 ``pilot1.doctor.json``。
省略 ``--runtime`` 只检查本地文件、路径与源码 revision，不能证明 CUDA 或 Python 依赖可用。
``--stage train`` 不要求仿真资产；``--stage evaluate`` 不要求训练数据。
doctor 不加载完整权重、不优化、不执行仿真、不调用 Qwen。

接着执行现有的真实组件测试：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$EXP_HOME/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
   export PI05_PYTHON="$OPENPI_PY"
   RPENT_RUN_PI05_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m pytest \
     tests/e2e_tests/robocasa/test_pi05_retention.py -v --timeout=1200

该测试会真实加载 π0.5、执行有限动作，并用离线 planner 检查 Harness 工具和清理；
任务成功不是通过条件。Qwen 的图像/工具能力需要另测。单卡首次加载和编译可能较慢。

7. 单任务：baseline → 训练 → 评测 → 汇总
----------------------------------------

下面所有命令使用同一份配置。plan 是正式实验的配置锁定点；完成后不要修改训练/评测设置
或源码再向同一输出目录写结果。若需要先试运行2步，请另建配置、另设 output_root，
再将该试跑配置的 ``train_steps`` 设为2；它不代表学会任务。

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" plan
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     evaluate --checkpoint base --mode direct
   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/pilot1.json" train --all --resume
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     evaluate --checkpoint all --mode direct
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" summarize

默认 direct 配置的 plan 为1个适应模型、24个回合（含 baseline）。如果已启用两个 Qwen，
则为72个回合，仅完成 direct 时 summarize 返回2，需完成第9步后才会完整。
训练调用官方 OpenPI 优化器；
训练过程在终端输出 loss 等日志，首次 JAX 编译期间可能暂时没有训练步输出。
成功后应出现 ``runs/pilot1/training/full/OpenDrawer/seed_0/checkpoint.json``，
其中记录最终权重绝对路径。训练步数1000对应最终目录名999，不要误找1000。
``--resume`` 会跳过已完成的训练，并尝试恢复未完成运行已保存的状态；尚未保存的步数需重跑。

查看 ``runs/pilot1/summary/summary.json``、``comparisons.csv`` 和 ``per_task.csv``。
pilot 的 ``complete=true`` 仅表示声明的子集已完成，``full_benchmark_complete`` 应为 false。
summarize 返回2表示有缺失配对，分数不会自动忽略这些回合。
评测返回1表示基础设施问题，先检查相应 ``worker.log``、``retention_result.json`` 和服务日志再重试。
正常任务失败也会留下完整结果，不能通过反复重试成功的方式替换它。

8. 三任务试验与8卡联合50任务
----------------------------

三任务先在单卡运行，后续步骤与第5–7步相同，只换配置路径：

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot3-joint.json" \
     init --preset pilot3-joint --workspace "$EXP_HOME"

需要三次独立微调时把 preset 和文件名换为 ``pilot3-independent``。
它们从基础 Human300 开始，不继承 pilot1 checkpoint。

转到8卡服务器时，复制当前代码和需要的数据，在新主机重新建立环境并生成本机绝对路径配置。
八卡**协同训练一个联合模型**，不能对同一个训练 ID 启动8个训练进程：

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" \
     init --preset joint50 --workspace "$EXP_HOME" --fsdp-devices 8
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" download-data --execute
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/joint50.json" doctor --stage train --runtime
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" plan
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/joint50.json" train --all --resume

这里 ``fsdp_devices=8``，``batch_size=16`` 是全局 batch，不是每卡16。
可见 GPU 数必须被 fsdp_devices 整除，全局 batch 必须被可见 GPU 数整除。
该配置默认包括 direct 和两个 Qwen：若暂时只研究策略，应在 plan 前改为 ``modes: ["direct"]``。
训练和评测进程顺序运行，训练期间不要让 Qwen 服务占用相同 GPU。
默认共1个适应模型，三种模式合计21,000回合；baseline 可在训练前或后跑，但必须用同一协议。

最后的独立50次使用 ``init --preset independent50``，为配置与输出保留独立目录。
``train --all`` 会依次执行50次；每次都从基础权重初始化，当前入口没有自动并行调度训练。

9. 接入 Qwen3-VL-4B 与 Qwen3.5-4B
---------------------------------

先确保对应模式已在计划中。如果第4步用了默认 direct 配置，可按下面命令创建一个新的
agent 试验；它复用已下载的数据，但按新协议重新训练和评测。若第4步已加入
``--with-planners``，则将 AGENT_CONFIG 指向原 pilot1.json，跳过本段创建、训练和 direct 评测。

.. code-block:: bash

   export AGENT_CONFIG="$EXP_HOME/configs/pilot1-agent.json"
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" \
     init --preset pilot1 --workspace "$EXP_HOME" --with-planners \
     --output-root "$EXP_HOME/runs/pilot1-agent"
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" plan
   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$AGENT_CONFIG" train --all --resume
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode direct

只需安装一次独立 Qwen 环境；不使用量化，以免引入额外变量：

.. code-block:: bash

   uv venv "$EXP_HOME/qwen-env" --python 3.12
   uv pip install --python "$QWEN_PY" -e "$RPENT_DIR" vllm --torch-backend auto
   uv pip check --python "$QWEN_PY"
   hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
   hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

HF 下载不指定 local-dir，模型会进入共同的 HF 缓存，服务按固定 model ID/revision 加载。
所有终端须使用第1步相同的 HF_HOME。先启动一个服务（保持前台运行）：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$QWEN_PY" -m robots.robocasa.retention.serve_planner \
     --config "$AGENT_CONFIG" --mode qwen3_vl_4b

另一个终端加载相同环境变量，并从 RPent 根目录检查和评测：

.. code-block:: bash

   export OPENAI_API_KEY=local
   export AGENT_CONFIG="$EXP_HOME/configs/pilot1-agent.json"
   "$EXP_HOME/sim-env/bin/rpent-check-llm" --planner api \
     --model openai-chat:Qwen/Qwen3-VL-4B-Instruct --base-url http://127.0.0.1:8000/v1
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" \
     evaluate --checkpoint base --mode qwen3_vl_4b --tasks OpenDrawer --max-cells 1

文本连接成功后，用这一真实回合的 agent 轨迹检查图像输入、工具调用及动作执行。
``--max-cells`` 只限制执行，不缩小统计分母。通过后去掉两个限制，
用下面命令完成该模式的前后评测：

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode qwen3_vl_4b

在服务终端用 Ctrl-C 停止第一个 Qwen，再启动第二个：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$QWEN_PY" -m robots.robocasa.retention.serve_planner \
     --config "$AGENT_CONFIG" --mode qwen35_4b

在评测终端执行：

.. code-block:: bash

   "$EXP_HOME/sim-env/bin/rpent-check-llm" --planner api \
     --model openai-chat:Qwen/Qwen3.5-4B --base-url http://127.0.0.1:8001/v1
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode qwen35_4b
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" summarize

``init`` 对两个 planner 都设置 ``gpu_memory_utilization=0.4``、
``max_num_seqs=1``、``max_model_len=32768``，作为 A800 单卡共用的起点。
π0.5 推理已关闭 JAX 预分配，但单卡 agent 运行仍需同时容纳 Qwen、π0.5 和仿真器，
必须实测峰值。长 agent 轨迹仍可能超过上下文限制；需在正式计划前根据 pilot 调整上下文、
显存与回合预算，不能通过降低正式评测预算掩盖失败。

10. 八卡分片评测
----------------

一种分配方式是 GPU0–5 各跑一个 env+π0.5 worker，GPU6、7 分别提供两个 Qwen 服务。
先停止训练，并用 joint50 配置在独立终端启动 Qwen（把第9步的 GPU 改为6、7）。
以下生成6份仅改变 GPU 分配的配置；这些字段不影响 protocol_id：

.. code-block:: bash

   "$SIM_PY" - <<'PY'
   import json, os
   from pathlib import Path
   from robots.robocasa.retention.protocol import Experiment
   root = Path(os.environ["EXP_HOME"]) / "configs"
   original = root / "joint50.json"
   cfg = json.loads(original.read_text())
   for gpu in range(6):
       shard = {**cfg, "env_cuda_device": gpu, "vla_cuda_device": gpu}
       path = root / f"joint50-gpu{gpu}.json"
       with path.open("x") as file:
           json.dump(shard, file, indent=2)
       assert Experiment.load(path).protocol_id == Experiment.load(original).protocol_id
   PY

.. code-block:: bash

   pids=()
   for gpu in 0 1 2 3 4 5; do
     "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50-gpu$gpu.json" \
       evaluate --checkpoint all --mode all --num-shards 6 --shard-index "$gpu" \
       > "$RPENT_DIR/logs/joint50-gpu$gpu.log" 2>&1 &
     pids+=("$!")
   done
   failed=0
   for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
   test "$failed" -eq 0
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" summarize

先 plan 再启动 worker；同一分片不能重复启动。worker 共享只读 checkpoint 和 planner 服务，
各自拥有独立 π0.5 服务及 episode 记忆。共享 planner 的吞吐可能限制速度，6个 worker
不代表必然加速6倍。失败后重跑对应分片会跳过已完成回合；残留 ``.running`` 锁只可在
确认原进程已结束后处理。

11. 保存版本与排错
------------------

在正式 plan 前完成软件版本确定，随后保存这三份记录及完整源码快照：

.. code-block:: bash

   uv pip freeze --python "$SIM_PY" > "$EXP_HOME/env/simulator.txt"
   uv pip freeze --python "$OPENPI_PY" > "$EXP_HOME/env/openpi.txt"
   # 安装了 Qwen 环境时执行
   uv pip freeze --python "$QWEN_PY" > "$EXP_HOME/env/qwen.txt"
   git -C "$OPENPI_DIR" rev-parse HEAD > "$EXP_HOME/env/openpi-revision.txt"
   git -C "$RPENT_DIR" rev-parse HEAD > "$EXP_HOME/env/rpent-head.txt"
   git -C "$RPENT_DIR" status --short > "$EXP_HOME/env/rpent-working-tree.txt"

RPent 和 RoboCasa 分支安装可能包含未发布修改，只有 HEAD 或 pip freeze 不能替代完整源码快照。
``plan.json`` 另含源文件和实验配置的协议指纹。Qwen 服务记录实际 vLLM/Torch/Transformers
版本及启动参数。

.. list-table:: 常见问题
   :header-rows: 1
   :widths: 30 70

   * - 现象
     - 检查
   * - OpenPI revision / import path 不符
     - 配置应指向本页固定 fork 及其 Python；不要复用 LIBERO 环境。
   * - JAX 只显示 CPU 或 Torch CUDA=false
     - 检查驱动、可见 GPU 和实际 Python；重新处理依赖，不开始正式训练。
   * - 找不到归一化文件
     - checkpoint 应指到75000这一层，并完整下载 assets；不从旧训练数据重算。
   * - CUDA OOM
     - 停止同卡其他服务；训练时在新协议中降低全局 batch/使用 LoRA 或8卡 FSDP；单卡 planner 检查显存比例和上下文长度。
   * - EGL / navview / 对象文件错误
     - 检查外置 assets、MUJOCO_GL、RPent robosuite 分支和真实组件测试。
   * - 已有目录属于另一协议
     - 使用新的 output_root；不要删除已有 plan 绕过检查。
   * - summarize 返回2
     - 检查 missing_pairs；补齐声明的全部 checkpoint、模式与任务，修复基础设施错误。

本仓库的离线测试覆盖上述配置、下载边界、失败报告与统计契约。
当前开发主机没有执行 A800 真实安装、训练和任务评测；doctor 通过也不等于已验证任务成功率。
参考 `OpenPI 固定版本说明 <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_、
`RoboCasa 数据说明 <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_、
`uv 安装 <https://docs.astral.sh/uv/getting-started/installation/>`_ 与
`vLLM Qwen 部署说明 <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>`_。
