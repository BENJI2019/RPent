单卡调试与 MTP 八卡：部署、训练和断点续跑
================================================

本页遵循仓库的 workspace-paths rules 和 local-vs-mtp skill。实验设定见
:doc:`robocasa_retention`。命令在 Linux Bash 中执行。先在单卡 A800 80GB 上验证，
再由你在平台提交单机8卡 A800 作业。脚本不会代替你提交任务。

目录与执行边界
--------------

算法包按 **hyy_vla** 提交，其中 **rpent/** 是包含本文件对应代码的仓库。
如果平台实际选择的是 rpent 本身作为包根，启动路径中就去掉 rpent 这一层；
脚本内部通过 SCRIPT_DIR 定位，不依赖外层包名。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 用途
     - 本机
     - MTP
   * - RPent 代码
     - /home/ma-user/work/algorithm/hyy_vla/rpent
     - /opt/huawei/schedule-train/algorithm/rpent
   * - OpenPI 代码
     - RPent 内 vendor/openpi
     - 同一提交包内 vendor/openpi
   * - 数据 / 公共环境
     - /home/ma-user/work/dataset
     - /opt/huawei/dataset
   * - 模型 / 用户输出
     - /home/ma-user/work/model
     - /opt/huawei/quoteModel

实验工作目录统一放在用户可写的
**dataset/Common_wl/hyy_vla_retention/**：configs、runtime、uv/pip/Conda
cache、tmp、env、checkpoints、retention_outputs。不要使用无权限的
``model/xiaoyi_tmpstorage``。三个运行环境和独立 ``tools`` 环境放在
``dataset/Common_wl/envs/``；工具环境只安装 uv 与 Hugging Face CLI。
场景和 Target 示范在 dataset/hyy_vla/robocasa365；
algorithm 只放代码与配置模板。
MTP 入口只在平台检查通过后创建路径软链，不覆盖已有真实目录或错误软链。
本地入口完全不执行这组软链操作。

1. 本机检查和路径设置
---------------------

同步包含本功能的完整源码，而非只克隆旧的上游版本。先确认 Linux、驱动、共享盘与系统依赖。
需要 git、git-lfs、编译工具、ffmpeg、EGL/OpenGL；缺少时按实际发行版处理，不能假设 Ubuntu。

.. code-block:: bash

   cd /home/ma-user/work/algorithm/hyy_vla/rpent
   cat /etc/os-release
   nvidia-smi
   df -h /home/ma-user/work/dataset /home/ma-user/work/model
   export RPENT_DIR="$PWD"
   export CLUSTER="$RPENT_DIR/robots/robocasa/retention/cluster"
   test -f "$CLUSTER/cluster.env" || cp "$CLUSTER/cluster.env.example" "$CLUSTER/cluster.env"
   # 现在编辑 cluster.env：将 RETENTION_HOME 和四个环境 prefix 指向有写权限的 Common_wl 路径。
   export RETENTION_CODE_ROOT="$RPENT_DIR"
   source "$CLUSTER/common.sh"
   load_settings
   mkdir -p "$RETENTION_HOME" "$RETENTION_HOME/cache/uv" "$RETENTION_HOME/cache/pip" "$RETENTION_HOME/cache/conda/pkgs"
   mkdir -p "$(dirname "$RETENTION_TOOLS_PREFIX")"
   test -w "$RETENTION_HOME" || { echo "RETENTION_HOME 不可写: $RETENTION_HOME" >&2; exit 1; }
   test -w "$(dirname "$RETENTION_TOOLS_PREFIX")" || { echo "tools 环境父目录不可写: $RETENTION_TOOLS_PREFIX" >&2; exit 1; }
   df -h "$RETENTION_HOME"
   export MINICONDA_PATH=/home/ma-user/work/dataset/Common_wl/miniconda3
   source "$MINICONDA_PATH/etc/profile.d/conda.sh"

先检查并修改 cluster.env。模板里的四个 prefix 是**待创建的新环境路径**，不是已经发现的服务器环境。
若已有 cluster.env，手动补上 RETENTION_TOOLS_PREFIX，并把 RETENTION_HOME 改到确认可写的目录；
不要用示例文件覆盖已有设置。先完成这些修改，再运行 load_settings。
复用现有环境时，分别激活、检查依赖，再将每次 echo "$CONDA_PREFIX" 的完整结果写入模板对应字段。
不要只填环境短名。三个运行环境不能混装。每次新终端重新加载这一节的变量并激活仿真环境。

2. 创建共享盘环境
-----------------

已有符合条件的环境可跳过创建命令。不要在 algorithm 里创建 .venv。
工具环境仅提供 uv 和 hf，不改变三个运行环境的依赖；它以及 Conda 包缓存、uv/pip 缓存
都放在 Common_wl 或 RETENTION_HOME，不写入用户默认的只读缓存目录。

.. code-block:: bash

   conda create -p "$RETENTION_SIM_PREFIX" python=3.11 pip -y
   conda create -p "$RETENTION_OPENPI_PREFIX" python=3.11 pip -y
   conda create -p "$RETENTION_QWEN_PREFIX" python=3.12 pip -y
   conda create -p "$RETENTION_TOOLS_PREFIX" python=3.12 pip -y
   "$RETENTION_TOOLS_PREFIX/bin/python" -m pip install uv huggingface_hub
   export PATH="$RETENTION_TOOLS_PREFIX/bin:$PATH"
   conda activate "$RETENTION_SIM_PREFIX"
   echo "$CONDA_PREFIX"
   uv pip install --python "$RETENTION_SIM_PREFIX/bin/python" "$RPENT_DIR[test,robocasa-pi05]" \
     -c "$RPENT_DIR/robots/robocasa/retention/runtime-constraints.txt" --torch-backend cu128
   uv pip check --python "$RETENTION_SIM_PREFIX/bin/python"

本地沿用已激活环境的 CUDA/NCCL。如果本地也需要显式使用共享 CUDA12.8，可在同一终端执行
configure_cuda /home/ma-user/work/dataset；它不会创建 MTP 软链。
运行入口会为 OpenPI、仿真和 Qwen 子进程分别激活对应 Conda prefix，
清理上一个环境的库路径，并选择当前环境的 cuDNN。

安装固定 OpenPI fork 到同一个算法包。保留其 .git 目录，MTP 的 doctor 需要核对 revision。
外部工作树的 .git 指针、指向另一算法包的源码软链都不能代替完整 checkout。
导出官方锁文件的第三方依赖，再安装到已有 Conda 环境；不要用 uv sync 在代码目录创建环境。

.. code-block:: bash

   export OPENPI_DIR="$RPENT_DIR/vendor/openpi"
   git clone https://github.com/robocasa-benchmark/openpi "$OPENPI_DIR"
   git -C "$OPENPI_DIR" checkout 5a6beda9ff99da30b4e1b59320f6a32971d7c397
   cd "$OPENPI_DIR"
   uv export --frozen --no-dev --no-emit-workspace --no-hashes \
     --output-file "$RETENTION_HOME/env/openpi-locked.txt"
   uv pip install --python "$RETENTION_OPENPI_PREFIX/bin/python" -r "$RETENTION_HOME/env/openpi-locked.txt"
   uv pip install --python "$RETENTION_OPENPI_PREFIX/bin/python" --no-deps \
     "$OPENPI_DIR" "$OPENPI_DIR/packages/openpi-client"
   uv pip freeze --python "$RETENTION_OPENPI_PREFIX/bin/python" | sed '/^-e /d' \
     > "$RETENTION_HOME/env/openpi-before-rpent.constraints.txt"
   uv pip install --python "$RETENTION_OPENPI_PREFIX/bin/python" "$RPENT_DIR[robocasa-pi05]" \
     -c "$RETENTION_HOME/env/openpi-before-rpent.constraints.txt"
   uv pip check --python "$RETENTION_OPENPI_PREFIX/bin/python"
   cd "$RPENT_DIR"
   bash "$CLUSTER/run_local.sh" python openpi -c "import jax; print(jax.__version__, jax.devices())"

运行时会将本次提交的 RPent、OpenPI 和 openpi-client 源码加入导入路径，避免共享环境的旧安装
把 MTP 导回本机源码。不要安装 LIBERO 的 OpenPI fork，也不要在 OpenPI 环境升级 Qwen 的依赖。
上游训练脚本把 JAX 缓存写死到 ~/.cache/jax；本适配器只重定向该缓存设置到用户输出盘，
保留固定上游训练代码和优化流程。

3. 下载资产、权重和一个任务的数据
---------------------------------

.. code-block:: bash

   "$RETENTION_SIM_PREFIX/bin/robocasa-download-assets" \
     --assets-path "$ROBOCASA_ASSETS_PATH" --no-macros --skip-existing -y
   hf download robocasa/robocasa365_checkpoints \
     --revision c484448aba1a9b60a04c9b0ca117241518ea69f3 \
     --include "pi05_pretrain_human300/multitask_learning/75000/**" \
     --local-dir "$RETENTION_HOME/checkpoints/robocasa365"
   bash "$CLUSTER/run_local.sh" init pilot1 pilot1
   bash "$CLUSTER/run_local.sh" cli pilot1 download-data
   bash "$CLUSTER/run_local.sh" cli pilot1 download-data --execute
   bash "$CLUSTER/run_local.sh" python openpi -c \
     "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

如果修改了 RETENTION_CHECKPOINT，下载的 local-dir 也应与该位置一致。权重必须包含 params 和
``assets/**/norm_stats.json``。只下载所选任务的 Target Human 示范，不下载 Pretraining 示范。
pilot1 只下载 OpenDrawer；已有完整目录会跳过，已有不完整目录会报错，避免覆盖。
先检查该任务具体目录，再将残缺数据移到备份位置后重试。数据检查核对元数据和视频/轨迹数量，
不代表完成全量解码或校验和验证。保存资产的 attribution 文件。

生成的配置在 $RETENTION_HOME/configs/local/pilot1.json，
输出在 $RETENTION_HOME/retention_outputs/local/pilot1。
init 不覆盖已有配置；之后反复执行训练命令时不要重新 init 或改变实验名。
openpi_python 和 simulator_python 指向输出盘上的小型启动包装器，包装器为子进程激活正确环境。
请保留这些文件，并通过 run_local.sh / run_mtp.sh 启动，不直接绕过它们运行旧 JSON。

4. 检查真实运行环境
-------------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 doctor --stage all --runtime
   export PI05_CHECKPOINT_PATH="$RETENTION_CHECKPOINT"
   export PI05_PYTHON="$RETENTION_HOME/runtime/local/pilot1/openpi-python"
   RPENT_RUN_PI05_INTEGRATION=1 bash "$CLUSTER/run_local.sh" python simulator -m pytest \
     "$RPENT_DIR/tests/e2e_tests/robocasa/test_pi05_retention.py" -v --timeout=1200 \
     -o cache_dir="$RETENTION_HOME/cache/pytest" --basetemp="$RETENTION_HOME/tmp/pi05-test"

doctor --runtime 会在配置指定的环境中检查 GPU、导入路径和资源，报告在配置旁。
不加 --runtime 只检查文件；doctor 不加载完整模型，也不执行任务。
真实组件测试加载 π0.5 并执行有限动作，使用固定离线 planner；它不证明 Qwen 图像工具链或任务成功率。
A800 80GB 是硬件起点，不能据此保证每种 batch/服务组合都不会 OOM。

5. 单任务和三任务训练
---------------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 plan
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct --checkpoint base
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
   bash "$CLUSTER/run_local.sh" cli pilot1 summarize
   bash "$CLUSTER/run_local.sh" init pilot3-joint pilot3
   bash "$CLUSTER/run_local.sh" cli pilot3 download-data --execute
   bash "$CLUSTER/run_local.sh" train pilot3-joint pilot3

train PRESET NAME 在配置不存在时创建；已有配置时直接使用它，不覆盖你修改的参数，
依次 doctor、plan、train --all --resume。PRESET 只控制首次创建。
三任务独立训练用 pilot3-independent；每次都从基础 Human300 初始化。
pilot 默认为1,000步、batch16、每100步保存；checkpoint 目录使用从0开始的步号，
最终是999。可以在首次 plan 前调整 save_interval，正式长训练可改为1000以减少保存开销。
全参数 checkpoint 很大，应预留空间并测量共享盘保存速度。

6. 调试中断后继续
-----------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 status
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
   bash "$CLUSTER/run_local.sh" cli pilot1 summarize

* 训练未完成：从 Orbax 最近可恢复的完整 checkpoint 恢复参数、优化器、step 和相关训练状态。
  尚未保存或正在写入而未完成的进度需要重做；首次 checkpoint 之前的故障仍会从基础权重开始。
* 训练已完成：验证 checkpoint 记录与归一化后跳过，不再次启动优化器。
* 评测未完成：保留已完成回合，只补缺失项。基础设施错误不冒充任务失败。
  崩溃遗留的 .running 标记会阻止覆盖；确认旧平台作业/进程已经结束后，才清理对应标记，
  不删除整段评测目录。
* status 只检查目录和记录，saved_step_candidates 不是成功恢复的证明。真正恢复由 Orbax 校验。
  本版本上游没有保存数据加载器游标，不能承诺采样顺序或位级结果与连续运行一致。
* 训练和计划刷新使用操作系统文件锁，进程退出后释放；不要删除锁文件来绕开仍在运行的任务。
  共享文件系统必须支持这些锁。独立50模式也会跳过此前完成的模型。

修改了 Harness、planner 或评测设置后，原 protocol_id 会变化。停止原任务和 planner，
先查看 status；training_matches=true 时可执行：

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 refresh-plan
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct

refresh-plan 把旧 evaluation、summary、services 和计划保存到 history/<旧协议ID>，
保留 training、checkpoints 与训练状态，重新生成评测计划。旧结果不会与新评测混算。
刷新中途异常时，用同一代码和配置再次运行 refresh-plan 完成归档；不要手动删除事务文件。

training_id 核对训练任务/模式、方法、种子、基础权重声明、数据目录、步数、学习率、全局 batch、
num_workers、FSDP、示范比例、固定任务目录和训练适配代码。改变这些内容需要新实验，
不能把不兼容优化器状态当作原实验继续。保存间隔、设备分配、评测设置与启动器修改不属于训练语义；
环境版本和数据内容仍须自行冻结，目录名一致不证明内容一致。
旧版本没有 training_id 的计划不能自动认证迁移。
单卡到八卡改变了 FSDP，不属于保证可接续的同一训练配置；按阶段新建实验。
本地/MTP 配置也分开保存，不能直接把本地绝对源码路径当成 MTP 配置。

7. 安装与比较两个 Qwen
----------------------

先创建包含两种 planner 的实验，也可以在原 pilot1 的 JSON 中加入相应 modes 后，
按第6节 refresh-plan 保留训练并重做评测。不要为切换 planner 修改训练任务。

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" init pilot1 pilot1-agent --with-planners
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1-agent
   export VLLM_RELEASE=latest
   export VLLM_WHEEL_URL="$("$RETENTION_TOOLS_PREFIX/bin/python" - <<'PY'
   import json, os, platform, urllib.request
   release = os.environ["VLLM_RELEASE"]
   suffix = "latest" if release == "latest" else "tags/" + release
   url = "https://api.github.com/repos/vllm-project/vllm/releases/" + suffix
   with urllib.request.urlopen(url) as response:
       metadata = json.load(response)
   matches = [a["browser_download_url"] for a in metadata["assets"]
              if "+cu128-" in a["name"] and a["name"].endswith(platform.machine() + ".whl")]
   if len(matches) != 1:
       raise SystemExit("Choose a release with one compatible cu128 wheel; do not substitute a different CUDA build.")
   print(matches[0])
   PY
   )"
   test -n "$VLLM_WHEEL_URL"
   printf '%s\n' "$VLLM_WHEEL_URL" > "$RETENTION_HOME/env/vllm-wheel-url.txt"
   uv pip install --python "$RETENTION_QWEN_PREFIX/bin/python" "$VLLM_WHEEL_URL" --torch-backend cu128
   uv pip install --python "$RETENTION_QWEN_PREFIX/bin/python" "$RPENT_DIR"
   uv pip check --python "$RETENTION_QWEN_PREFIX/bin/python"
   hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
   hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
   bash "$CLUSTER/run_local.sh" python qwen -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   bash "$CLUSTER/run_local.sh" evaluate pilot1-agent direct
   bash "$CLUSTER/run_local.sh" evaluate pilot1-agent qwen3_vl_4b
   bash "$CLUSTER/run_local.sh" evaluate pilot1-agent qwen35_4b
   bash "$CLUSTER/run_local.sh" cli pilot1-agent summarize

这里按 vLLM 官方 release 资产选择 CUDA12.8 wheel 并保存准确 URL；若当前 release 没有该构建，
选择官方提供 cu128 且支持两种 Qwen 的版本，把 VLLM_RELEASE 设成其 v 开头的 tag 后重试。
检查 glibc、驱动和 wheel 的实际要求，不能只升级 Torch 来修复不匹配的 vLLM 二进制。
正式实验沿用 pilot 验证过的 URL 和环境，不反复安装 latest。

evaluate 入口在同一作业内启动当前 Qwen 服务、等待模型列表就绪、运行 worker 并清理自己启动的进程组。
默认单卡 GPU0 共用，Qwen 显存比例0.4、并发1、上下文32768；必须实测峰值和真实图像/工具回合。
服务就绪不代表这些能力已经验证。失败日志位于输出目录的 batch_logs。
该入口会评测声明的任务范围；要先做一个真实回合，可用 planner NAME MODE 0 保持服务前台，
另一个同环境终端执行 cli NAME evaluate --checkpoint base --mode MODE --tasks OpenDrawer --max-cells 1。
完成后停止手动服务，再运行自动 evaluate，避免占用同一端口。

8. 本机准备全量数据，提交 MTP 训练
----------------------------------

先在本机创建仅用于准备资源的 joint50 配置并下载50个 Target 任务。不要在本机运行 run_mtp.sh。

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" init joint50 joint50-data
   bash "$CLUSTER/run_local.sh" cli joint50-data download-data --execute
   git -C "$OPENPI_DIR" rev-parse HEAD
   test -d "$OPENPI_DIR/.git"
   test -f "$CLUSTER/cluster.env"

向平台提交 hyy_vla 整个算法包，包含 rpent、cluster.env、vendor/openpi 及 OpenPI .git。
忽略规则只影响 Git，不保证平台会包含文件；检查平台实际快照，不能漏掉这两项被 Git 忽略的资源。
不打包本地环境、缓存、数据或结果。平台数据/模型挂载应对应本机共享盘；
镜像须具备所需系统库和兼容驱动。设置**1节点、8 GPU**，启动命令：

.. code-block:: bash

   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh train joint50 joint50

入口读取 MA_NUM_HOSTS、MA_NUM_GPUS、VC_TASK_INDEX、VC_WORKER_HOSTS；缺失参数或多节点直接报错，
不会默认偷偷使用8卡。它依次建立软链、配置 distutils/CUDA12.8/NCCL/超时、初始化 Conda、
按本机同一绝对 prefix 激活环境，再恢复 CUDA/cuDNN 路径。NCCL_P2P_DISABLE 不默认开启。
这里没有多机 JAX 初始化，设置 DeepSpeed 超时变量也不意味着使用 DeepSpeed。

首次创建 configs/mtp/joint50.json，fsdp_devices 取平台 GPU 数；单进程 JAX 使用8卡协同训练一个模型，
batch16 是全局 batch。数据必须已经下载，脚本不会在训练阶段重新装环境。
重提同一个命令、同一个实验名会执行 --resume。源码有修改必须重新提交快照；
如协议因此变化，先提交一次 cli joint50 status / cli joint50 refresh-plan，
再重提训练。改变训练语义则换实验名。不要同时提交两个同名训练作业。

最后阶段把 preset 与实验名换为 independent50，50次训练按顺序执行，每个模型使用所配置的8卡，
不是自动并行训练50个模型。已完成的模型在续跑时跳过。

9. MTP 分片评测与汇总
---------------------

训练结束后，分别提交下面的评测命令，每个都是一个完整平台作业。Qwen 与 worker 必须处于同一作业，
不能把使用 localhost 的 planner 和评测器分别提交到不同容器。

.. code-block:: bash

   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 direct
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 qwen3_vl_4b
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 qwen35_4b
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh cli joint50 summarize

direct 使用8个 env+π0.5 worker；Qwen 模式使用GPU0–6的7个 worker，并把当前 Qwen 放在GPU7。
两个 Qwen 模式顺序运行。自动生成的 worker_configs 只改变设备字段，不改变任务分母。
每个模式都包含基础权重及适应后模型，故训练后运行也能得到 baseline。
若要训练前先评测基础权重，先提交 init joint50 joint50，再提交 evaluate joint50 MODE --checkpoint base，
之后提交 train；各步骤沿用同一配置。默认联合50计划为21,000回合，独立50为535,500回合。

失败后重提相同命令会跳过完整回合；停止其它 worker 后检查残留 .running。
正常任务失败不能删掉后重试到成功。汇总返回2表示仍有声明范围内的配对缺失，
complete=true 的 pilot 也不等于 full_benchmark_complete=true。
summary 下有 summary.json、comparisons.csv、per_task.csv。

10. 保存版本与排错
------------------

.. code-block:: bash

   uv pip freeze --python "$RETENTION_SIM_PREFIX/bin/python" > "$RETENTION_HOME/env/simulator.txt"
   uv pip freeze --python "$RETENTION_OPENPI_PREFIX/bin/python" > "$RETENTION_HOME/env/openpi.txt"
   uv pip freeze --python "$RETENTION_QWEN_PREFIX/bin/python" > "$RETENTION_HOME/env/qwen.txt"
   git -C "$OPENPI_DIR" rev-parse HEAD > "$RETENTION_HOME/env/openpi-revision.txt"
   git -C "$RPENT_DIR" rev-parse HEAD > "$RETENTION_HOME/env/rpent-head.txt"
   git -C "$RPENT_DIR" status --short > "$RETENTION_HOME/env/rpent-working-tree.txt"

同时保留完整提交快照；只有 HEAD 无法表示未提交代码和被忽略的 cluster.env。
doctor 报路径错误时先确认包根和挂载；CUDA=false 时确认实际 prefix、驱动和库版本。
OOM 时先停止其它服务；改变训练 batch/FSDP/方法属于新实验，不能删除计划绕过校验。
数据损坏不使用 --all 重下所有旧数据。源码修复后用 status 判断能否 refresh-plan，
保留原检查点，不通过删除 checkpoints 重试。

开发主机仅能验证离线逻辑、脚本和文档；本页不声称已完成真实 A800/MTP 安装、训练或任务成功率测试。
官方依据：`OpenPI 固定源码 <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_、
`uv export <https://docs.astral.sh/uv/reference/cli/#uv-export>`_、
`vLLM CUDA wheel 安装 <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>`_、
`RoboCasa 数据 <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_。
