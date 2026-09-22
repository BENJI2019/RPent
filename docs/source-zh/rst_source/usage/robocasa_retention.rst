π0.5 的下游适应与旧任务能力保持
================================

首次在服务器部署，请从 :doc:`robocasa_retention_setup` 开始。该页面按单卡/8卡
A800 80GB 给出环境安装、按任务下载数据、训练前检查、训练与评测的完整命令。

研究结论与任务定义
------------------

这个设定适合研究：**不能使用旧训练样本时，单任务或多任务适应引起的旧任务能力变化，
以及 Harness agent 能否改善适应后的系统表现。** 起点为官方
π0.5 Human300 权重，不包含 MimicGen 联合预训练。这里的“无旧数据”限制针对
微调过程；评测者仍能访问旧任务仿真器。若连旧任务环境也无法访问，
这个实验就不能代表那种更严格的设定。

官方 Target50 与 Pretrain300 有 34 个任务名称重合：18 个 atomic_seen，
16 个 composite_seen；另外 16 个 composite_unseen 是新任务。
Target 使用不同的厨房场景。因此这同时包含已知任务的场景适应和新组合任务学习，
并非 50 个全新任务。代码固定了官方任务表及 horizon，保留
pretrain 和 target 两种场景域，不能把同名任务去重后混成一个评测集。

训练支持两种方式：``independent`` 为每个选定任务分别训练一个专用模型；
``joint`` 将选定任务的示范混合，训练一个模型。每种方式均按训练方法和训练 seed
分别产生 checkpoint，且每次都从同一 Human300 权重开始，只读取选定 Target Human
任务的示范。支持全参数 SFT 和 LoRA SFT；没有实现顺序持续学习或 RL 训练器，
也不把 LoRA 当作无遗忘保证。

报告区分三层能力：

* **动作策略能力**：direct 模式直接闭环执行 π0.5，是检测权重层面遗忘的主结果。
* **agent 系统能力**：相同 π0.5 接入 Qwen3-VL-4B-Instruct 或 Qwen3.5-4B，
  使用相同 Harness 工具、观测和预算，比较各自微调前后的变化。
* **更广泛的基础能力**：视觉理解、语言推理及 π0.5 其他来源的预训练能力
  没有被这 300 个机器人任务覆盖，不能从本实验推出它们都得到保留。

Harness 的脚本控制、导航和恢复行为可能补偿策略退化。因此 agent 成功率不下降，
不能单独证明 VLA 权重没有遗忘。本实现保持现有工具名称
``rldx_skill`` / ``rldx_arm``，但在 π0.5 模式下调用 π0.5 服务。
这些工具仍使用环境提供的完整任务指令；没有新增子任务语言训练。
每个回合使用独立空记忆，不加载公开历史 recipe；这是一组受控的 Harness
实验，不等同于原论文使用历史记忆的完整复现。

分阶段实验与配置
----------------

**50 次独立微调不是前置要求。** 联合50任务更直接回答“一个适应后的模型能否
学会下游任务并保持旧能力”；独立50次适合进一步研究哪些任务更容易引起遗忘。
建议按以下顺序推进，表中的模型数量均以一种训练方法、一个训练 seed 为单位：

.. list-table:: 阶段与目的
   :header-rows: 1
   :widths: 20 30 15 35

   * - 阶段
     - 训练方式
     - 模型数
     - 评测目的
   * - 小规模试验
     - 单任务；或选1–3任务联合/独立训练
     - 联合1；独立1–3
     - 验证下游学习及预先选定的旧任务探针
   * - 主实验
     - 50任务联合训练
     - 1
     - 完整 Target50 与 Pretrain300 前后对照
   * - 后续分析
     - 50任务独立训练
     - 50
     - 比较适应任务之间的差异与迁移矩阵

可以只微调一个任务、只测试该任务，这适合检查训练是否有效，**无法判断旧任务
是否遗忘**。早期保留率试验应同时评测少量旧任务，并与基础权重配对。
1–3任务试验只能提供局部证据，不能证明对完整旧300或所有下游任务都有效。
如果依据旧任务探针调整方法，它们就是开发集；正式报告应区分探针和未用于调参的
旧任务，不能把全部旧300描述成从未使用过的测试集。

配置中的三个字段互相独立：

* ``adaptation_mode``：``independent``（默认）或 ``joint``。
* ``adaptation_tasks``：选定的 Target 任务名称列表；``null`` 或省略表示全部50任务。
* ``evaluation_tasks``：按 ``pretrain`` / ``target`` 指定评测任务；``null`` 或省略
  表示完整旧300＋新50。显式字典中省略某个域表示不评测该域，两个域不能同时为空。

例如，在配置中使用以下字段可联合训练三个任务，并评测它们和三个旧任务探针：

.. code-block:: json

   {
     "adaptation_mode": "joint",
     "adaptation_tasks": ["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"],
     "evaluation_tasks": {
       "target": ["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"],
       "pretrain": ["CloseDrawer", "CloseCabinet", "TurnOffMicrowave"]
     }
   }

三个下游任务分别来自 atomic_seen、composite_seen、composite_unseen。
三个旧任务探针均不属于 Target50；它们只是便于启动的例子，不能代表旧任务的全部难度。
只测单个新任务时，使用 ``evaluation_tasks: {"target": ["OpenDrawer"]}``。
只训练一个任务却评测完整350个任务也支持：保留 ``evaluation_tasks: null``。

仓库提供可复制的完整配置：

* :download:`pilot1.json <../../../../robots/robocasa/retention/examples/pilot1.json>`：
  单任务、三个旧探针，1个模型，24个评测回合。
* :download:`pilot3-joint.json <../../../../robots/robocasa/retention/examples/pilot3-joint.json>`：
  三任务联合、三个旧探针，1个模型，36个评测回合。
* :download:`pilot3-independent.json <../../../../robots/robocasa/retention/examples/pilot3-independent.json>`：
  三任务独立，3个模型，72个评测回合。
* :download:`joint50.json <../../../../robots/robocasa/retention/examples/joint50.json>`：
  联合50任务，1个模型，完整评测21,000回合。
* :download:`example.json <../../../../robots/robocasa/retention/example.json>`：
  独立50任务，50个模型，完整评测535,500回合。

以上回合数包含共享 baseline。pilot 配置只用 direct、3个评测 seed、1,000个训练步，
用于小规模检查，不能据此宣称统计非劣；正式两个配置使用3个评测模式、10个评测 seed、
10,000个训练步。要在 pilot 对比两个 Qwen，应在首次运行前把 ``modes`` 设为
``["direct", "qwen3_vl_4b", "qwen35_4b"]`` 并启动相应服务。
每阶段使用独立 ``output_root``；后续阶段重新从 Human300 出发，不从 pilot 权重接着训练。

联合训练调用固定版本 OpenPI 的多数据集加载器，使用其默认的
``dataset_length ** 0.4`` 混合权重，不是各任务均匀采样；评测仍对任务等权。
``train_steps`` 是**每个训练运行的总步数**，不会自动乘任务数。
联合50任务跑10,000步与独立50次各跑10,000步的总训练量不同，论文中应明确选择
匹配总计算量还是匹配每任务训练量，并报告实际采样与预算。
``plan.json`` 记录每次及总训练步数、采样方式和评测规模。

指标与公平性
------------

每份微调权重在声明的任务集上评测，并与同模式下的原始权重配对；默认是完整
Target50 和 Pretrain300。输出包含被适应任务的 Target 成功率、全部 Target50 的迁移表现、
旧 300 任务、排除所有当前适应任务后的旧集合、排除全部 Target50 后的旧 266 任务、
重合的 34 个旧任务，以及 Target 的三个分组。不能把 50 个专用模型对角线的平均值
称为“一个通用模型的 Target50 成绩”。联合模式的 ``adapted_targets`` 则是同一模型
在训练任务集上的表现；联合50时，它和 ``target50`` 使用同一组任务。

显式选择小评测集时，统计仅覆盖声明的集合，相应组名带 ``_subset``，同时列出
参考任务数和实际评测任务数。例如只测3个旧任务时为 ``pretrain300_subset``，
不会标为完整旧300结果。``complete`` 表示声明的网格完成；只有完整350任务全部完成，
``full_benchmark_complete`` 才为真。``retention_evaluation_configured`` 标识是否配置了任何旧任务评测，
不表示能力已得到保持。未评测的分组不生成分数。

成功率对任务等权。令 Δ = 微调后成功率 − 微调前成功率；
forgetting = −Δ。报告同时保留逐任务结果与最差任务变化，避免均值掩盖退化。
默认容忍幅度 ε = 0.05，即 5 个百分点；应在看结果前确定。
代码给出单项比较的一侧 Hoeffding 下界，只有下界大于 −ε 才标记 noninferior。
这个保守界假设固定任务集中的配对回合相互独立，不包含训练随机性的总体推断，
也不提供多模型、多分组的同时保证。单纯“差异不显著”不足以证明能力保持；
旧任务初始成功率过低也会掩盖遗忘。

场景随机种子按场景域、任务、回合编号确定；前后共享种子、动作预算和
π0.5 初始采样噪声。统计要求初始 MuJoCo 物理状态指纹一致。
物理状态指纹不等于对全部纹理和渲染状态的证明；应固定软件与资产版本。
服务器限制一次 reset 和任务对应的总仿真步数，脚本动作也占用预算。
API 故障、进程超时、缺失结果保留为未完成，不能记成机器人失败或从分母删除。
正常主动结束但未成功则计为失败。

独立50模式默认 1 个训练种子、10 个评测回合、3 个模式，加上共享 baseline，
完整计划是 51 × 350 × 10 × 3 = **535,500 回合**。
同等条件下联合50模式为 2 × 350 × 10 × 3 = **21,000 回合**。
按上面的阶段逐步扩展，正式实验建议考虑至少三个训练种子，并固定超参数，
不要根据旧 300 任务评测反复选择学习率或 checkpoint；否则旧任务测试集变成了验证集。
当前代码固定训练步数并评测最终 checkpoint，保存恢复不等于按测试集挑选模型。

部署、运行与恢复
----------------

按 :doc:`robocasa_retention_setup` 配置本地单卡和 MTP 单机8卡环境。
该指南是安装、路径、数据下载及启动命令的统一入口，使用共享盘上的三个 Conda 环境：
仿真/Harness、固定官方 OpenPI、Qwen/vLLM。MTP 的同一算法包必须包含 RPent 与 vendor/openpi，
代码通过 SCRIPT_DIR 定位，环境使用本机验证过的绝对 prefix。
产物在 model/xiaoyi_tmpstorage/hyy_files/rpent/retention_outputs，
不在 algorithm 下创建环境、缓存或训练结果。

只下载所选 Target Human 的 LeRobot 数据。datasets_root 包含 v1.0/target；
旧300评测只需要仿真代码和资产，不需要旧示范。
归一化来自固定 Human300 checkpoint 的 ``assets/**/norm_stats.json``，
找不到或发现多个文件时会拒绝运行，不读取旧示范重新估计。

先用 run_local.sh init 创建配置，依次执行 doctor、plan、baseline、
train --all --resume、适应后评测和 summarize。MTP 通过 run_mtp.sh 在平台手动提交，
不能在本机运行该入口。八卡训练是一进程 JAX/FSDP，不是八个重复训练进程。
两个 Qwen 模式分别在同一作业内提供 planner 和评测 worker；本机可顺序共享GPU0。

单任务训练 ID 是 full/OpenDrawer/seed_0；联合50是 full/joint_target50/seed_0；
联合子集 ID 含任务集合摘要，可从 plan.json 读取。方法可选 full 和 lora。
demo_fraction 默认1.0，对应500_demos。训练 ID 与任务范围确定后不应随意更改。

断点恢复
~~~~~~~~

save_interval 控制中间状态保存。重跑同一配置的 train --all --resume
恢复最新完整 checkpoint 的优化器等状态，并跳过已有完成记录的模型。
上游没有保存数据加载器游标，恢复不保证与未中断的采样顺序逐步相同。
status 报告已有保存目录与计划兼容性；真正可恢复性仍由 Orbax 检验。

仅修改 Harness 或评测代码/配置时，可在停止相关进程后执行 refresh-plan：
训练身份 training_id 相同才允许保留训练，将旧评测、汇总、服务记录和计划归档到
history/<旧协议ID>，新旧评测互不混用。改变训练任务、学习率、batch、FSDP 等，
或修改训练适配代码，则需要新实验。旧版没有 training_id 的计划不能自动认证复用。
完整说明、恢复命令及不完整 checkpoint 的处理见部署指南。

评测可用 --tasks / --max-cells 先限制执行，再去掉限制补齐；
它们不改变声明分母。已完成回合会跳过，基础设施错误保持未完成。
遗留 .running 标记只能在确认原进程已停止后处理。
不要删除正常失败回合来反复测试到成功。

summary/comparisons.csv 提供每模型、模式与分组的变化；
per_task.csv 提供逐任务结果，summary.json 包含覆盖范围及缺失配对。
缺失配对的组分数为空，summarize 返回2；运行基础设施错误返回1。
保留旧测试结果用于最终统计，不把它们送回训练器或 agent 记忆。

研究边界与后续方法
------------------

此项目提供遗忘测量和 agent 补偿的基线，不承诺微调后的同一组权重绝不遗忘。
缺少旧分布信息时，有限下游样本上的约束无法保证未知旧分布上的表现。
若研究目标是“确保保持”，可以进一步设计冻结基础策略、独立 adapter 与路由的系统：
旧路径不变可以保留该路径本身，但路由错误仍会导致整个系统退化，必须单列 oracle
路由和实际 agent 路由。也可以在新的实验协议中比较参数正则或教师蒸馏，
明确它们使用的是 Target 观测、生成观测还是允许的旧环境交互；
不能把旧环境收集的训练回放称为完全没有旧分布数据。

资料与验证范围
--------------

* `RoboCasa 数据划分 <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_
* `官方数据下载与格式 <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_
* `Human300 多任务模型 <https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html>`_
* `固定版本 OpenPI fork <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_
* `π0.5 Human300 权重 <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3/pi05_pretrain_human300/multitask_learning/75000>`_
* `Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_
  与 `Qwen3.5-4B <https://huggingface.co/Qwen/Qwen3.5-4B>`_
* `vLLM Qwen3.5 服务说明 <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>`_

离线测试覆盖协议、分片、统计、配对检查、预算、模型数据边界与训练恢复。
真实 GPU 训练、检查点加载、仿真动作链和两个 Qwen 的工具调用需在上述环境验证；
没有将离线测试结果当作机器人任务成功率。
