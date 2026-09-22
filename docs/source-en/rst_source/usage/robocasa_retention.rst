π0.5 adaptation and retention of old tasks
==========================================

For first-time server deployment, start with :doc:`robocasa_retention_setup`.
It covers one/eight A800 80GB GPUs, installation, selected-task downloads,
preflight checks, training and evaluation commands.

Scope and interpretation
------------------------

This experiment studies **single-task or multitask adaptation without old training samples,
and whether a Harness agent can improve the resulting system's performance**.
Its starting point is the official π0.5 Human300 checkpoint, without MimicGen
joint pretraining. The adaptation process cannot read old demonstrations; the
evaluator can still access old-task simulators. This does not represent the
stricter setting where old environments are also unavailable.

Target50 shares 34 task names with Pretrain300: 18 atomic_seen and 16
composite_seen. The remaining 16 composite_unseen tasks are new compositions.
Target tasks use a different kitchen domain. Keep both pretrain and target
evaluations for shared task names: this measures both scene adaptation and
new-task learning. The checked-in catalog pins official task names and horizons.

Two training designs are available: ``independent`` trains one specialist for
each selected task; ``joint`` mixes the selected demonstrations to train one model.
Each method and training seed produces separate checkpoints, always starting from
the same Human300 weights and reading only selected Target Human demonstrations.
Supported methods are full SFT and LoRA SFT. There is no sequential continual
learning or RL trainer, and LoRA is not assumed to prevent forgetting.

Three interpretations must stay separate:

* **Policy retention:** direct closed-loop π0.5 execution measures changes in the
  action policy itself.
* **System retention:** Qwen3-VL-4B-Instruct and Qwen3.5-4B use the same Harness
  tools, observations and budgets; compare each mode before and after adaptation.
* **Broader foundation capabilities:** these 300 robot tasks do not evaluate all
  visual, language or other pretraining capabilities of π0.5.

Scripted control, navigation and recovery can compensate for policy degradation.
Stable agent success therefore cannot establish that the VLA weights retained
their abilities. Existing tool names ``rldx_skill`` / ``rldx_arm`` are preserved
and call π0.5 when that backend is selected. They still use the environment's
complete task instruction; no subtask-language training is added.
Each episode receives independent empty memory, without historical recipes.
This is a controlled Harness experiment, not a reproduction of the paper's
full historical-memory setting.

Staged experiments and configuration
-------------------------------------

**Fifty independent adaptations are not a prerequisite.** Joint50 directly tests
whether one adapted model can learn downstream tasks while retaining old abilities.
Independent50 can then investigate which adaptation tasks cause more forgetting.
Recommended stages, with model counts per training method and seed:

.. list-table:: Stages and objectives
   :header-rows: 1
   :widths: 20 30 15 35

   * - Stage
     - Training
     - Models
     - Evaluation objective
   * - Pilot
     - One task, or 1–3 tasks jointly/independently
     - Joint: 1; independent: 1–3
     - Downstream learning and predefined old-task probes
   * - Main experiment
     - Joint50
     - 1
     - Paired full Target50 and Pretrain300 evaluations
   * - Further analysis
     - Independent50
     - 50
     - Task-specific differences and transfer matrices

Training and evaluating only one downstream task is supported. It checks learning,
but **cannot measure forgetting on old tasks**. Include paired baseline/adapted
evaluations on old-task probes for pilot retention measurements. A 1–3 task pilot
provides local evidence, not validation over all old300 or downstream tasks.
Probes used to tune the method become development data: separate them from untouched
old tasks in final reporting rather than calling all old300 an unused test set.

Three configuration fields control distinct choices:

* ``adaptation_mode``: ``independent`` (default) or ``joint``.
* ``adaptation_tasks``: selected Target task names; ``null`` or omission selects all50.
* ``evaluation_tasks``: task lists by ``pretrain`` / ``target`` domain; ``null`` or
  omission selects old300 plus Target50. In an explicit dictionary an omitted
  domain selects no tasks; at least one task must be selected overall.

For example, these fields jointly train three tasks and evaluate them plus three
old-task probes:

.. code-block:: json

   {
     "adaptation_mode": "joint",
     "adaptation_tasks": ["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"],
     "evaluation_tasks": {
       "target": ["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"],
       "pretrain": ["CloseDrawer", "CloseCabinet", "TurnOffMicrowave"]
     }
   }

The downstream tasks respectively belong to atomic_seen, composite_seen and
composite_unseen. The three old probes are outside Target50. They are starter
examples, not a representative sample of all old-task difficulty.
For target-only evaluation use ``evaluation_tasks: {"target": ["OpenDrawer"]}``.
One-task training with full350 evaluation is also supported: keep
``evaluation_tasks: null``.

Complete example configurations are available:

* :download:`pilot1.json <../../../../robots/robocasa/retention/examples/pilot1.json>`:
  one training task, three old probes, one model, 24 evaluation episodes.
* :download:`pilot3-joint.json <../../../../robots/robocasa/retention/examples/pilot3-joint.json>`:
  three jointly trained tasks, three old probes, one model, 36 episodes.
* :download:`pilot3-independent.json <../../../../robots/robocasa/retention/examples/pilot3-independent.json>`:
  three independent adaptations, three models, 72 episodes.
* :download:`joint50.json <../../../../robots/robocasa/retention/examples/joint50.json>`:
  joint50 training, one model, 21,000 full-benchmark episodes.
* :download:`example.json <../../../../robots/robocasa/retention/example.json>`:
  independent50 training, 50 models, 535,500 full-benchmark episodes.

Episode counts include shared baselines. Pilots use direct execution only, three
evaluation seeds and 1,000 training steps for small checks, not statistical
noninferiority claims. The two full configurations use three evaluation modes,
ten evaluation seeds and 10,000 training steps. To compare both Qwen models in a
pilot, set ``modes`` to ``["direct", "qwen3_vl_4b", "qwen35_4b"]`` before the first
run and start the corresponding services. Each stage uses its own ``output_root``
and starts again from Human300, rather than continuing from pilot weights.

Joint training uses the pinned OpenPI mixture loader's default
``dataset_length ** 0.4`` weights, not uniform task sampling; evaluation remains
task weighted. ``train_steps`` is the **total number of steps per training run**,
with no automatic multiplication by task count. Joint50 at 10,000 steps and fifty
independent 10,000-step runs consume different total training budgets. State whether
comparisons match total compute or per-task exposure, and report actual sampling
and budgets. ``plan.json`` records per-run and total steps, sampling and evaluation
counts.

Metrics and controls
--------------------

Each adapted checkpoint is evaluated on its declared task set, defaulting to
Target50 and Pretrain300, against the same-mode base checkpoint. Reports include
adapted Target tasks, all Target50 tasks, old300, old tasks excluding all currently adapted tasks,
old266 excluding every Target50 task, the overlapping old34, and all three
Target groups. An average over specialist diagonal entries is not a generalist
model's Target50 score. In joint mode ``adapted_targets`` measures one model on
its training task set; for joint50 it uses the same tasks as ``target50``.

An explicit smaller evaluation set produces ``_subset`` group names and lists
both reference and evaluated task counts. Three old-task probes yield
``pretrain300_subset``, not a full old300 score. ``complete`` means the declared
grid is complete; ``full_benchmark_complete`` additionally requires the full350.
``retention_evaluation_configured`` indicates that any old-task evaluation is configured, not
that retention was demonstrated. Groups with no selected tasks produce no score.

Tasks have equal weight. Δ is adapted minus baseline success rate;
forgetting is −Δ. Per-task changes and the worst task change accompany averages.
The default noninferiority margin ε is 0.05 (five percentage points), to be
chosen before inspecting results. A one-sided Hoeffding bound marks a comparison
noninferior only when its lower bound exceeds −ε. This conservative bound assumes
independent paired episodes on a fixed task set. It does not include population
inference over training seeds or simultaneous coverage across comparisons.
A nonsignificant difference does not prove retention. Low baseline success can
also hide forgetting through a floor effect.

Scene seeds depend on domain, task and replicate. Before/after evaluations share
seeds, action budgets and initial policy sampling noise. Summaries require matching
initial MuJoCo physical-state fingerprints. These fingerprints do not certify
every texture or rendering detail: freeze simulator software and assets as well.
The server enforces one reset and the catalog's total simulator-step budget;
scripted actions count toward it. API failures, process timeouts and missing
records remain incomplete, never robot failures or omitted denominator entries.
Normal voluntary termination without success counts as failure.

For independent50, the default one training seed, ten replicates and three modes require
51 × 350 × 10 × 3 = **535,500 episodes**, including shared baselines.
Under the same settings joint50 requires 2 × 350 × 10 × 3 = **21,000 episodes**.
Expand through the stages above and consider at least three training seeds for
final experiments. Preregister hyperparameters.
Do not repeatedly select learning rates or checkpoints using old300 test results.
The implementation evaluates the final fixed-step checkpoint; resuming training
does not select a checkpoint using test performance.

Installation and data
---------------------

Execution requires Linux, CUDA GPUs, RoboCasa assets, Target Human demonstrations
and the Human300 checkpoint. The official OpenPI fork recommends at least 80 GB
GPU memory for training; actual requirements depend on batch size, full/LoRA and
parallelism. Adjust the example paths to your host. Use three separate environments,
starting from the RPent checkout:

.. code-block:: bash

   # 1. Simulator and Harness, without the RLDX policy backend
   uv venv .venv --python 3.11
   uv pip install --python .venv/bin/python -e ".[test,robocasa-pi05]" \
     -c robots/robocasa/retention/runtime-constraints.txt --torch-backend auto
   # Follow the ordinary RoboCasa guide for EGL/system dependencies and assets
   .venv/bin/robocasa-download-assets --help

   # 2. Official benchmark OpenPI; use absolute directory paths
   export RPENT_DIR="$PWD"
   export OPENPI_DIR=/opt/robocasa-openpi
   git clone https://github.com/robocasa-benchmark/openpi "$OPENPI_DIR"
   git -C "$OPENPI_DIR" checkout 5a6beda9ff99da30b4e1b59320f6a32971d7c397
   cd "$OPENPI_DIR"
   uv sync --python 3.11
   uv pip freeze --python .venv/bin/python | sed '/^-e /d' > openpi-constraints.txt
   uv pip install --python .venv/bin/python -e "$RPENT_DIR[robocasa-pi05]" \
     -c openpi-constraints.txt
   cd "$RPENT_DIR"

   # 3. Qwen serving; preserve resolved versions after the pilot
   uv venv .venv-qwen --python 3.12
   uv pip install --python .venv-qwen/bin/python -e . vllm --torch-backend auto
   mkdir -p logs
   uv pip freeze --python .venv-qwen/bin/python > logs/qwen-requirements.txt

Do not install RPent's LIBERO OpenPI fork or Qwen's newer Transformers into the
second environment. Training and evaluation verify its OpenPI source revision and
import location. The simulator dependencies of ``.[robocasa-pi05]`` track RPent
branches: save ``uv pip freeze`` output for all three environments, including
resolved source commits, after the pilot and keep them fixed for final runs.
The code checks OpenPI source and declared model revisions, not every byte of
large local weight files. Download the pinned revision below and keep weights
read-only.

.. code-block:: bash

   # Use an environment containing hf; download only this checkpoint subtree
   hf download robocasa/robocasa365_checkpoints \
     --revision c484448aba1a9b60a04c9b0ca117241518ea69f3 \
     --include "pi05_pretrain_human300/multitask_learning/75000/**" \
     --local-dir /data/checkpoints/robocasa365

   # Target Human only; do not use --all
   .venv/bin/python -m robocasa.scripts.download_datasets \
     --split target --source human

For direct upstream downloads, set ``DATASET_BASE_PATH`` through this RPent
branch's ``ROBOCASA_MACROS_PATH``. The new ``download-data --execute`` command
sets the configured path within its download process without editing installed
packages; see :doc:`robocasa_retention_setup`.
``datasets_root`` is the directory containing ``v1.0/target/...``.
Data is already in LeRobot format. Old300 evaluation needs simulator code and
assets, not old demonstrations. Normalization remains fixed to the checkpoint's
single ``assets/**/norm_stats.json``; missing or ambiguous assets raise errors
instead of recomputing statistics from old data.

Execution
---------

Copy a stage configuration above to ``logs/retention.json`` and update paths and
GPU ordinals. Start with ``pilot1.json``, then use ``joint50.json`` and a fresh
output directory for the main experiment.
``openpi_python`` points to environment 2; ``simulator_python`` to environment 1.
Training uses the caller's ``CUDA_VISIBLE_DEVICES``; evaluation uses configured
``env_cuda_device`` and ``vla_cuda_device``. Qwen services select GPUs in their
launch commands. Evaluate Qwen modes sequentially if memory cannot hold both.

.. code-block:: bash

   .venv/bin/python -m robots.robocasa.retention --config logs/retention.json plan

   # Separate terminals; only the planner being evaluated needs to run
   CUDA_VISIBLE_DEVICES=2 .venv-qwen/bin/python \
     -m robots.robocasa.retention.serve_planner \
     --config logs/retention.json --mode qwen3_vl_4b
   CUDA_VISIBLE_DEVICES=3 .venv-qwen/bin/python \
     -m robots.robocasa.retention.serve_planner \
     --config logs/retention.json --mode qwen35_4b

Services use pinned model revisions, greedy generation and disabled thinking.
Tool parsers are hermes and qwen3_coder respectively. Commands and resolved vLLM,
Torch and Transformers versions are recorded in ``services/*_launch.json``;
changed launch metadata requires a fresh output directory. External APIs can be
configured through ``planners``, but their operator must ensure stable model
versions and decoding settings. Before scaling up, use ``rpent-check-llm``
from :doc:`configure_planner` to check text connectivity. It does not test images
or tools; use bounded episodes with the real Qwen model to check both.

Run the real component and bounded policy-chain test first:

.. code-block:: bash

   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export PI05_CHECKPOINT_PATH=/data/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000
   export PI05_PYTHON=/opt/robocasa-openpi/.venv/bin/python
   RPENT_RUN_PI05_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0 \
     .venv/bin/python -m pytest \
     tests/e2e_tests/robocasa/test_pi05_retention.py -v --timeout=1200

This test uses the real model and simulator, then a deterministic offline planner
to verify Harness tool execution and cleanup. It does not require task success
and does not establish Qwen agent performance.

.. code-block:: bash

   # Base checkpoint; all means every mode enabled in the configuration
   .venv/bin/python -m robots.robocasa.retention --config logs/retention.json \
     evaluate --checkpoint base --mode all

   # Configured independent or joint training; resume skips completed checkpoints
   CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m robots.robocasa.retention \
     --config logs/retention.json train --all --resume

   # Reuses completed baseline cells, then evaluates all adapted models
   .venv/bin/python -m robots.robocasa.retention --config logs/retention.json \
     evaluate --checkpoint all --mode all
   .venv/bin/python -m robots.robocasa.retention --config logs/retention.json summarize

The independent one-task run ID is ``full/OpenDrawer/seed_0``; joint50 uses
``full/joint_target50/seed_0``. Joint subset IDs include a task-set hash and can be
read from ``plan.json``. Select a training job with ``train --run-id <ID> --resume``
and evaluate it with ``evaluate --checkpoint <ID> --mode direct``.
``methods: ["full", "lora"]`` adds LoRA adaptations and evaluations.
``demo_fraction`` must select a demonstration filter key supplied by the dataset;
the default 1.0 selects 500_demos.

Shard with ``--num-shards N --shard-index i``, assigning workers independent π0.5
GPUs. Run plan once before dispatching shards. Cells have exclusive ownership.
Completed results are skipped on retry. Remove a crash's stale ``.running`` lock
only after checking its original process has stopped. Protocol/code/path identity
changes refuse existing output. Changing training design, training tasks, evaluation
scope or modes also requires a fresh ``output_root``. Existing configurations may
omit the new fields to retain independent training and full evaluation defaults.
Training results are not imported across protocols. To evaluate one checkpoint
on a few tasks first and later complete the full benchmark, declare
``evaluation_tasks: null`` from the outset, run with ``--tasks`` / ``--max-cells``,
then remove those limits. These flags restrict execution without changing the
declared denominator.

``summary/comparisons.csv`` contains model/mode/group changes.
``per_task.csv`` contains tasks with all expected paired episodes.
``summary.json`` includes evaluation scope, missing-pair counts, examples and coverage. Incomplete groups have no
score and summarize exits with code 2. Infrastructure failures exit evaluation
with code 1 and can be retried after repair. Never feed held-out old-task results
back into training or agent memory.

Limits and further methods
--------------------------

This project implements forgetting measurement and agent-compensation baselines,
not a guarantee that adapted weights never forget. Constraints on finite target
samples cannot establish performance on an unknown old distribution.
A further system can freeze the base policy, add independent adapters and route
between them. An unchanged base path preserves that path, while routing mistakes
can still degrade the deployed system; report oracle and actual-agent routing
separately. Parameter regularization and teacher distillation are also possible
extensions, with explicit declarations of whether their observations come from
Target data, generation or permitted old-environment interaction. Old-environment
training replay must not be described as lacking all old-distribution data.

Sources and validation scope
----------------------------

* `RoboCasa dataset splits <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_
* `Dataset download and format <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_
* `Human300 multitask models <https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html>`_
* `Pinned OpenPI fork <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_
* `Human300 π0.5 checkpoint <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3/pi05_pretrain_human300/multitask_learning/75000>`_
* `Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_
  and `Qwen3.5-4B <https://huggingface.co/Qwen/Qwen3.5-4B>`_
* `vLLM Qwen3.5 serving <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>`_

Offline tests cover protocol identities, sharding, statistics, pairing, budgets,
model data boundaries and training resumption. Actual GPU training, checkpoint
loading, simulator actions and both Qwen models' tool calls require validation in
the environments above. Offline tests are not robot task-success measurements.
