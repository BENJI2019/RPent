π0.5 adaptation and retention of old tasks
===========================================

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

Deployment, execution and recovery
-----------------------------------

Use :doc:`robocasa_retention_setup` as the single installation and command guide for
local single-GPU debugging and single-host eight-GPU MTP jobs. It uses three shared
Conda environments: simulator/Harness, pinned official OpenPI, and Qwen/vLLM.
The submitted package must contain RPent and vendor/openpi. Scripts locate code
through SCRIPT_DIR and activate the absolute prefixes validated locally.
Artifacts use model/xiaoyi_tmpstorage/hyy_files/rpent/retention_outputs.
Do not put environments, caches or experiment results under algorithm.

Download selected Target Human LeRobot data only. datasets_root contains
v1.0/target. Old300 evaluation needs simulator code and assets, not old demonstrations.
Normalization is fixed to the Human300 checkpoint ``assets/**/norm_stats.json``.
Missing or ambiguous normalization fails instead of reading old data to estimate it.

Create a configuration with run_local.sh init, then run doctor, plan, baseline,
train --all --resume, adapted evaluation and summarize. Manually submit run_mtp.sh
through the platform; do not run it on the local host. Eight-GPU training uses
one JAX/FSDP process, not eight duplicate trainers. For each Qwen mode, planner and
evaluation workers run in the same job. Locally, run modes sequentially on GPU0.

Single-task run IDs look like full/OpenDrawer/seed_0; joint50 uses
full/joint_target50/seed_0. Joint subset IDs include a task-set digest in plan.json.
Methods are full and lora. Default demo_fraction=1.0 selects500_demos.
Keep the declared training IDs and task scope fixed during a run.

Checkpoint recovery
~~~~~~~~~~~~~~~~~~~~

save_interval controls intermediate saves. Repeating train --all --resume on the
same configuration restores the latest complete optimizer state and skips models
with valid completion records. This upstream version does not save the data-loader
cursor, so resumed sampling is not guaranteed to match uninterrupted training.
status reports saved directories and plan compatibility; Orbax validates restoration.

After Harness or evaluation-only changes, stop related processes and run
refresh-plan. It requires an unchanged training_id, retains training, and archives
previous evaluation, summaries, service records and plan under history/<old-ID>.
New and old evaluation results are not mixed. Changes to training tasks, learning
rate, batch, FSDP or training-adapter source require a new experiment. Legacy plans
without training_id cannot be certified automatically. See the setup guide for
recovery commands and incomplete-checkpoint behavior.

Use --tasks / --max-cells to run a small execution subset before completing the
declared grid; these flags do not change denominators. Completed cells are skipped;
infrastructure errors remain missing. Clear stale .running markers only after the
original process stopped. Do not delete ordinary task failures to retry until success.

summary/comparisons.csv reports model/mode/group changes, per_task.csv reports
task results, and summary.json records scope and missing pairs. Group scores with
missing pairs remain null and summarize exits2; infrastructure execution failures
exit1. Never feed held-out old-task results into training or agent memory.

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
