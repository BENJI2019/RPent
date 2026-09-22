Single-GPU debugging and eight-GPU MTP jobs
===========================================

This walkthrough follows the repository workspace-paths rules and local-vs-mtp
skill. See :doc:`robocasa_retention` for the scientific protocol. Run commands
in Linux Bash. Validate on one A800 80GB first, then manually submit a single-host,
eight-A800 job through the platform. These scripts do not submit platform jobs.

Storage and execution boundaries
--------------------------------

Submit **hyy_vla** as the algorithm package, containing this checkout at **rpent/**.
If the platform instead selects rpent itself as the package root, omit rpent from
the platform entry path. Scripts locate their checkout through SCRIPT_DIR.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Purpose
     - Local
     - MTP
   * - RPent code
     - /home/ma-user/work/algorithm/hyy_vla/rpent
     - /opt/huawei/schedule-train/algorithm/rpent
   * - OpenPI code
     - vendor/openpi inside RPent
     - vendor/openpi in the same submitted package
   * - Data / shared environments
     - /home/ma-user/work/dataset
     - /opt/huawei/dataset
   * - Models / user outputs
     - /home/ma-user/work/model
     - /opt/huawei/quoteModel

Keep configs, runtime wrappers, cache, tmp, env records, checkpoints and
retention_outputs under **model/xiaoyi_tmpstorage/hyy_files/rpent/**.
Assets and Target demonstrations use dataset/hyy_vla/robocasa365; Conda environments
use dataset/Common_wl/envs. The algorithm package contains code and configuration
templates. Only the MTP entry creates mount aliases after validating platform
metadata; it refuses conflicting paths. The local entry never creates those aliases.

1. Inspect the local host and set paths
---------------------------------------

Synchronize the complete source containing this feature. Inspect Linux, the driver
and shared volumes. Required system components include git, git-lfs, build tools,
ffmpeg and EGL/OpenGL. Install missing components for the actual distribution;
Ubuntu is not assumed.

.. code-block:: bash

   cd /home/ma-user/work/algorithm/hyy_vla/rpent
   cat /etc/os-release
   nvidia-smi
   df -h /home/ma-user/work/dataset /home/ma-user/work/model
   export RPENT_DIR="$PWD"
   export CLUSTER="$RPENT_DIR/robots/robocasa/retention/cluster"
   test -f "$CLUSTER/cluster.env" || cp "$CLUSTER/cluster.env.example" "$CLUSTER/cluster.env"
   export RETENTION_CODE_ROOT="$RPENT_DIR"
   source "$CLUSTER/common.sh"
   load_settings
   export MINICONDA_PATH=/home/ma-user/work/dataset/Common_wl/miniconda3
   source "$MINICONDA_PATH/etc/profile.d/conda.sh"

Inspect cluster.env first. Its three prefixes are proposed **new environment paths**,
not detected installations. To reuse an existing environment, activate and validate
it locally, then paste the exact output of echo "$CONDA_PREFIX" into the matching
field. Do not use short names or combine the three dependency stacks.
In each new terminal, reload these settings and activate the simulator environment.

2. Create environments on shared storage
----------------------------------------

Skip creation for existing validated environments. Do not create .venv inside
algorithm. The tools environment supplies uv and hf without modifying runtime
dependencies.

.. code-block:: bash

   conda create -p "$RETENTION_SIM_PREFIX" python=3.11 pip -y
   conda create -p "$RETENTION_OPENPI_PREFIX" python=3.11 pip -y
   conda create -p "$RETENTION_QWEN_PREFIX" python=3.12 pip -y
   export TOOLS_PREFIX="$RETENTION_HOME/envs/tools"
   conda create -p "$TOOLS_PREFIX" python=3.12 pip -y
   "$TOOLS_PREFIX/bin/python" -m pip install uv huggingface_hub
   export PATH="$TOOLS_PREFIX/bin:$PATH"
   conda activate "$RETENTION_SIM_PREFIX"
   echo "$CONDA_PREFIX"
   uv pip install --python "$RETENTION_SIM_PREFIX/bin/python" "$RPENT_DIR[test,robocasa-pi05]" \
     -c "$RPENT_DIR/robots/robocasa/retention/runtime-constraints.txt" --torch-backend cu128
   uv pip check --python "$RETENTION_SIM_PREFIX/bin/python"

Local execution retains the activated environment's CUDA/NCCL setup. To select the
shared CUDA12.8 stack locally, run configure_cuda /home/ma-user/work/dataset in the
same terminal; it does not create MTP symlinks. Each OpenPI, simulator and Qwen
child activates its own Conda prefix, removes the previous role's library paths
and selects cuDNN from the active environment.

Keep the pinned OpenPI checkout inside the submitted package, including its .git
directory for revision checks. An external worktree .git pointer or a symlink to
another algorithm package is insufficient. Export third-party dependencies from
the official lockfile into the existing Conda environment instead of creating an
environment in the code directory with uv sync.

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

Runtime imports select RPent, OpenPI and openpi-client from this submitted package,
avoiding stale source locations in shared installations. Do not install the LIBERO
OpenPI fork or Qwen dependencies into the OpenPI environment. The pinned trainer
hardcodes ~/.cache/jax; this adapter redirects that cache setting to shared user
storage while retaining the upstream optimizer and source checkout.

3. Download assets, weights and one task's data
-----------------------------------------------

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

If RETENTION_CHECKPOINT is overridden, adjust the download local-dir accordingly.
The checkpoint must include params and ``assets/**/norm_stats.json``. Download only
selected Target Human demonstrations, never pretraining demonstrations. pilot1
selects OpenDrawer. Complete datasets are skipped; partial directories cause an
error rather than being overwritten. Inspect and move the affected partial task
directory aside before retrying. Checks count metadata, trajectory and video
files; they do not perform full decoding or checksum verification. Preserve
asset attribution files.

The generated configuration is $RETENTION_HOME/configs/local/pilot1.json;
outputs use $RETENTION_HOME/retention_outputs/local/pilot1.
init never overwrites. To retry training, keep the same experiment name and do
not rerun init. openpi_python and simulator_python reference small wrappers on
the output volume that activate the correct child environments. Preserve these
files and use the local/MTP entry instead of bypassing it with an old JSON.

4. Check the actual runtime
---------------------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 doctor --stage all --runtime
   export PI05_CHECKPOINT_PATH="$RETENTION_CHECKPOINT"
   export PI05_PYTHON="$RETENTION_HOME/runtime/local/pilot1/openpi-python"
   RPENT_RUN_PI05_INTEGRATION=1 bash "$CLUSTER/run_local.sh" python simulator -m pytest \
     "$RPENT_DIR/tests/e2e_tests/robocasa/test_pi05_retention.py" -v --timeout=1200 \
     -o cache_dir="$RETENTION_HOME/cache/pytest" --basetemp="$RETENTION_HOME/tmp/pi05-test"

doctor --runtime probes GPU visibility, imports and resources in the configured
environments, writing its report beside the configuration. Without --runtime it
only checks files. It does not load the full model or execute a task. The real
component test loads π0.5 and executes bounded actions with a scripted planner;
it does not establish Qwen vision/tool behavior or benchmark success.
An A800 80GB is a starting point, not an OOM guarantee for every batch/service mix.

5. Train one or three tasks
---------------------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 plan
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct --checkpoint base
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
   bash "$CLUSTER/run_local.sh" cli pilot1 summarize
   bash "$CLUSTER/run_local.sh" init pilot3-joint pilot3
   bash "$CLUSTER/run_local.sh" cli pilot3 download-data --execute
   bash "$CLUSTER/run_local.sh" train pilot3-joint pilot3

train PRESET NAME initializes only when the configuration is absent; otherwise it
uses the existing settings unchanged, then runs doctor, plan and train --all --resume.
PRESET controls initial creation only. Use pilot3-independent for three separate
models, each initialized from Human300. Pilots use 1,000 steps, global batch16 and
save_interval=100. Checkpoint directory labels are zero-based; the final label is
999. Set save_interval before plan; longer runs can use 1000 to reduce I/O.
Full checkpoints are large, so measure shared-storage throughput and reserve space.

6. Continue after an interruption
---------------------------------

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 status
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct
   bash "$CLUSTER/run_local.sh" cli pilot1 summarize

* Interrupted training restores parameters, optimizer, step and related state from
  the latest complete checkpoint that Orbax can restore. Unsaved or unfinished
  asynchronous writes are lost; a failure before the first checkpoint still starts
  from the base model.
* Completed training validates its record and normalization, then skips optimization.
* Evaluation retains completed cells and runs missing cells only. Infrastructure
  errors remain missing. Clear a stale .running marker only after confirming that
  its original process/platform job stopped; do not remove the evaluation directory.
* status inspects directories and records. saved_step_candidates does not certify
  restoration; Orbax performs that validation. This upstream version does not save
  the data-loader cursor, so resumed sampling and bitwise results are not guaranteed
  to match uninterrupted execution.
* OS locks prevent concurrent training owners and plan refreshes during execution.
  Locks release on process exit; do not delete lock files to bypass a live job.
  The shared filesystem must support locking. Independent50 also skips finished models.

Harness, planner or evaluation changes alter protocol_id. Stop old jobs and
planners, inspect status, and proceed only when training_matches=true:

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" cli pilot1 refresh-plan
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1
   bash "$CLUSTER/run_local.sh" evaluate pilot1 direct

refresh-plan archives evaluation, summary, services and the old plan under
history/<old-protocol-ID>, preserving training, checkpoints and optimizer state.
Old evaluation results are not mixed with the new protocol. If archiving is
interrupted, rerun refresh-plan with the same code/configuration to finish it;
do not delete its transaction journal.

training_id checks training tasks/mode, method, seeds, base checkpoint declaration,
dataset location, steps, learning rate, global batch, num_workers, FSDP, demo fraction,
catalog and training-adapter source. Changes require a new experiment.
Save frequency, device placement, evaluation settings and launchers are not training
semantics. Freeze environment versions and data contents separately: matching paths
do not prove identical contents. Legacy plans without training_id cannot be certified
automatically. Changing FSDP from one to eight GPUs is a new training configuration.
Local/MTP configurations are separate; local absolute source paths are not MTP paths.

7. Install and compare the two Qwen planners
--------------------------------------------

Create an experiment with both planners, or add their modes to the original pilot1
JSON and use section 6's refresh-plan to preserve training while rerunning evaluation.
Switching planners does not require changing training tasks.

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" init pilot1 pilot1-agent --with-planners
   bash "$CLUSTER/run_local.sh" train pilot1 pilot1-agent
   export VLLM_RELEASE=latest
   export VLLM_WHEEL_URL="$("$TOOLS_PREFIX/bin/python" - <<'PY'
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

The installer selects an official CUDA12.8 wheel asset and records its exact URL.
If the selected release lacks one, choose an official cu128 release supporting both
Qwen models and set VLLM_RELEASE to its v-prefixed tag. Check glibc, driver and wheel
requirements; upgrading Torch alone does not repair an incompatible vLLM binary.
Keep the pilot-validated wheel URL and environment for formal runs rather than
repeatedly installing latest.

evaluate starts the selected Qwen service in the same job, waits for its advertised
model, runs workers and cleans up owned process groups. One local GPU0 is shared;
defaults are 0.4 planner memory fraction, concurrency1 and context32768. Measure
peak usage and validate real vision/tool episodes. Server readiness alone does
not establish those capabilities. Logs are under batch_logs.
For one initial episode, keep planner NAME MODE 0 running in a foreground terminal,
then run cli NAME evaluate --checkpoint base --mode MODE --tasks OpenDrawer --max-cells 1
from another identically configured terminal. Stop that manual service before
using automatic evaluate to avoid a port conflict.

8. Prepare all Target data locally, then submit MTP training
------------------------------------------------------------

Create a local joint50 configuration for resource preparation and download its
50 Target tasks. Do not run run_mtp.sh on the local debugging host.

.. code-block:: bash

   bash "$CLUSTER/run_local.sh" init joint50 joint50-data
   bash "$CLUSTER/run_local.sh" cli joint50-data download-data --execute
   git -C "$OPENPI_DIR" rev-parse HEAD
   test -d "$OPENPI_DIR/.git"
   test -f "$CLUSTER/cluster.env"

Submit the whole hyy_vla package, including rpent, cluster.env, vendor/openpi and
the latter's .git directory. Git ignore rules do not guarantee snapshot inclusion:
verify that the platform includes the ignored settings and vendor checkout.
Exclude environments, caches, datasets and results. Mount the same shared data/model
volumes and use an image with the required system libraries and compatible driver.
Select **one host, eight GPUs** and this platform entry:

.. code-block:: bash

   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh train joint50 joint50

The entry reads MA_NUM_HOSTS, MA_NUM_GPUS, VC_TASK_INDEX and VC_WORKER_HOSTS.
Missing metadata or multiple hosts fail; GPU count never silently defaults to8.
It creates aliases, sets distutils/CUDA12.8/NCCL/timeouts, initializes Conda,
activates the exact locally validated prefix, and restores CUDA/cuDNN paths.
NCCL_P2P_DISABLE is not enabled by default. Multi-host JAX is not implemented;
the DeepSpeed timeout variable does not imply a DeepSpeed trainer.

The first submission creates configs/mtp/joint50.json with fsdp_devices from the
platform GPU count. One JAX process trains one model across eight GPUs; batch16 is
global. Data must already exist, and the job does not reinstall environments.
Resubmit the same command and experiment name to resume. Source changes require
a new platform snapshot. If its protocol changed, submit cli joint50 status and
cli joint50 refresh-plan before retrying training. Training-semantic changes
require a new name. Do not submit two live training jobs with the same name.

For the final stage use independent50 as preset and experiment name. Its 50 models
train sequentially, each using the configured eight GPUs. This does not schedule
50 parallel training jobs. Retries skip completed models.

9. Sharded MTP evaluation and summaries
---------------------------------------

After training, submit each evaluation command as its own complete platform job.
Qwen and workers must be inside the same job; localhost cannot connect a planner
and evaluator submitted into different containers.

.. code-block:: bash

   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 direct
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 qwen3_vl_4b
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh evaluate joint50 qwen35_4b
   bash /opt/huawei/schedule-train/algorithm/rpent/robots/robocasa/retention/cluster/run_mtp.sh cli joint50 summarize

Direct mode uses eight env+π0.5 workers. Qwen mode uses seven workers on GPU0–6
and the selected planner on GPU7. Run the two Qwen modes sequentially.
worker_configs changes placement only, preserving denominators. Each mode evaluates
both base and adapted checkpoints, so post-training evaluation still supplies a baseline.
For a pre-training baseline, first submit init joint50 joint50, then evaluate
joint50 MODE --checkpoint base, then train, using the same configuration.
Default joint50 requires21,000 episodes; independent50 requires535,500.

Retries skip complete cells. Inspect stale .running markers after stopping workers.
Never delete ordinary failed task episodes to retry until success. Summary exit2
means declared pairs are missing. Pilot complete=true does not imply
full_benchmark_complete=true. Outputs include summary.json, comparisons.csv and
per_task.csv under summary.

10. Record versions and troubleshoot
------------------------------------

.. code-block:: bash

   uv pip freeze --python "$RETENTION_SIM_PREFIX/bin/python" > "$RETENTION_HOME/env/simulator.txt"
   uv pip freeze --python "$RETENTION_OPENPI_PREFIX/bin/python" > "$RETENTION_HOME/env/openpi.txt"
   uv pip freeze --python "$RETENTION_QWEN_PREFIX/bin/python" > "$RETENTION_HOME/env/qwen.txt"
   git -C "$OPENPI_DIR" rev-parse HEAD > "$RETENTION_HOME/env/openpi-revision.txt"
   git -C "$RPENT_DIR" rev-parse HEAD > "$RETENTION_HOME/env/rpent-head.txt"
   git -C "$RPENT_DIR" status --short > "$RETENTION_HOME/env/rpent-working-tree.txt"

Preserve the complete submitted snapshot as well: HEAD alone omits uncommitted
code and ignored cluster.env. For path failures inspect the package root and mounts;
for CUDA=false check the actual prefix, driver and libraries. Stop competing
services before diagnosing OOM. Changing training batch/FSDP/method requires a new
experiment, never removal of plan checks. Do not download all old data to repair a
partial Target task. After code fixes use status to decide whether refresh-plan
can retain training; do not delete checkpoints to retry.

Development-host checks cover offline logic, scripts and documentation, not actual
A800/MTP installation, training or task success. References:
`pinned OpenPI <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_,
`uv export <https://docs.astral.sh/uv/reference/cli/#uv-export>`_,
`vLLM CUDA wheels <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>`_,
`RoboCasa data <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_.
