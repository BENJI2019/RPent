Start to finish: π0.5 on A800 GPUs
==================================

This walkthrough targets **one A800 80GB for pilots** and **eight A800 80GB GPUs
on one machine for final experiments**. Start with direct execution, then add
the Qwen planners. See :doc:`robocasa_retention` for the scientific protocol.
Commands below use Linux Bash, not Windows PowerShell.

0. Check the server and source checkout
---------------------------------------

Transfer the current RPent source, including newly added files, to the server,
for example ``~/RPent``. A fresh upstream clone may not contain these unpublished
features. Run commands from the source root; ``robots`` is not a separate wheel.

.. code-block:: bash

   cd "$HOME/RPent"
   cat /etc/os-release
   nvidia-smi
   test -f robots/robocasa/retention/prepare.py
   df -h .

Check available GPUs and disk space for demonstration videos, caches and training
checkpoints. The official OpenPI fork recommends at least 80 GB for training; that
does not guarantee every batch/method fits. On Ubuntu/Debian, install missing
system packages and uv as follows; other distributions need equivalent packages:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git git-lfs curl build-essential cmake pkg-config \
     ffmpeg libegl1 libgl1 libgl1-mesa-dri libglib2.0-0
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   uv --version

The NVIDIA driver must support the pinned CUDA 12 JAX runtime. Verify actual GPU
visibility with ``doctor --runtime`` below. RLDX weights and RPent's LIBERO
OpenPI fork are not needed.

1. Set directories and environment variables
--------------------------------------------

Run these exports in the same shell, and save them to your own environment file
for future logins and additional terminals. Change ``EXP_HOME`` to an absolute
data-disk path if needed.

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

.. code-block:: text

   RPent/                          Current source checkout
   robocasa-retention/
     sim-env/                      Simulator, Harness, experiment driver
     openpi/.venv/                 Official OpenPI, JAX training and policy serving
     qwen-env/                     Optional vLLM planner serving
     assets/                       Scenes and objects
     datasets/v1.0/target/          Selected Target Human demonstrations only
     checkpoints/robocasa365/       Pinned Human300 base weights
     configs/                      Stage JSON files and doctor reports
     runs/pilot1/                  Training, checkpoints, evaluations, summaries
     env/                          Resolved software versions

2. Install the simulator and Harness environment
------------------------------------------------

.. code-block:: bash

   uv venv "$EXP_HOME/sim-env" --python 3.11
   uv pip install --python "$SIM_PY" -e "$RPENT_DIR[test,robocasa-pi05]" \
     -c "$RPENT_DIR/robots/robocasa/retention/runtime-constraints.txt" \
     --torch-backend auto
   uv pip check --python "$SIM_PY"
   "$SIM_PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   "$SIM_PY" -m robots.robocasa.retention --help

Torch must report CUDA availability and the expected visible device count.
The ``robocasa-pi05`` extra explicitly includes Torch, which simulator startup
requires. The installer selects a backend from the driver; preserve its resolved
versions. A CPU Torch installation is insufficient.

.. code-block:: bash

   "$EXP_HOME/sim-env/bin/robocasa-download-assets" \
     --assets-path "$ROBOCASA_ASSETS_PATH" --no-macros --skip-existing -y

Keep attribution files and download manifests. Rerun this command after an
interrupted asset download, then verify actual environment execution in step 6.
``--no-macros`` works with the new dataset command, which sets its own data root.

3. Install the separate official OpenPI environment
---------------------------------------------------

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

Preserve OpenPI's resolved dependencies, filtering editable entries that cannot
be constraints. Do not apply simulator constraints or upgrade Transformers for
Qwen inside this environment. Running ``uv sync`` again can remove the additional
RPent installation; repeat the whole section when rebuilding.

4. Download base weights and initialize a single-GPU pilot
----------------------------------------------------------

Install the HF CLI in a separate tool environment:

.. code-block:: bash

   uv tool install huggingface_hub
   hf download robocasa/robocasa365_checkpoints \
     --revision c484448aba1a9b60a04c9b0ca117241518ea69f3 \
     --include "pi05_pretrain_human300/multitask_learning/75000/**" \
     --local-dir "$EXP_HOME/checkpoints/robocasa365"

Use the JAX checkpoint containing ``params/`` and ``assets/**/norm_stats.json``,
not a different safetensors-only model. The repository is public. If your HF
configuration requires authentication, use ``hf auth login`` in your own terminal
and keep credentials out of experiment JSON files.

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     init --preset pilot1 --workspace "$EXP_HOME"

``init`` writes absolute resource paths, records the current simulator Python,
sets both env/VLA GPUs to 0 and refuses to replace a configuration. It neither
downloads nor trains. Inspect the JSON: pilot1 uses OpenDrawer, 1,000 steps,
batch 16, direct evaluation, three evaluation seeds and three old-task probes.
Change methods, batch or learning rate before plan. Repeating a preset as a new
experiment requires both a fresh configuration filename and ``output_root``.
Pass ``init --output-root /absolute/path/to/new-run`` to choose a new result directory.

To include both Qwen comparisons from the outset, add ``--with-planners`` and
complete step 9 before evaluation. Do not change ``modes`` after training:
doing so changes the protocol identity.

5. Download only configured Target Human data
---------------------------------------------

.. code-block:: bash

   # Preview task names and destinations; no download or protocol lock
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" download-data
   # Execute the displayed download plan
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" download-data --execute

pilot1 downloads only OpenDrawer Target Human demonstrations. Old probes need
the simulator, not demonstration downloads. The command reuses the installed
RoboCasa downloader and sets its data root within the child process, without
editing site-packages. It checks registry paths against the pinned catalog,
rejects incomplete existing destinations and detects upstream download failures
even when upstream returns normally. It downloads all 500 demonstrations for each
selected task; ``demo_fraction`` changes training selection, not download volume.
Pretraining Human/MimicGen data is excluded; do not use upstream ``--all``.

For an incomplete directory, inspect its exact path and disk space, preserve
needed files and move that directory aside before retrying. Do not delete the
entire datasets root. Checks cover basic metadata and whether parquet/three-camera
file counts match total_episodes; they do not checksum or decode every sample.

Prefetch the additional OpenPI PaliGemma tokenizer with the same account:

.. code-block:: bash

   "$OPENPI_PY" -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

This uses OpenPI's separate cache and requires access to its official storage on
first use. It is not included in the HF checkpoint subtree.

6. Check prerequisites and the real policy interface
----------------------------------------------------

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/pilot1.json" doctor --stage all --runtime

Every check should pass. The report is ``pilot1.doctor.json`` beside the config.
Without ``--runtime``, only files, paths and revision are checked; CUDA and Python
imports remain unverified. ``--stage train`` does not require simulator assets;
``--stage evaluate`` does not require demonstrations. Doctor does not restore the
full model, optimize, run simulator episodes or call planners.

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$EXP_HOME/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
   export PI05_PYTHON="$OPENPI_PY"
   RPENT_RUN_PI05_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m pytest \
     tests/e2e_tests/robocasa/test_pi05_retention.py -v --timeout=1200

This opt-in test loads real π0.5 weights, applies bounded actions and uses an
offline planner to test Harness tools and cleanup. Task success is not required,
and Qwen image/tool behavior is not covered. Initial loading and compilation may
take time.

7. One task: baseline, training, evaluation and summary
-------------------------------------------------------

Keep the same configuration throughout. Plan locks the formal protocol. Do not
change training/evaluation settings or source code while reusing its output.
For a two-step optimizer smoke run, create a separate config and output_root and
set its ``train_steps`` to 2; it does not establish learning.

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" plan
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     evaluate --checkpoint base --mode direct
   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/pilot1.json" train --all --resume
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" \
     evaluate --checkpoint all --mode direct
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot1.json" summarize

The default direct-only plan has one adapted model and 24 episodes including baseline.
Enabling both Qwen modes increases this to 72; summary exits 2 after direct alone
until step 9 completes. The official OpenPI
optimizer writes progress to the terminal; initial JAX compilation can precede
step output. On completion,
``runs/pilot1/training/full/OpenDrawer/seed_0/checkpoint.json`` records the final
weight path. With 1,000 training steps the final step directory is 999.
``--resume`` skips completed training or restores a saved incomplete run;
steps since the last save must be repeated.

Inspect ``summary/summary.json``, ``comparisons.csv`` and ``per_task.csv`` beneath
``runs/pilot1``. A pilot's ``complete=true`` refers to its declared subset;
``full_benchmark_complete`` remains false. Summary exit 2 means missing pairs,
which are not silently removed. Evaluation exit 1 indicates infrastructure errors:
inspect ``worker.log``, ``retention_result.json`` and service logs before retrying.
Valid robot failures remain completed outcomes and are not rerun until success.

8. Three-task pilots and eight-GPU joint50 training
---------------------------------------------------

Initialize a three-task single-GPU pilot, then repeat steps 5–7 with its config:

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/pilot3-joint.json" \
     init --preset pilot3-joint --workspace "$EXP_HOME"

Use ``pilot3-independent`` for three specialists. Each stage starts from Human300,
not a pilot1 checkpoint. On the eight-GPU server, transfer source/data, recreate
the environments and generate config paths for that machine.
Eight GPUs **cooperate on one joint model**; do not launch eight processes for
the same training ID:

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" \
     init --preset joint50 --workspace "$EXP_HOME" --fsdp-devices 8
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" download-data --execute
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/joint50.json" doctor --stage train --runtime
   "$SIM_PY" -m robots.robocasa.retention --config "$EXP_HOME/configs/joint50.json" plan
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$SIM_PY" -m robots.robocasa.retention \
     --config "$EXP_HOME/configs/joint50.json" train --all --resume

``fsdp_devices=8``; batch 16 is global, not per GPU. Visible GPU count must be
divisible by fsdp_devices, and global batch must be divisible by visible GPUs.
This preset includes direct and both Qwen modes; change to ``modes: ["direct"]``
before plan for policy-only work. Stop planners on training GPUs during training.
The default plan has one adapted model and 21,000 episodes. Baseline can run before
or after training under the same protocol.

Use ``init --preset independent50`` with distinct config/output paths for the
final specialist study. ``train --all`` runs those 50 adaptations sequentially
from base; it does not automatically schedule parallel training jobs.

9. Add Qwen3-VL-4B and Qwen3.5-4B
---------------------------------

Ensure these modes were included before plan. If step 4 used the default direct-only
config, create a new agent experiment below. It reuses downloaded data but trains
and evaluates again under the new protocol. If you used ``--with-planners`` in
step 4, point AGENT_CONFIG at the original pilot1.json and skip this creation,
training and direct-evaluation block.

.. code-block:: bash

   export AGENT_CONFIG="$EXP_HOME/configs/pilot1-agent.json"
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" \
     init --preset pilot1 --workspace "$EXP_HOME" --with-planners \
     --output-root "$EXP_HOME/runs/pilot1-agent"
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" plan
   CUDA_VISIBLE_DEVICES=0 "$SIM_PY" -m robots.robocasa.retention \
     --config "$AGENT_CONFIG" train --all --resume
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode direct

Install the separate Qwen environment once, without adding quantization as another
experimental variable:

.. code-block:: bash

   uv venv "$EXP_HOME/qwen-env" --python 3.12
   uv pip install --python "$QWEN_PY" -e "$RPENT_DIR" vllm --torch-backend auto
   uv pip check --python "$QWEN_PY"
   hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
   hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

Omitting local-dir puts both models in the shared HF cache, matching serving by
pinned model ID/revision. Use the same HF_HOME in all terminals. Start one server
in the foreground:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$QWEN_PY" -m robots.robocasa.retention.serve_planner \
     --config "$AGENT_CONFIG" --mode qwen3_vl_4b

In a second terminal, load the same exports and return to the RPent root:

.. code-block:: bash

   export OPENAI_API_KEY=local
   export AGENT_CONFIG="$EXP_HOME/configs/pilot1-agent.json"
   "$EXP_HOME/sim-env/bin/rpent-check-llm" --planner api \
     --model openai-chat:Qwen/Qwen3-VL-4B-Instruct --base-url http://127.0.0.1:8000/v1
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" \
     evaluate --checkpoint base --mode qwen3_vl_4b --tasks OpenDrawer --max-cells 1

The connectivity check is text-only. Inspect the real episode's agent trace for
image inputs, tool calls and actions. Execution limits do not shrink statistical
denominators. Once checked, complete that mode:

.. code-block:: bash

   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode qwen3_vl_4b

Stop the first server with Ctrl-C in its terminal, then start the second:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 "$QWEN_PY" -m robots.robocasa.retention.serve_planner \
     --config "$AGENT_CONFIG" --mode qwen35_4b

In the evaluation terminal:

.. code-block:: bash

   "$EXP_HOME/sim-env/bin/rpent-check-llm" --planner api \
     --model openai-chat:Qwen/Qwen3.5-4B --base-url http://127.0.0.1:8001/v1
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" evaluate --checkpoint all --mode qwen35_4b
   "$SIM_PY" -m robots.robocasa.retention --config "$AGENT_CONFIG" summarize

Generated configs use ``gpu_memory_utilization=0.4``, ``max_num_seqs=1`` and
``max_model_len=32768`` for both planners as a single-A800 starting point.
π0.5 inference disables JAX preallocation, but planner, VLA and simulator still
coexist during each agent episode; measure peak memory. Long traces may exceed
the context limit. Adjust context, memory and episode budgets during pilot development,
before the final protocol, rather than changing formal budgets to hide failures.

10. Shard evaluation across eight GPUs
--------------------------------------

One layout uses GPUs 0–5 for six env+π0.5 workers and GPUs 6/7 for the Qwen services.
Stop training and start planners with the joint50 config on GPUs 6/7 in separate
terminals. Create six configs differing only in device assignment, which does
not affect protocol_id:

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

Run plan before workers and never duplicate a running shard. Workers share
read-only checkpoints and planner services, with private VLA servers and episode
memories. Planner throughput can limit speed; six workers do not imply 6× speedup.
Retries skip completed cells. Resolve stale ``.running`` locks only after their
original processes have stopped.

11. Preserve versions and troubleshoot
--------------------------------------

Finalize software before plan; preserve these records and a full source snapshot:

.. code-block:: bash

   uv pip freeze --python "$SIM_PY" > "$EXP_HOME/env/simulator.txt"
   uv pip freeze --python "$OPENPI_PY" > "$EXP_HOME/env/openpi.txt"
   # If the Qwen environment is installed
   uv pip freeze --python "$QWEN_PY" > "$EXP_HOME/env/qwen.txt"
   git -C "$OPENPI_DIR" rev-parse HEAD > "$EXP_HOME/env/openpi-revision.txt"
   git -C "$RPENT_DIR" rev-parse HEAD > "$EXP_HOME/env/rpent-head.txt"
   git -C "$RPENT_DIR" status --short > "$EXP_HOME/env/rpent-working-tree.txt"

HEAD and pip freeze do not capture unpublished source changes. Keep those files
as well. The plan includes a source/configuration fingerprint; planner launch
records include actual library versions and command arguments.

.. list-table:: Troubleshooting
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Check
   * - OpenPI revision/import mismatch
     - Select the pinned benchmark fork and its Python, not the LIBERO environment.
   * - CPU-only JAX or Torch CUDA=false
     - Check driver, visible devices and actual interpreter; repair dependencies before training.
   * - Missing normalization
     - Point to 75000 and download its assets completely; do not recompute from old data.
   * - CUDA OOM
     - Stop other services; adjust global batch/LoRA/FSDP in a new training protocol, or planner memory/context before formal evaluation.
   * - EGL/navview/object errors
     - Check external assets, MUJOCO_GL, the RPent robosuite branch and real component test.
   * - Output belongs to another protocol
     - Choose a fresh output_root; do not delete plan to bypass identity checks.
   * - Summary exit 2
     - Inspect missing_pairs; complete every declared checkpoint, mode and task and repair infrastructure errors.

Offline tests cover preparation, download boundaries, failure reporting and
statistics. A800 installation, real training and robot evaluation were not run
on the development host; passing doctor does not establish task success.
References: `pinned OpenPI <https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397>`_,
`RoboCasa data <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`_,
`uv installation <https://docs.astral.sh/uv/getting-started/installation/>`_ and
`vLLM Qwen serving <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>`_.
