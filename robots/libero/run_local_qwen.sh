#!/usr/bin/env bash
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run from the repo root: bash robots/libero/run_local_qwen.sh /absolute/output [eval flags]
set -euo pipefail

if [[ $# -lt 1 || "$1" != /* ]]; then
  echo "Usage: bash robots/libero/run_local_qwen.sh /absolute/new-output [continual_eval flags]" >&2
  exit 2
fi
output_root=$1
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

PI05_CHECKPOINT_PATH=${PI05_CHECKPOINT_PATH:-/home/ma-user/work/model/xiaoyi_tmpstorage/hyy_files/rpent/RLinf-Pi05-LIBERO-130-fullshot-SFT}
SAM3_MODEL_DIR=${SAM3_MODEL_DIR:-/home/ma-user/work/model/xiaoyi_tmpstorage/hyy_files/rpent/sam3}
SAM3_CHECKPOINT_PATH=${SAM3_CHECKPOINT_PATH:-$SAM3_MODEL_DIR/sam3.pt}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/home/ma-user/work/model/xiaoyi_tmpstorage/hyy_files/rpent/Qwen3.5-27B}
QWEN_SERVED_MODEL_NAME=${QWEN_SERVED_MODEL_NAME:-Qwen3.5-27B}
QWEN_GPUS=${QWEN_GPUS:-0,1,2,3}
QWEN_TP=${QWEN_TP:-4}
QWEN_MAX_MODEL_LEN=${QWEN_MAX_MODEL_LEN:-65536}
VLLM_BIN=${VLLM_BIN:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/vllm/bin/vllm}
[[ -x "$VLLM_BIN" ]] || { echo "vllm binary missing: $VLLM_BIN" >&2; exit 2; }
PI05_GPU=${PI05_GPU:-4}
SAM3_GPU=${SAM3_GPU:-5}
LIBERO_GPU=${LIBERO_GPU:-6}

[[ -d "$PI05_CHECKPOINT_PATH" ]] || { echo "Pi0.5 directory missing: $PI05_CHECKPOINT_PATH" >&2; exit 2; }
[[ -f "$SAM3_CHECKPOINT_PATH" ]] || { echo "SAM3 checkpoint missing: $SAM3_CHECKPOINT_PATH" >&2; exit 2; }
[[ -d "$QWEN_MODEL_PATH" ]] || { echo "Qwen directory missing: $QWEN_MODEL_PATH" >&2; exit 2; }
for tool in python curl; do
  command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 2; }
done
IFS=, read -r -a qwen_gpu_list <<< "$QWEN_GPUS"
[[ ${#qwen_gpu_list[@]} -eq $QWEN_TP ]] || { echo "QWEN_TP must equal the number of QWEN_GPUS" >&2; exit 2; }
for gpu in "${qwen_gpu_list[@]}"; do
  [[ "$gpu" != "$PI05_GPU" && "$gpu" != "$SAM3_GPU" && "$gpu" != "$LIBERO_GPU" ]] || {
    echo "Qwen and robot services must use separate GPUs" >&2; exit 2;
  }
done
[[ "$PI05_GPU" != "$SAM3_GPU" && "$PI05_GPU" != "$LIBERO_GPU" && "$SAM3_GPU" != "$LIBERO_GPU" ]] || {
  echo "Pi0.5, SAM3, and LIBERO must use separate GPUs" >&2; exit 2;
}

service_logs="${output_root}.services"
mkdir -p "$service_logs"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

export PI05_CHECKPOINT_PATH SAM3_CHECKPOINT_PATH
export LOCAL_API_KEY=${LOCAL_API_KEY:-EMPTY}
CUDA_VISIBLE_DEVICES="$QWEN_GPUS" "$VLLM_BIN" serve "$QWEN_MODEL_PATH" \
  --host 127.0.0.1 --port 8000 --served-model-name "$QWEN_SERVED_MODEL_NAME" \
  --tensor-parallel-size "$QWEN_TP" --max-model-len "$QWEN_MAX_MODEL_LEN" \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder >"$service_logs/qwen.log" 2>&1 &
qwen_pid=$!
pids+=("$qwen_pid")

python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --host 127.0.0.1 --port 8220 --cuda-device "$PI05_GPU" \
  >"$service_logs/pi05.log" 2>&1 &
pids+=("$!")
python -m rpent.robots.components.sam3_server \
  --host 127.0.0.1 --port 8114 --cuda-device "$SAM3_GPU" \
  >"$service_logs/sam3.log" 2>&1 &
pids+=("$!")

ready=0
for ((attempt=1; attempt<=180; attempt++)); do
  if ! kill -0 "$qwen_pid" 2>/dev/null; then
    echo "Qwen service exited; inspect $service_logs/qwen.log" >&2
    exit 1
  fi
  if curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "Qwen did not become ready; inspect $service_logs/qwen.log" >&2; exit 1; }

python robots/libero/continual_eval.py --output-root "$output_root" \
  --model "openai-chat:$QWEN_SERVED_MODEL_NAME" --base-url http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_API_KEY --cuda-device "$LIBERO_GPU" \
  --vla-endpoint http://127.0.0.1:8220 --sam3-endpoint http://127.0.0.1:8114 "$@"
